"""Arm C: ResNet-50 End-to-End Direct Ergonomic Scoring Pipeline.

Uses ImageNet-pretrained ResNet-50 backbone fused with non-visual ErgoConfig features
to predict RULA/REBA grand scores via multi-task regression and risk classification heads.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple
import os
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights, resnet50
import torchvision.transforms.functional as F

from models.base import BaseErgoModel, PosturePrediction
from rula_reba.rules import (
    ErgoConfig,
    RULAResult,
    REBAResult,
    rula_action_level,
    reba_action_level,
)
from rula_reba.counterfactual import suggest_minimal_posture_fix
from rula_reba.angles import COCO_KEYPOINTS


class ResNet50ErgoMultiTask(nn.Module):
    """ResNet-50 backbone with multi-task posture regression and risk classification heads."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        base = resnet50(weights=weights)

        # Image feature extractor (up to avgpool, 2048 dims)
        self.conv1 = base.conv1
        self.bn1 = base.bn1
        self.relu = base.relu
        self.maxpool = base.maxpool
        self.layer1 = base.layer1
        self.layer2 = base.layer2
        self.layer3 = base.layer3
        self.layer4 = base.layer4
        self.avgpool = base.avgpool

        # Non-visual config embedding MLP (5 dims -> 64 dims)
        self.config_mlp = nn.Sequential(
            nn.Linear(5, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
        )

        # Fused latent representation (2048 + 64 = 2112 dims)
        self.fusion = nn.Sequential(
            nn.Linear(2048 + 64, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
        )

        # RULA Heads: Regression (1) + 4-Class Classification (4)
        self.rula_reg_head = nn.Linear(512, 1)
        self.rula_cls_head = nn.Linear(512, 4)

        # REBA Heads: Regression (1) + 5-Class Classification (5)
        self.reba_reg_head = nn.Linear(512, 1)
        self.reba_cls_head = nn.Linear(512, 5)

    def forward(self, x: torch.Tensor, config_vec: torch.Tensor) -> dict[str, torch.Tensor]:
        # Image features
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        img_feat = torch.flatten(out, 1)  # (B, 2048)

        # Config features
        cfg_feat = self.config_mlp(config_vec)  # (B, 64)

        # Fused features
        fused = self.fusion(torch.cat([img_feat, cfg_feat], dim=1))  # (B, 512)

        return {
            "rula_score": self.rula_reg_head(fused).squeeze(-1),
            "rula_logits": self.rula_cls_head(fused),
            "reba_score": self.reba_reg_head(fused).squeeze(-1),
            "reba_logits": self.reba_cls_head(fused),
        }


class ArmCResNetE2E(BaseErgoModel):
    """Arm C Pipeline: End-to-End Multitask ResNet-50 Ergonomic Scoring."""

    def __init__(
        self,
        weights_path: Optional[str] = None,
        pretrained: bool = True,
        device: str = "cpu",
    ):
        super().__init__(model_name="Arm C (ResNet-50 End-to-End)", device=device)
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")
        self.network = ResNet50ErgoMultiTask(pretrained=pretrained)

        if weights_path and os.path.exists(weights_path):
            state = torch.load(weights_path, map_location=self.device)
            self.network.load_state_dict(state)

        self.network.to(self.device)
        self.network.eval()

    def detect_keypoints(self, frame: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Arm C is direct-to-score, returning placeholder keypoints for interface compliance."""
        return np.zeros((17, 3), dtype=np.float32), list(COCO_KEYPOINTS)

    def predict(
        self,
        frame: np.ndarray,
        config: Optional[ErgoConfig] = None,
        view_plane: Any = None,
    ) -> PosturePrediction:
        """Run direct-to-score multi-task inference."""
        import time
        t0 = time.perf_counter()

        cfg = config or ErgoConfig()

        # Format image
        if frame.ndim == 3 and frame.shape[2] == 3:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        resized = cv2.resize(rgb, (224, 224))
        tensor = F.to_tensor(resized).unsqueeze(0).to(self.device)
        # Normalize with standard ImageNet statistics
        tensor = F.normalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

        # Format non-visual config features
        # [load_norm, muscle_use, twist_norm, coupling_norm, activity_norm]
        cfg_vec = torch.tensor([[
            min(cfg.rula.load_force_score / 3.0, 1.0),
            1.0 if cfg.rula.muscle_use else 0.0,
            float(cfg.rula.wrist_twist - 1),
            cfg.reba.coupling_score / 3.0,
            1.0 if (cfg.reba.activity_static_held or cfg.reba.activity_repeated) else 0.0,
        ]], dtype=torch.float32, device=self.device)

        with torch.no_grad():
            preds = self.network(tensor, cfg_vec)

        # Extract predicted scores
        rula_cont = float(preds["rula_score"].item())
        reba_cont = float(preds["reba_score"].item())
        rula_cls = int(torch.argmax(preds["rula_logits"], dim=1).item())
        reba_cls = int(torch.argmax(preds["reba_logits"], dim=1).item())

        # Map to valid ergonomic bounds
        # If model is un-finetuned, fallback to reasonable prior from classification or default
        rula_score = int(np.clip(np.round(rula_cont), 1, 7)) if abs(rula_cont) < 20 else (rula_cls + 1)
        reba_score = int(np.clip(np.round(reba_cont), 1, 15)) if abs(reba_cont) < 30 else (reba_cls * 2 + 1)

        rula_act, rula_risk, rula_rec = rula_action_level(rula_score)
        reba_act, reba_risk, reba_rec = reba_action_level(reba_score)

        rula_res = RULAResult(
            upper_arm_score=2, lower_arm_score=1, wrist_score=1, wrist_twist_score=cfg.rula.wrist_twist,
            table_a_score=2, muscle_use_score=1 if cfg.rula.muscle_use else 0, load_force_score=cfg.rula.load_force_score,
            score_a=min(8, 2 + cfg.rula.load_force_score), neck_score=2, trunk_score=2, legs_score=1,
            table_b_score=2, score_b=min(7, 2 + cfg.rula.load_force_score),
            grand_score=rula_score, action_level=rula_act, risk_level=rula_risk, action_recommendation=rula_rec,
            details={"e2e_continuous_rula": round(rula_cont, 2), "e2e_class": rula_cls},
        )

        reba_res = REBAResult(
            trunk_score=2, neck_score=1, legs_score=1, table_a_score=2, load_score=cfg.reba.load_score,
            score_a=min(12, 2 + cfg.reba.load_score), upper_arm_score=2, lower_arm_score=1, wrist_score=1,
            table_b_score=2, coupling_score=cfg.reba.coupling_score, score_b=min(12, 2 + cfg.reba.coupling_score),
            score_c=max(1, reba_score - 1), activity_score=1,
            grand_score=reba_score, action_level=reba_act, risk_level=reba_risk, action_recommendation=reba_rec,
            details={"e2e_continuous_reba": round(reba_cont, 2), "e2e_class": reba_cls},
        )

        dummy_angles = {
            "trunk_flexion": 15.0, "neck_flexion": 10.0, "upper_arm_flexion": 25.0,
            "lower_arm_flexion": 90.0, "wrist_flexion": 5.0,
        }
        fix_res = suggest_minimal_posture_fix(dummy_angles, rula_config=cfg.rula, reba_config=cfg.reba)

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return PosturePrediction(
            pipeline_name=self.model_name,
            keypoints=np.zeros((17, 3), dtype=np.float32),
            keypoint_names=list(COCO_KEYPOINTS),
            joint_angles=dummy_angles,
            angle_confidences={k: 0.8 for k in dummy_angles},
            view_plane="sagittal_right",
            dominant_side="right",
            rula_result=rula_res,
            reba_result=reba_res,
            counterfactual_fix=fix_res,
            inference_time_ms=round(latency_ms, 2),
            projection_limits={
                "arm_c_end_to_end": "Arm C operates directly on visual appearance and task configuration without intermediate 2D angle projection."
            },
            raw_model_output=preds,
        )

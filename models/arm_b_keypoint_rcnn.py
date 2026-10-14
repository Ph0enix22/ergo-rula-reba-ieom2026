"""Arm B: ResNet-50 Keypoint Detector Pipeline.

Uses TorchVision Keypoint R-CNN with ResNet-50-FPN backbone (COCO pretrained)
to detect 17 anatomical keypoints, compute joint angles, and evaluate RULA/REBA rules.
"""

from __future__ import annotations

from typing import List, Optional, Tuple
import cv2
import numpy as np
import torch
import torchvision.transforms.functional as F
from torchvision.models.detection import (
    KeypointRCNN_ResNet50_FPN_Weights,
    keypointrcnn_resnet50_fpn,
)

from models.base import BaseErgoModel
from rula_reba.angles import COCO_KEYPOINTS


class ArmBKeypointRCNN(BaseErgoModel):
    """Arm B Pipeline using TorchVision Keypoint R-CNN ResNet-50 FPN."""

    def __init__(
        self,
        weights: Optional[KeypointRCNN_ResNet50_FPN_Weights] = KeypointRCNN_ResNet50_FPN_Weights.DEFAULT,
        score_threshold: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__(model_name="Arm B (ResNet-50 Keypoint R-CNN)", device=device)
        self.score_threshold = score_threshold
        self.device = torch.device(device if torch.cuda.is_available() and device == "cuda" else "cpu")

        # Load Keypoint R-CNN with ResNet-50-FPN backbone
        if weights is None:
            self.model = keypointrcnn_resnet50_fpn(weights=None, weights_backbone=None)
        else:
            self.model = keypointrcnn_resnet50_fpn(weights=weights)
        self.model.to(self.device)
        self.model.eval()

    def detect_keypoints(self, frame: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Run Keypoint R-CNN inference on BGR or RGB frame.

        Returns:
        - keypoints: (17, 3) array [x, y, confidence]
        - names: list of 17 COCO keypoint names
        """
        # Convert BGR to RGB
        if frame.ndim == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        # Convert to Tensor (C, H, W) in [0.0, 1.0]
        tensor = F.to_tensor(rgb_frame).to(self.device)

        with torch.no_grad():
            outputs = self.model([tensor])

        if not outputs or len(outputs) == 0:
            return np.zeros((17, 3), dtype=np.float32), list(COCO_KEYPOINTS)

        output = outputs[0]
        scores = output["scores"].cpu().numpy()
        keypoints = output["keypoints"].cpu().numpy()  # (N, 17, 3)
        kp_scores = output.get("keypoints_scores")

        # Filter for highest scoring person detection above threshold
        valid_indices = np.where(scores >= self.score_threshold)[0]
        if len(valid_indices) == 0:
            # If no detection above threshold, check if any person was detected at all
            if len(scores) > 0:
                best_idx = int(np.argmax(scores))
            else:
                return np.zeros((17, 3), dtype=np.float32), list(COCO_KEYPOINTS)
        else:
            best_idx = int(valid_indices[0])

        best_kps = keypoints[best_idx].copy()  # (17, 3)

        # Ensure confidence column contains normalized probability
        if kp_scores is not None and len(kp_scores) > best_idx:
            kps_conf = kp_scores[best_idx].cpu().numpy()
            best_kps[:, 2] = kps_conf
        else:
            # Visibility column in COCO format: 0=unlabeled, 1=labeled not visible, 2=labeled visible
            best_kps[:, 2] = np.clip(best_kps[:, 2] / 2.0, 0.0, 1.0)

        return best_kps.astype(np.float32), list(COCO_KEYPOINTS)

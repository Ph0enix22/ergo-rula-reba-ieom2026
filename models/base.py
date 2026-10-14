"""Unified Base Interface and Visualization for All Ergonomic Assessment Pipelines.

Defines the common interface predict(frame) -> PosturePrediction and shared visualization logic.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional
import time
import cv2
import numpy as np

from rula_reba.rules import (
    ErgoConfig,
    RULAResult,
    REBAResult,
    calculate_rula_from_angles,
    calculate_reba_from_angles,
)
from rula_reba.angles import (
    JointAngleResult,
    ViewPlane,
    extract_joint_angles,
)
from rula_reba.counterfactual import suggest_minimal_posture_fix


# Standard skeleton connections for 17-keypoint COCO format
COCO_SKELETON_EDGES = [
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
]

# Action level risk color maps (BGR for OpenCV)
RISK_COLORS_BGR = {
    1: (46, 204, 113),   # Green (Acceptable / Negligible)
    2: (46, 204, 113),   # Green / Light Green
    3: (241, 196, 15),   # Yellow / Amber (Low)
    4: (241, 196, 15),
    5: (230, 126, 34),   # Orange (Medium)
    6: (230, 126, 34),
    7: (231, 76, 60),    # Red (High / Very High)
}


@dataclass
class PosturePrediction:
    """Standardized output produced across all three pipeline arms."""
    pipeline_name: str
    keypoints: np.ndarray                  # (K, 3) [x, y, conf]
    keypoint_names: list[str]
    joint_angles: dict[str, float]
    angle_confidences: dict[str, float]
    view_plane: str
    dominant_side: str
    rula_result: RULAResult
    reba_result: REBAResult
    counterfactual_fix: dict[str, Any]
    inference_time_ms: float
    projection_limits: dict[str, str] = field(default_factory=dict)
    raw_model_output: Optional[Any] = None


class BaseErgoModel(ABC):
    """Abstract base class for all ergonomic posture assessment arms."""

    def __init__(self, model_name: str, device: str = "cpu"):
        self.model_name = model_name
        self.device = device

    @abstractmethod
    def detect_keypoints(self, frame: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Extract 2D keypoints [x, y, confidence] from RGB or BGR frame."""
        pass

    def predict(
        self,
        frame: np.ndarray,
        config: Optional[ErgoConfig] = None,
        view_plane: ViewPlane = ViewPlane.AUTO,
    ) -> PosturePrediction:
        """Run end-to-end posture scoring pipeline on frame."""
        t0 = time.perf_counter()

        cfg = config or ErgoConfig()
        keypoints, names = self.detect_keypoints(frame)

        # Build name -> [x, y, conf] mapping
        kp_dict = {
            name: keypoints[i] for i, name in enumerate(names) if i < len(keypoints)
        }

        # 1. Geometric joint angles
        angle_res = extract_joint_angles(kp_dict, view_plane=view_plane)

        # 2. Rule evaluation
        rula_res = calculate_rula_from_angles(angle_res.angles, cfg.rula)
        reba_res = calculate_reba_from_angles(angle_res.angles, cfg.reba)

        # 3. Counterfactual XAI posture fix (Paper ID 283)
        fix_res = suggest_minimal_posture_fix(
            angle_res.angles,
            rula_config=cfg.rula,
            reba_config=cfg.reba,
            target_action_level=2,
        )

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return PosturePrediction(
            pipeline_name=self.model_name,
            keypoints=keypoints,
            keypoint_names=names,
            joint_angles=angle_res.angles,
            angle_confidences=angle_res.confidences,
            view_plane=angle_res.view_plane.value,
            dominant_side=angle_res.dominant_side,
            rula_result=rula_res,
            reba_result=reba_res,
            counterfactual_fix=fix_res,
            inference_time_ms=round(latency_ms, 2),
            projection_limits=angle_res.projection_limits,
        )

    def draw_skeleton_overlay(
        self,
        frame: np.ndarray,
        pred: PosturePrediction,
        blur_face: bool = False,
    ) -> np.ndarray:
        """Render ergonomic skeleton overlay, angle indicators, and privacy face blur."""
        out = frame.copy()
        h, w = out.shape[:2]

        kp_dict = {
            name: pred.keypoints[i]
            for i, name in enumerate(pred.keypoint_names)
            if i < len(pred.keypoints)
        }

        # Privacy Face Blur (100% local OpenCV Gaussian Blur)
        if blur_face:
            face_pts = [
                kp_dict[k][:2]
                for k in ["nose", "left_eye", "right_eye", "left_ear", "right_ear"]
                if k in kp_dict and kp_dict[k][2] > 0.1
            ]
            if len(face_pts) >= 2:
                pts = np.array(face_pts)
                min_x = max(0, int(np.min(pts[:, 0]) - 30))
                max_x = min(w, int(np.max(pts[:, 0]) + 30))
                min_y = max(0, int(np.min(pts[:, 1]) - 40))
                max_y = min(h, int(np.max(pts[:, 1]) + 40))

                if max_x > min_x and max_y > min_y:
                    roi = out[min_y:max_y, min_x:max_x]
                    # Heavy Gaussian blur kernel
                    ksize = (max(3, (roi.shape[1] // 3) * 2 + 1), max(3, (roi.shape[0] // 3) * 2 + 1))
                    blurred_roi = cv2.GaussianBlur(roi, ksize, 30)
                    out[min_y:max_y, min_x:max_x] = blurred_roi

        # Risk color determined by worst score
        rula_color = RISK_COLORS_BGR.get(pred.rula_result.grand_score, (0, 255, 0))

        # Draw skeleton connections
        for p1_name, p2_name in COCO_SKELETON_EDGES:
            if p1_name in kp_dict and p2_name in kp_dict:
                p1 = kp_dict[p1_name]
                p2 = kp_dict[p2_name]
                if p1[2] > 0.2 and p2[2] > 0.2:
                    pt1 = (int(p1[0]), int(p1[1]))
                    pt2 = (int(p2[0]), int(p2[1]))
                    cv2.line(out, pt1, pt2, rula_color, 3, cv2.LINE_AA)

        # Draw joint nodes
        for name, pt in kp_dict.items():
            if pt[2] > 0.2:
                cv2.circle(out, (int(pt[0]), int(pt[1])), 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(out, (int(pt[0]), int(pt[1])), 6, (0, 0, 0), 1, cv2.LINE_AA)

        # Draw HUD badge with scores
        badge_h = 95
        badge_w = 340
        overlay = out.copy()
        cv2.rectangle(overlay, (15, 15), (15 + badge_w, 15 + badge_h), (20, 24, 33), -1)
        cv2.addWeighted(overlay, 0.82, out, 0.18, 0, out)
        cv2.rectangle(out, (15, 15), (15 + badge_w, 15 + badge_h), rula_color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        rula_txt = f"RULA: {pred.rula_result.grand_score}/7 (L{pred.rula_result.action_level})"
        reba_txt = f"REBA: {pred.reba_result.grand_score}/15 (L{pred.reba_result.action_level})"
        fps_txt = f"{pred.pipeline_name} | {pred.inference_time_ms:.1f}ms"

        cv2.putText(out, rula_txt, (25, 45), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, reba_txt, (25, 75), font, 0.65, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(out, fps_txt, (25, 98), font, 0.42, (180, 180, 180), 1, cv2.LINE_AA)

        return out

    def draw_overlay(
        self,
        frame: np.ndarray,
        pred: PosturePrediction,
        blur_faces: bool = False,
        blur_face: bool = False,
    ) -> np.ndarray:
        """Alias for draw_skeleton_overlay for cross-model compatibility."""
        return self.draw_skeleton_overlay(frame, pred, blur_face=blur_faces or blur_face)

"""Arm A (Comparative Flag): Ultralytics YOLOv8-Pose Pipeline.

Extracts COCO 17 keypoints for side-by-side comparison with MediaPipe Pose,
addressing peer-review requests to evaluate multiple open-source pose estimation engines.
"""

from __future__ import annotations

import os
from typing import Tuple
import numpy as np
import cv2
from ultralytics import YOLO

from models.base import BaseErgoModel
from rula_reba.angles import COCO_KEYPOINTS


class ArmAYOLOv8(BaseErgoModel):
    """Arm A Comparative Pipeline using Ultralytics YOLOv8-Pose (17 keypoints)."""

    def __init__(
        self,
        weights_path: str = "yolov8n-pose.pt",
        confidence_threshold: float = 0.25,
        device: str = "cpu",
    ):
        super().__init__(model_name="Arm A (YOLOv8-Pose)", device=device)
        self.weights_path = weights_path
        self.confidence_threshold = confidence_threshold
        self.model = YOLO(self.weights_path)

    def detect_keypoints(self, frame: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Run YOLOv8-pose inference on frame.

        Returns:
        - keypoints: (17, 3) array [x, y, confidence] in pixel space
        - names: list of 17 COCO keypoint names
        """
        # Guard against blank/all-zero frames causing Linux Python 3.13 autobackend FP exception
        if frame is None or frame.size == 0 or frame.max() == 0:
            return np.zeros((17, 3), dtype=np.float32), list(COCO_KEYPOINTS)

        dev = "cpu" if str(self.device).lower() == "cpu" else self.device
        results = self.model(
            frame,
            device=dev,
            verbose=False,
            half=False,
            conf=self.confidence_threshold,
        )

        if not results or len(results) == 0:
            return np.zeros((17, 3), dtype=np.float32), list(COCO_KEYPOINTS)

        res = results[0]
        if res.keypoints is None or res.keypoints.data is None or len(res.keypoints.data) == 0:
            return np.zeros((17, 3), dtype=np.float32), list(COCO_KEYPOINTS)

        # Get first detected person's keypoints
        # Shape: (17, 3) or (17, 2)
        kp_tensor = res.keypoints.data[0].cpu().numpy()

        if kp_tensor.shape[1] == 2:
            conf = np.ones((17, 1), dtype=np.float32)
            kp_tensor = np.hstack([kp_tensor, conf])

        return kp_tensor.astype(np.float32), list(COCO_KEYPOINTS)

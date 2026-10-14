"""Arm A: MediaPipe Pose Pipeline for Ergonomic Posture Assessment.

Uses Google MediaPipe PoseLandmarker (33 landmarks, optimized CPU XNNPACK delegate)
to extract 2D landmarks, evaluate joint angles, and compute official RULA/REBA scores.
"""

from __future__ import annotations

import os
import urllib.request
from typing import Optional, Tuple
import cv2
import numpy as np

import mediapipe as mp
from mediapipe.tasks.python import BaseOptions
from mediapipe.tasks.python.vision import (
    PoseLandmarker,
    PoseLandmarkerOptions,
    RunningMode,
)

from models.base import BaseErgoModel
from rula_reba.angles import MEDIAPIPE_KEYPOINTS_MAP

DEFAULT_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    "pose_landmarker_lite/float16/latest/pose_landmarker_lite.task"
)
DEFAULT_MODEL_PATH = os.path.join("models", "weights", "pose_landmarker_lite.task")


class ArmAMediaPipe(BaseErgoModel):
    """Arm A Pipeline using MediaPipe Pose (33 keypoints)."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL_PATH,
        min_detection_confidence: float = 0.5,
        min_presence_confidence: float = 0.5,
        device: str = "cpu",
    ):
        super().__init__(model_name="Arm A (MediaPipe Pose)", device=device)
        self.model_path = model_path
        self._ensure_model_exists()

        base_options = BaseOptions(model_asset_path=self.model_path)
        options = PoseLandmarkerOptions(
            base_options=base_options,
            running_mode=RunningMode.IMAGE,
            min_pose_detection_confidence=min_detection_confidence,
            min_pose_presence_confidence=min_presence_confidence,
            num_poses=1,
        )
        self.landmarker = PoseLandmarker.create_from_options(options)

    def _ensure_model_exists(self) -> None:
        """Download model asset if not already present on disk."""
        if not os.path.exists(self.model_path):
            os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
            urllib.request.urlretrieve(DEFAULT_MODEL_URL, self.model_path)

    def detect_keypoints(self, frame: np.ndarray) -> tuple[np.ndarray, list[str]]:
        """Run landmark detection on frame (BGR or RGB).

        Returns:
        - keypoints: (33, 3) array [x, y, confidence] in pixel space
        - names: list of landmark names
        """
        h, w = frame.shape[:2]

        # Convert to RGB for MediaPipe
        if frame.ndim == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        detection_result = self.landmarker.detect(mp_image)

        names = [MEDIAPIPE_KEYPOINTS_MAP.get(i, f"landmark_{i}") for i in range(33)]

        if not detection_result.pose_landmarks or len(detection_result.pose_landmarks) == 0:
            # Fallback: No person detected -> return zeros with 0 confidence
            return np.zeros((33, 3), dtype=np.float32), names

        first_pose = detection_result.pose_landmarks[0]
        kps = np.zeros((33, 3), dtype=np.float32)

        for i, lm in enumerate(first_pose):
            if i >= 33:
                break
            # Denormalize coordinates to image pixel dimensions
            px = lm.x * w
            py = lm.y * h
            # Visibility / presence score
            conf = getattr(lm, "visibility", 1.0)
            if conf is None or conf <= 0.0:
                conf = getattr(lm, "presence", 1.0) or 0.8
            kps[i] = [px, py, float(conf)]

        return kps, names

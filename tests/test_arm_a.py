"""Integration tests for Arm A pipelines: MediaPipe Pose and YOLOv8-Pose.

Verifies end-to-end prediction, rule engine integration, counterfactual fixes,
and privacy face blur rendering.
"""

import numpy as np
import cv2
import pytest
from models.arm_a_mediapipe import ArmAMediaPipe
from models.arm_a_yolov8 import ArmAYOLOv8
from rula_reba.rules import ErgoConfig, RULAConfig, REBAConfig


@pytest.fixture(scope="module")
def sample_blank_frame():
    """640x480 black image."""
    return np.zeros((480, 640, 3), dtype=np.uint8)


@pytest.fixture(scope="module")
def sample_synthetic_worker_frame():
    """Create a 640x480 frame with a synthetic worker stick figure in right profile."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 40
    # Head
    cv2.circle(frame, (320, 100), 25, (220, 220, 220), -1)
    # Torso
    cv2.line(frame, (320, 125), (320, 260), (220, 220, 220), 8)
    # Arm (forward reach ~90° elbow)
    cv2.line(frame, (320, 140), (320, 200), (220, 220, 220), 6)
    cv2.line(frame, (320, 200), (380, 200), (220, 220, 220), 6)
    # Legs
    cv2.line(frame, (320, 260), (320, 360), (220, 220, 220), 8)
    cv2.line(frame, (320, 360), (320, 460), (220, 220, 220), 8)
    return frame


class TestArmAMediaPipe:
    """Test MediaPipe Pose Arm A implementation."""

    @pytest.fixture(scope="module")
    def model(self):
        return ArmAMediaPipe()

    def test_blank_frame_no_detection_graceful(self, model, sample_blank_frame):
        pred = model.predict(sample_blank_frame)

        assert pred.pipeline_name == "Arm A (MediaPipe Pose)"
        assert pred.keypoints.shape[0] == 33
        assert 1 <= pred.rula_result.grand_score <= 7
        assert 1 <= pred.reba_result.grand_score <= 15
        assert isinstance(pred.counterfactual_fix, dict)
        assert pred.inference_time_ms >= 0.0

    def test_predict_synthetic_worker(self, model, sample_synthetic_worker_frame):
        cfg = ErgoConfig(
            rula=RULAConfig(muscle_use=True, load_force_score=1),
            reba=REBAConfig(load_score=1, coupling_score=1),
        )
        pred = model.predict(sample_synthetic_worker_frame, config=cfg)

        assert pred.rula_result.action_level in (1, 2, 3, 4)
        assert pred.reba_result.action_level in (0, 1, 2, 3, 4)
        assert "trunk_flexion" in pred.joint_angles
        assert "upper_arm_flexion" in pred.joint_angles
        assert "lower_arm_flexion" in pred.joint_angles

    def test_draw_skeleton_and_face_blur(self, model, sample_synthetic_worker_frame):
        pred = model.predict(sample_synthetic_worker_frame)
        overlay = model.draw_skeleton_overlay(sample_synthetic_worker_frame, pred, blur_face=True)

        assert overlay.shape == sample_synthetic_worker_frame.shape
        assert overlay.dtype == np.uint8


class TestArmAYOLOv8:
    """Test YOLOv8-Pose Arm A comparative implementation."""

    @pytest.fixture(scope="module")
    def model(self):
        return ArmAYOLOv8()

    def test_blank_frame_no_detection_graceful(self, model, sample_blank_frame):
        pred = model.predict(sample_blank_frame)

        assert pred.pipeline_name == "Arm A (YOLOv8-Pose)"
        assert pred.keypoints.shape[0] == 17
        assert 1 <= pred.rula_result.grand_score <= 7
        assert 1 <= pred.reba_result.grand_score <= 15
        assert pred.inference_time_ms >= 0.0

    def test_predict_synthetic_worker(self, model, sample_synthetic_worker_frame):
        pred = model.predict(sample_synthetic_worker_frame)

        assert pred.keypoints.shape == (17, 3)
        assert len(pred.keypoint_names) == 17
        assert 1 <= pred.rula_result.grand_score <= 7
        assert 1 <= pred.reba_result.grand_score <= 15

    def test_draw_skeleton(self, model, sample_synthetic_worker_frame):
        pred = model.predict(sample_synthetic_worker_frame)
        overlay = model.draw_skeleton_overlay(sample_synthetic_worker_frame, pred, blur_face=False)

        assert overlay.shape == sample_synthetic_worker_frame.shape

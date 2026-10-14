"""Integration tests for Arm B: ResNet-50 Keypoint R-CNN pipeline.

Verifies COCO 17-keypoint extraction, angle propagation, and rule engine integration.
"""

import numpy as np
import cv2
import pytest
from models.arm_b_keypoint_rcnn import ArmBKeypointRCNN
from rula_reba.rules import ErgoConfig, RULAConfig, REBAConfig


@pytest.fixture(scope="module")
def arm_b_model():
    """Instantiate Keypoint R-CNN on CPU once for test module."""
    return ArmBKeypointRCNN(weights=None, device="cpu")


@pytest.fixture(scope="module")
def sample_worker_frame():
    """640x480 frame containing a synthetic worker silhouette."""
    frame = np.ones((480, 640, 3), dtype=np.uint8) * 35
    # Head
    cv2.circle(frame, (320, 100), 25, (220, 220, 220), -1)
    # Torso
    cv2.line(frame, (320, 125), (320, 260), (220, 220, 220), 8)
    # Arm
    cv2.line(frame, (320, 140), (320, 200), (220, 220, 220), 6)
    cv2.line(frame, (320, 200), (380, 200), (220, 220, 220), 6)
    # Legs
    cv2.line(frame, (320, 260), (320, 360), (220, 220, 220), 8)
    cv2.line(frame, (320, 360), (320, 460), (220, 220, 220), 8)
    return frame


class TestArmBKeypointRCNN:
    """Verify Arm B behavior and model contract."""

    def test_model_initialization(self, arm_b_model):
        assert arm_b_model.model_name == "Arm B (ResNet-50 Keypoint R-CNN)"
        assert arm_b_model.device.type == "cpu"

    def test_blank_frame_no_detection_graceful(self, arm_b_model):
        blank = np.zeros((240, 320, 3), dtype=np.uint8)
        pred = arm_b_model.predict(blank)

        assert pred.keypoints.shape == (17, 3)
        assert len(pred.keypoint_names) == 17
        assert 1 <= pred.rula_result.grand_score <= 7
        assert 1 <= pred.reba_result.grand_score <= 15
        assert pred.inference_time_ms >= 0.0

    def test_predict_synthetic_worker(self, arm_b_model, sample_worker_frame):
        cfg = ErgoConfig(
            rula=RULAConfig(muscle_use=True, load_force_score=1),
            reba=REBAConfig(load_score=1, coupling_score=1),
        )
        pred = arm_b_model.predict(sample_worker_frame, config=cfg)

        assert pred.keypoints.shape == (17, 3)
        assert 1 <= pred.rula_result.grand_score <= 7
        assert 1 <= pred.reba_result.grand_score <= 15
        assert "trunk_flexion" in pred.joint_angles
        assert "upper_arm_flexion" in pred.joint_angles
        assert isinstance(pred.counterfactual_fix, dict)

    def test_draw_skeleton_overlay(self, arm_b_model, sample_worker_frame):
        pred = arm_b_model.predict(sample_worker_frame)
        overlay = arm_b_model.draw_skeleton_overlay(sample_worker_frame, pred, blur_face=True)

        assert overlay.shape == sample_worker_frame.shape
        assert overlay.dtype == np.uint8

"""Integration tests for Arm C: ResNet-50 End-to-End Direct Ergonomic Scoring.

Verifies multi-task network forward pass, config fusion, and BaseErgoModel compatibility.
"""

import numpy as np
import pytest
import torch
from models.arm_c_resnet_e2e import ArmCResNetE2E, ResNet50ErgoMultiTask
from rula_reba.rules import ErgoConfig, RULAConfig, REBAConfig


@pytest.fixture(scope="module")
def arm_c_model():
    """Instantiate Arm C on CPU once for test module."""
    return ArmCResNetE2E(pretrained=False, device="cpu")


class TestArmCResNetE2E:
    """Verify Arm C neural architecture and interface compliance."""

    def test_multitask_network_shapes(self):
        net = ResNet50ErgoMultiTask(pretrained=False)
        net.eval()

        dummy_img = torch.randn(2, 3, 224, 224)
        dummy_cfg = torch.randn(2, 5)

        with torch.no_grad():
            outputs = net(dummy_img, dummy_cfg)

        assert "rula_score" in outputs
        assert "rula_logits" in outputs
        assert "reba_score" in outputs
        assert "reba_logits" in outputs

        assert outputs["rula_score"].shape == (2,)
        assert outputs["rula_logits"].shape == (2, 4)
        assert outputs["reba_score"].shape == (2,)
        assert outputs["reba_logits"].shape == (2, 5)

    def test_arm_c_predict_interface(self, arm_c_model):
        frame = np.ones((480, 640, 3), dtype=np.uint8) * 80
        cfg = ErgoConfig(
            rula=RULAConfig(muscle_use=True, load_force_score=2),
            reba=REBAConfig(load_score=2, coupling_score=1),
        )

        pred = arm_c_model.predict(frame, config=cfg)

        assert pred.pipeline_name == "Arm C (ResNet-50 End-to-End)"
        assert 1 <= pred.rula_result.grand_score <= 7
        assert pred.rula_result.action_level in (1, 2, 3, 4)
        assert 1 <= pred.reba_result.grand_score <= 15
        assert pred.reba_result.action_level in (0, 1, 2, 3, 4)
        assert pred.inference_time_ms >= 0.0

    def test_draw_overlay(self, arm_c_model):
        frame = np.ones((200, 200, 3), dtype=np.uint8) * 100
        pred = arm_c_model.predict(frame)
        overlay = arm_c_model.draw_skeleton_overlay(frame, pred, blur_face=False)

        assert overlay.shape == frame.shape
        assert overlay.dtype == np.uint8

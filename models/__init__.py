"""Ergonomic Posture Scoring Models Package.

Exports BaseErgoModel and all three pipeline arms:
- Arm A: MediaPipe Pose and YOLOv8-Pose
- Arm B: ResNet-50 Keypoint R-CNN
- Arm C: ResNet-50 End-to-End Direct Scoring
"""

from models.base import BaseErgoModel, PosturePrediction
from models.arm_a_mediapipe import ArmAMediaPipe
from models.arm_a_yolov8 import ArmAYOLOv8
from models.arm_b_keypoint_rcnn import ArmBKeypointRCNN
from models.arm_c_resnet_e2e import ArmCResNetE2E, ResNet50ErgoMultiTask

__all__ = [
    "BaseErgoModel",
    "PosturePrediction",
    "ArmAMediaPipe",
    "ArmAYOLOv8",
    "ArmBKeypointRCNN",
    "ArmCResNetE2E",
    "ResNet50ErgoMultiTask",
]

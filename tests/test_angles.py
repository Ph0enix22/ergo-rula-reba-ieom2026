"""Unit tests for geometric 2D joint angle engine and view plane detection.

Validates trigonometric accuracy, confidence propagation, and projection limits.
"""

import math
import numpy as np
import pytest
from rula_reba.angles import (
    ViewPlane,
    compute_vector_angle_2d,
    compute_segment_angle_from_vertical,
    detect_view_plane,
    extract_joint_angles,
    normalize_keypoints,
)


class TestVectorMath:
    """Verify basic 2D vector trigonometry."""

    def test_orthogonal_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([0.0, 1.0])
        assert math.isclose(compute_vector_angle_2d(v1, v2), 90.0, abs_tol=1e-4)

    def test_parallel_vectors(self):
        v1 = np.array([2.0, 3.0])
        v2 = np.array([4.0, 6.0])
        assert math.isclose(compute_vector_angle_2d(v1, v2), 0.0, abs_tol=1e-4)

    def test_antiparallel_vectors(self):
        v1 = np.array([1.0, 0.0])
        v2 = np.array([-1.0, 0.0])
        assert math.isclose(compute_vector_angle_2d(v1, v2), 180.0, abs_tol=1e-4)

    def test_segment_from_vertical(self):
        # In image coords, upward vertical is (0, -1)
        p1 = np.array([100.0, 200.0])
        p2 = np.array([100.0, 100.0])  # Straight up
        assert math.isclose(compute_segment_angle_from_vertical(p1, p2), 0.0, abs_tol=1e-4)

        p3 = np.array([200.0, 200.0])  # Horizontal right (90° from vertical)
        assert math.isclose(compute_segment_angle_from_vertical(p1, p3), 90.0, abs_tol=1e-4)


class TestViewPlaneDetection:
    """Verify automatic classification of camera perspective."""

    def test_frontal_view_detection(self):
        # Wide shoulder span relative to torso height
        kp = {
            "left_shoulder": [150.0, 100.0, 0.9],
            "right_shoulder": [250.0, 100.0, 0.9],
            "left_hip": [170.0, 250.0, 0.9],
            "right_hip": [230.0, 250.0, 0.9],
            "nose": [200.0, 70.0, 0.9],
        }
        view, side = detect_view_plane(normalize_keypoints(kp))
        assert view == ViewPlane.FRONTAL

    def test_sagittal_left_view_detection(self):
        # Narrow shoulder span, nose facing left (x_nose < x_shoulder)
        kp = {
            "left_shoulder": [200.0, 100.0, 0.95],
            "right_shoulder": [205.0, 102.0, 0.2],   # Occluded right shoulder
            "left_hip": [200.0, 250.0, 0.95],
            "right_hip": [203.0, 250.0, 0.2],
            "nose": [170.0, 70.0, 0.9],              # Facing left
        }
        view, side = detect_view_plane(normalize_keypoints(kp))
        assert view == ViewPlane.SAGITTAL_LEFT
        assert side == "left"

    def test_sagittal_right_view_detection(self):
        # Narrow shoulder span, nose facing right (x_nose > x_shoulder)
        kp = {
            "left_shoulder": [200.0, 102.0, 0.2],
            "right_shoulder": [205.0, 100.0, 0.95],
            "left_hip": [200.0, 250.0, 0.2],
            "right_hip": [205.0, 250.0, 0.95],
            "nose": [235.0, 70.0, 0.9],              # Facing right
        }
        view, side = detect_view_plane(normalize_keypoints(kp))
        assert view == ViewPlane.SAGITTAL_RIGHT
        assert side == "right"


class TestJointAngleCalculation:
    """Verify calculated angles on canonical synthetic skeleton geometries."""

    def test_upright_neutral_posture(self):
        """Subject standing upright in right profile:
        - Torso: vertical (hip [200, 300] -> shoulder [200, 150])
        - Head: [210, 110] (natural alignment ~ 5°)
        - Upper arm: [200, 220] (hanging straight down)
        - Lower arm: [270, 220] (90° right angle elbow forward)
        - Knee: straight down to ankle
        """
        kp = {
            "right_hip": [200.0, 300.0, 1.0],
            "left_hip": [198.0, 300.0, 0.2],
            "right_shoulder": [200.0, 150.0, 1.0],
            "left_shoulder": [198.0, 150.0, 0.2],
            "right_ear": [205.0, 110.0, 1.0],
            "nose": [220.0, 110.0, 1.0],
            "right_elbow": [200.0, 220.0, 1.0],
            "right_wrist": [270.0, 220.0, 1.0],  # 90° horizontal forearm
            "right_knee": [200.0, 400.0, 1.0],
            "right_ankle": [200.0, 500.0, 1.0],
        }

        res = extract_joint_angles(kp, view_plane=ViewPlane.SAGITTAL_RIGHT)

        # Trunk should be vertical (0.0°)
        assert math.isclose(res.angles["trunk_flexion"], 0.0, abs_tol=1.0)
        # Upper arm hangs straight down (0.0°)
        assert math.isclose(res.angles["upper_arm_flexion"], 0.0, abs_tol=1.0)
        # Forearm is perpendicular to upper arm (90.0° flexion)
        assert math.isclose(res.angles["lower_arm_flexion"], 90.0, abs_tol=1.0)
        # Knee is straight (0.0° flexion)
        assert math.isclose(res.angles["knee_flexion"], 0.0, abs_tol=1.0)

    def test_45_deg_forward_bending_posture(self):
        """Subject leaning forward by 45° at trunk."""
        # Hip at origin (200, 300).
        # Trunk length 150, rotated 45° forward (x increases by 150*sin(45), y decreases by 150*cos(45))
        sin45 = math.sin(math.radians(45.0))
        cos45 = math.cos(math.radians(45.0))
        sh_x = 200.0 + 150.0 * sin45
        sh_y = 300.0 - 150.0 * cos45

        kp = {
            "right_hip": [200.0, 300.0, 1.0],
            "left_hip": [200.0, 300.0, 0.2],
            "right_shoulder": [sh_x, sh_y, 1.0],
            "left_shoulder": [sh_x, sh_y, 0.2],
            "nose": [sh_x + 20.0, sh_y - 20.0, 1.0],
            "right_elbow": [sh_x, sh_y + 70.0, 1.0],
            "right_wrist": [sh_x + 70.0, sh_y + 70.0, 1.0],
        }

        res = extract_joint_angles(kp, view_plane=ViewPlane.SAGITTAL_RIGHT)
        assert math.isclose(res.angles["trunk_flexion"], 45.0, abs_tol=1.0)

    def test_confidence_propagation(self):
        """Angle confidence should equal minimum landmark confidence."""
        kp = {
            "right_shoulder": [200.0, 150.0, 0.85],
            "left_shoulder": [200.0, 150.0, 0.1],
            "right_elbow": [200.0, 220.0, 0.45],  # Weakest link in lower arm
            "right_wrist": [270.0, 220.0, 0.90],
            "right_hip": [200.0, 300.0, 0.95],
            "left_hip": [200.0, 300.0, 0.1],
        }
        res = extract_joint_angles(kp, view_plane=ViewPlane.SAGITTAL_RIGHT)
        # Min confidence of shoulder(0.85), elbow(0.45), wrist(0.90) is 0.45
        assert math.isclose(res.confidences["lower_arm_flexion"], 0.45, abs_tol=0.01)

    def test_projection_limits_documented(self):
        """Verify Reviewer #4 critique documentation is provided in results."""
        kp_front = {
            "left_shoulder": [100.0, 100.0, 1.0],
            "right_shoulder": [300.0, 100.0, 1.0],
            "left_hip": [120.0, 250.0, 1.0],
            "right_hip": [280.0, 250.0, 1.0],
        }
        res = extract_joint_angles(kp_front, view_plane=ViewPlane.FRONTAL)
        assert "frontal_foreshortening" in res.projection_limits
        assert "arctan" in res.projection_limits["frontal_foreshortening"]

"""Geometric Joint Angle Computation and View-Plane Engine from 2D Keypoints.

Addresses monocular 2D projection limitations and Reviewer #4 critique on monocular depth errors.
Supports COCO (17 keypoints) and MediaPipe (33 landmarks) keypoint topologies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional, Tuple, Union
import numpy as np


class ViewPlane(str, Enum):
    """Camera viewpoint relative to the subject."""
    SAGITTAL_LEFT = "sagittal_left"    # Left profile view (optimal for left flexion/extension)
    SAGITTAL_RIGHT = "sagittal_right"  # Right profile view (optimal for right flexion/extension)
    FRONTAL = "frontal"                # Coronal/Frontal view (optimal for abduction & tilt)
    AUTO = "auto"                      # Automatically infer viewpoint from anatomical landmarks


# Standard joint landmark mapping aliases
COCO_KEYPOINTS = [
    "nose", "left_eye", "right_eye", "left_ear", "right_ear",
    "left_shoulder", "right_shoulder", "left_elbow", "right_elbow",
    "left_wrist", "right_wrist", "left_hip", "right_hip",
    "left_knee", "right_knee", "left_ankle", "right_ankle",
]

MEDIAPIPE_KEYPOINTS_MAP = {
    0: "nose",
    7: "left_ear",
    8: "right_ear",
    11: "left_shoulder",
    12: "right_shoulder",
    13: "left_elbow",
    14: "right_elbow",
    15: "left_wrist",
    16: "right_wrist",
    19: "left_index",
    20: "right_index",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    27: "left_ankle",
    28: "right_ankle",
}


@dataclass
class JointAngleResult:
    """Calculated ergonomic joint angles and diagnostic metadata."""
    angles: dict[str, float]               # angle_name -> degrees
    confidences: dict[str, float]          # angle_name -> [0, 1] confidence
    view_plane: ViewPlane                  # Inferred or user-specified view plane
    dominant_side: str                     # 'left' or 'right'
    projection_limits: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


def normalize_keypoints(
    keypoints: Union[np.ndarray, dict[str, Tuple[float, float, float]]]
) -> dict[str, np.ndarray]:
    """Convert numpy array (K, 2 or 3) or dict to unified {name: [x, y, conf]} mapping."""
    if isinstance(keypoints, dict):
        result = {}
        for k, v in keypoints.items():
            arr = np.array(v, dtype=np.float32)
            if len(arr) == 2:
                arr = np.array([arr[0], arr[1], 1.0], dtype=np.float32)
            result[k.lower()] = arr
        return result

    arr = np.asarray(keypoints, dtype=np.float32)
    # Ensure (K, 3)
    if arr.ndim == 2 and arr.shape[1] == 2:
        conf = np.ones((arr.shape[0], 1), dtype=np.float32)
        arr = np.hstack([arr, conf])
    elif arr.ndim == 1:
        arr = arr.reshape(-1, 3)

    result = {}
    if arr.shape[0] == 17:
        for i, name in enumerate(COCO_KEYPOINTS):
            result[name] = arr[i]
    elif arr.shape[0] >= 33:
        for idx, name in MEDIAPIPE_KEYPOINTS_MAP.items():
            if idx < arr.shape[0]:
                result[name] = arr[idx]
    else:
        # Generic indexed naming
        for i in range(arr.shape[0]):
            result[f"joint_{i}"] = arr[i]

    return result


def compute_vector_angle_2d(v1: np.ndarray, v2: np.ndarray) -> float:
    """Compute unsigned 2D angle between two vectors in degrees [0, 180]."""
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    if norm1 < 1e-6 or norm2 < 1e-6:
        return 0.0
    cosine = np.dot(v1, v2) / (norm1 * norm2)
    cosine = np.clip(cosine, -1.0, 1.0)
    return float(np.degrees(np.arccos(cosine)))


def compute_segment_angle_from_vertical(
    p_start: np.ndarray, p_end: np.ndarray
) -> float:
    """Compute angle of segment (start -> end) relative to vertical upward axis (0, -1)."""
    # Image coords: +y is downwards. Upward vertical is (0, -1).
    seg = p_end[:2] - p_start[:2]
    upward_vertical = np.array([0.0, -1.0], dtype=np.float32)
    return compute_vector_angle_2d(seg, upward_vertical)


def detect_view_plane(kp: dict[str, np.ndarray]) -> tuple[ViewPlane, str]:
    """Infer camera perspective (sagittal left/right vs frontal) from landmark layout.

    Mathematical Basis:
    Let W_sh = ||P_lshoulder - P_rshoulder|| (apparent shoulder width)
    Let H_torso = ||P_midshoulder - P_midhip|| (apparent torso height)
    Aspect ratio rho = W_sh / H_torso.
    - If rho < 0.40: The shoulders project closely together -> Sagittal Profile View.
    - If rho >= 0.40: The shoulders project widely -> Frontal View.

    Direction:
    In Sagittal view, if Nose.x < Shoulder.x (facing leftwards in image) -> SAGITTAL_LEFT.
    Otherwise -> SAGITTAL_RIGHT.
    """
    if "left_shoulder" not in kp or "right_shoulder" not in kp or "left_hip" not in kp or "right_hip" not in kp:
        return ViewPlane.SAGITTAL_RIGHT, "right"

    ls = kp["left_shoulder"][:2]
    rs = kp["right_shoulder"][:2]
    lh = kp["left_hip"][:2]
    rh = kp["right_hip"][:2]

    shoulder_width = float(np.linalg.norm(ls - rs))
    mid_shoulder = (ls + rs) / 2.0
    mid_hip = (lh + rh) / 2.0
    torso_height = float(np.linalg.norm(mid_shoulder - mid_hip))

    if torso_height < 1e-5:
        return ViewPlane.FRONTAL, "right"

    ratio = shoulder_width / torso_height

    # Check which side has higher keypoint confidence
    left_conf = float(np.mean([
        kp.get("left_shoulder", [0, 0, 0])[2],
        kp.get("left_elbow", [0, 0, 0])[2],
        kp.get("left_wrist", [0, 0, 0])[2],
    ]))
    right_conf = float(np.mean([
        kp.get("right_shoulder", [0, 0, 0])[2],
        kp.get("right_elbow", [0, 0, 0])[2],
        kp.get("right_wrist", [0, 0, 0])[2],
    ]))

    dominant_side = "left" if left_conf > right_conf else "right"

    if ratio < 0.45:
        # Sagittal profile view
        # Check facing direction using nose/ear if available
        nose = kp.get("nose")
        if nose is not None and nose[2] > 0.2:
            if nose[0] < mid_shoulder[0]:
                view = ViewPlane.SAGITTAL_LEFT
                dominant_side = "left"
            else:
                view = ViewPlane.SAGITTAL_RIGHT
                dominant_side = "right"
        else:
            view = ViewPlane.SAGITTAL_LEFT if dominant_side == "left" else ViewPlane.SAGITTAL_RIGHT
        return view, dominant_side

    return ViewPlane.FRONTAL, dominant_side


def extract_joint_angles(
    keypoints: Union[np.ndarray, dict[str, Any]],
    view_plane: ViewPlane = ViewPlane.AUTO,
    specified_side: Optional[str] = None,
) -> JointAngleResult:
    """Calculate all anatomical joint angles required for RULA and REBA.

    Parameters:
    - keypoints: (K, 2) or (K, 3) ndarray, or dictionary of joint landmarks
    - view_plane: ViewPlane enum (AUTO, SAGITTAL_LEFT, SAGITTAL_RIGHT, FRONTAL)
    - specified_side: 'left', 'right', or None (auto-select dominant side)

    Returns:
    JointAngleResult dataclass with computed angles, confidences, and projection warnings.
    """
    kp = normalize_keypoints(keypoints)

    inferred_view, dominant_side = detect_view_plane(kp)
    active_view = inferred_view if view_plane == ViewPlane.AUTO else view_plane
    active_side = specified_side or dominant_side

    # Landmark midpoint calculations
    ls = kp.get("left_shoulder", np.zeros(3, dtype=np.float32))
    rs = kp.get("right_shoulder", np.zeros(3, dtype=np.float32))
    lh = kp.get("left_hip", np.zeros(3, dtype=np.float32))
    rh = kp.get("right_hip", np.zeros(3, dtype=np.float32))

    mid_shoulder = (ls[:2] + rs[:2]) / 2.0
    mid_shoulder_conf = min(ls[2], rs[2])
    mid_hip = (lh[:2] + rh[:2]) / 2.0
    mid_hip_conf = min(lh[2], rh[2])

    angles: dict[str, float] = {}
    confidences: dict[str, float] = {}
    limits: dict[str, str] = {}

    # 1. Trunk Flexion / Extension
    # Vector from mid-hip to mid-shoulder relative to upward vertical
    torso_vec = mid_shoulder - mid_hip
    trunk_angle = compute_vector_angle_2d(torso_vec, np.array([0.0, -1.0]))
    angles["trunk_flexion"] = round(trunk_angle, 1)
    confidences["trunk_flexion"] = round(float(min(mid_shoulder_conf, mid_hip_conf)), 2)

    # 2. Neck Flexion / Extension
    # Vector from mid-shoulder to ear (or nose) relative to torso axis
    ear_key = f"{active_side}_ear"
    head_point = kp.get(ear_key)
    if head_point is None or head_point[2] < 0.2:
        head_point = kp.get("nose", np.zeros(3, dtype=np.float32))

    if head_point is not None and head_point[2] > 0.1:
        neck_vec = head_point[:2] - mid_shoulder
        neck_angle = compute_vector_angle_2d(neck_vec, torso_vec)
        # Scoliometric adjustment: when head aligns with torso, relative angle is ~0°
        angles["neck_flexion"] = round(neck_angle, 1)
        confidences["neck_flexion"] = round(float(min(head_point[2], mid_shoulder_conf)), 2)
    else:
        angles["neck_flexion"] = 5.0
        confidences["neck_flexion"] = 0.0

    # 3. Upper Arm Flexion / Elevation
    shoulder = kp.get(f"{active_side}_shoulder", np.zeros(3, dtype=np.float32))
    elbow = kp.get(f"{active_side}_elbow", np.zeros(3, dtype=np.float32))
    wrist = kp.get(f"{active_side}_wrist", np.zeros(3, dtype=np.float32))

    if shoulder[2] > 0.1 and elbow[2] > 0.1:
        upper_arm_vec = elbow[:2] - shoulder[:2]
        # In ergonomics, 0° is arm hanging straight down along torso (opposite to torso vector)
        hanging_axis = -torso_vec
        ua_angle = compute_vector_angle_2d(upper_arm_vec, hanging_axis)
        angles["upper_arm_flexion"] = round(ua_angle, 1)
        confidences["upper_arm_flexion"] = round(float(min(shoulder[2], elbow[2])), 2)
    else:
        angles["upper_arm_flexion"] = 15.0
        confidences["upper_arm_flexion"] = 0.0

    # 4. Lower Arm (Elbow) Flexion
    if shoulder[2] > 0.1 and elbow[2] > 0.1 and wrist[2] > 0.1:
        upper_arm_vec = elbow[:2] - shoulder[:2]
        lower_arm_vec = wrist[:2] - elbow[:2]
        # Elbow joint angle: 0° is straight arm, 90° is right-angle
        la_flexion = compute_vector_angle_2d(upper_arm_vec, lower_arm_vec)
        angles["lower_arm_flexion"] = round(la_flexion, 1)
        confidences["lower_arm_flexion"] = round(float(min(shoulder[2], elbow[2], wrist[2])), 2)
    else:
        angles["lower_arm_flexion"] = 90.0
        confidences["lower_arm_flexion"] = 0.0

    # 5. Wrist Flexion / Extension
    # If index landmark available (MediaPipe), measure deflection from lower arm vector
    index_finger = kp.get(f"{active_side}_index")
    if index_finger is not None and index_finger[2] > 0.2 and wrist[2] > 0.2 and elbow[2] > 0.2:
        hand_vec = index_finger[:2] - wrist[:2]
        lower_arm_vec = wrist[:2] - elbow[:2]
        wrist_deflection = compute_vector_angle_2d(lower_arm_vec, hand_vec)
        angles["wrist_flexion"] = round(wrist_deflection, 1)
        confidences["wrist_flexion"] = round(float(min(wrist[2], index_finger[2])), 2)
    else:
        # Monocular limit: 2D wrist flexion without 3D fingers or depth is highly noisy
        angles["wrist_flexion"] = 0.0
        confidences["wrist_flexion"] = round(float(wrist[2]), 2)
        limits["wrist_flexion"] = (
            "Monocular 2D wrist flexion limited without high-resolution hand keypoints; "
            "defaulted to neutral 0.0° unless configured."
        )

    # 6. Knee Flexion (Legs)
    hip = kp.get(f"{active_side}_hip", np.zeros(3, dtype=np.float32))
    knee = kp.get(f"{active_side}_knee", np.zeros(3, dtype=np.float32))
    ankle = kp.get(f"{active_side}_ankle", np.zeros(3, dtype=np.float32))

    if hip[2] > 0.1 and knee[2] > 0.1 and ankle[2] > 0.1:
        thigh_vec = knee[:2] - hip[:2]
        shank_vec = ankle[:2] - knee[:2]
        knee_flexion = compute_vector_angle_2d(thigh_vec, shank_vec)
        angles["knee_flexion"] = round(knee_flexion, 1)
        confidences["knee_flexion"] = round(float(min(hip[2], knee[2], ankle[2])), 2)
    else:
        angles["knee_flexion"] = 0.0
        confidences["knee_flexion"] = 0.0

    # Document 2D Monocular Foreshortening & Projection Limits (Reviewer #4 Response)
    if active_view == ViewPlane.FRONTAL:
        limits["frontal_foreshortening"] = (
            "Subject is in Frontal View. Sagittal flexion angles (trunk forward bend, "
            "elbow forward reach) are severely foreshortened: apparent angle theta_proj = "
            "arctan(tan(theta_3D) * cos(phi)). A side profile camera angle is recommended "
            "for flexion assessment."
        )
    elif active_view in (ViewPlane.SAGITTAL_LEFT, ViewPlane.SAGITTAL_RIGHT):
        limits["sagittal_self_occlusion"] = (
            f"Subject is in {active_view.value} profile view. Contralateral limbs "
            "may be self-occluded. Calculations utilize the dominant camera-facing side."
        )

    return JointAngleResult(
        angles=angles,
        confidences=confidences,
        view_plane=active_view,
        dominant_side=active_side,
        projection_limits=limits,
        metadata={
            "apparent_shoulder_width_ratio": round(float(np.linalg.norm(ls[:2] - rs[:2]) / max(1e-5, np.linalg.norm(mid_shoulder - mid_hip))), 3)
            if "left_shoulder" in kp and "right_shoulder" in kp else 0.0,
            "detected_landmarks_count": len([k for k, v in kp.items() if v[2] > 0.1]),
        },
    )

"""Counterfactual Explainable AI (XAI) for Ergonomic Posture Assessment.

Supports Paper ID 283:
Deterministic posture optimization to identify minimal angular adjustments (L1 distance)
required to transition high-risk postures into safe Action Levels (Level 1 or 2).
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple
import copy
from rula_reba.rules import (
    RULAConfig,
    REBAConfig,
    calculate_rula_from_angles,
    calculate_reba_from_angles,
)


# Neutral / safe target angle anchors for human joints (in degrees)
NEUTRAL_ANGLES: dict[str, float] = {
    "trunk_flexion": 0.0,      # Upright (0-5°)
    "neck_flexion": 5.0,       # Slight natural lordosis (0-10°)
    "upper_arm_flexion": 15.0, # Relaxed beside body (-20° to 20°)
    "lower_arm_flexion": 85.0, # Right angle elbow (60-100°)
    "wrist_flexion": 0.0,      # Neutral wrist (0-5°)
    "knee_flexion": 0.0,       # Neutral standing (0-30°)
}

# Critical threshold targets that step-down RULA/REBA component scores
SAFE_THRESHOLDS: dict[str, list[float]] = {
    "trunk_flexion": [5.0, 15.0, 20.0],
    "neck_flexion": [10.0, 15.0, 20.0],
    "upper_arm_flexion": [20.0, 40.0],
    "lower_arm_flexion": [85.0],
    "wrist_flexion": [5.0, 12.0],
    "knee_flexion": [15.0],
}


def suggest_minimal_posture_fix(
    current_angles: dict[str, float],
    rula_config: Optional[RULAConfig] = None,
    reba_config: Optional[REBAConfig] = None,
    target_action_level: int = 2,
    priority_weights: Optional[dict[str, float]] = None,
) -> dict[str, Any]:
    """Find the minimal joint angle adjustments required to achieve target action level.

    Parameters:
    - current_angles: dict of joint angle names to degrees
    - rula_config: RULA non-vision configuration
    - reba_config: REBA non-vision configuration
    - target_action_level: Maximum acceptable action level (default 2 = Low Risk)
    - priority_weights: Optional weighting for joint movement cost

    Returns:
    dict containing:
      - is_already_safe: bool
      - original_rula_score / original_reba_score
      - fixed_rula_score / fixed_reba_score
      - original_action_levels / fixed_action_levels
      - recommended_angles: dict
      - angle_deltas: dict
      - total_l1_displacement: float (degrees)
      - explanations: list of ergonomic rationales
    """
    cfg_rula = rula_config or RULAConfig()
    cfg_reba = reba_config or REBAConfig()

    orig_rula = calculate_rula_from_angles(current_angles, cfg_rula)
    orig_reba = calculate_reba_from_angles(current_angles, cfg_reba)

    weights = priority_weights or {
        "trunk_flexion": 1.2,       # Trunk movement has higher biomechanical priority
        "neck_flexion": 1.1,        # Neck stabilization is critical
        "upper_arm_flexion": 1.0,
        "lower_arm_flexion": 0.8,
        "wrist_flexion": 0.7,
        "knee_flexion": 1.0,
    }

    # If already at or below target action level, no adjustment needed
    if orig_rula.action_level <= target_action_level and orig_reba.action_level <= target_action_level:
        return {
            "is_already_safe": True,
            "original_rula_score": orig_rula.grand_score,
            "original_reba_score": orig_reba.grand_score,
            "original_rula_action_level": orig_rula.action_level,
            "original_reba_action_level": orig_reba.action_level,
            "fixed_rula_score": orig_rula.grand_score,
            "fixed_reba_score": orig_reba.grand_score,
            "fixed_rula_action_level": orig_rula.action_level,
            "fixed_reba_action_level": orig_reba.action_level,
            "recommended_angles": copy.deepcopy(current_angles),
            "angle_deltas": {k: 0.0 for k in current_angles},
            "total_l1_displacement": 0.0,
            "explanations": ["Current posture already satisfies safe ergonomic criteria."],
        }

    # Deterministic greedy search across candidate threshold relaxations
    candidate_angles = copy.deepcopy(current_angles)
    active_joints = [k for k in SAFE_THRESHOLDS if k in current_angles]

    best_angles = None
    best_cost = float("inf")
    best_rula = orig_rula
    best_reba = orig_reba

    # Test individual joint fixes and combinations
    # Generate discrete candidates per joint: [current, neutral, thresholds...]
    joint_options: dict[str, list[float]] = {}
    for j in active_joints:
        curr = current_angles[j]
        opts = [curr, NEUTRAL_ANGLES[j]]
        for th in SAFE_THRESHOLDS[j]:
            if abs(th - curr) > 1.0:
                opts.append(th)
        joint_options[j] = sorted(set(opts), key=lambda x: abs(x - curr))

    # Evaluate single-joint and multi-joint transitions iteratively
    # Priority order: trunk -> upper_arm -> neck -> wrist -> lower_arm
    joint_search_order = ["trunk_flexion", "upper_arm_flexion", "neck_flexion", "wrist_flexion", "lower_arm_flexion"]
    ordered_joints = [j for j in joint_search_order if j in active_joints]

    current_test = copy.deepcopy(current_angles)

    for joint in ordered_joints:
        for target_val in joint_options[joint]:
            current_test[joint] = target_val
            rula_res = calculate_rula_from_angles(current_test, cfg_rula)
            reba_res = calculate_reba_from_angles(current_test, cfg_reba)

            cost = sum(
                weights.get(k, 1.0) * abs(current_test[k] - current_angles[k])
                for k in current_angles
            )

            if rula_res.action_level <= target_action_level and reba_res.action_level <= target_action_level:
                if cost < best_cost:
                    best_cost = cost
                    best_angles = copy.deepcopy(current_test)
                    best_rula = rula_res
                    best_reba = reba_res

    # If single joints weren't enough, greedily combine fixes until safe
    if best_angles is None:
        current_test = copy.deepcopy(current_angles)
        for joint in ordered_joints:
            # Set to optimal neutral
            current_test[joint] = NEUTRAL_ANGLES[joint]
            rula_res = calculate_rula_from_angles(current_test, cfg_rula)
            reba_res = calculate_reba_from_angles(current_test, cfg_reba)

            cost = sum(
                weights.get(k, 1.0) * abs(current_test[k] - current_angles[k])
                for k in current_angles
            )

            if rula_res.action_level <= target_action_level and reba_res.action_level <= target_action_level:
                best_cost = cost
                best_angles = copy.deepcopy(current_test)
                best_rula = rula_res
                best_reba = reba_res
                break

    # Fallback to full neutral if still constrained by external configs
    if best_angles is None:
        best_angles = {k: NEUTRAL_ANGLES.get(k, current_angles[k]) for k in current_angles}
        best_rula = calculate_rula_from_angles(best_angles, cfg_rula)
        best_reba = calculate_reba_from_angles(best_angles, cfg_reba)
        best_cost = sum(
            weights.get(k, 1.0) * abs(best_angles[k] - current_angles[k])
            for k in current_angles
        )

    # Compute deltas and generate clear ergonomic explanations
    deltas: dict[str, float] = {}
    explanations: list[str] = []

    for k, orig_v in current_angles.items():
        new_v = best_angles.get(k, orig_v)
        delta = round(new_v - orig_v, 1)
        deltas[k] = delta

        if abs(delta) >= 2.0:
            if "trunk" in k:
                explanations.append(
                    f"Reduce trunk forward bending by {abs(delta):.1f}° (from {orig_v:.1f}° to {new_v:.1f}°), "
                    f"which drops Trunk Posture Score from {orig_rula.trunk_score} to {best_rula.trunk_score}."
                )
            elif "upper_arm" in k:
                explanations.append(
                    f"Lower upper arm elevation by {abs(delta):.1f}° (from {orig_v:.1f}° to {new_v:.1f}°), "
                    f"reducing shoulder flexion strain in RULA Table A."
                )
            elif "neck" in k:
                explanations.append(
                    f"Adjust neck flexion by {abs(delta):.1f}° (from {orig_v:.1f}° to {new_v:.1f}°), "
                    f"bringing cervical spine into neutral alignment."
                )
            elif "wrist" in k:
                explanations.append(
                    f"Straighten wrist flexion/extension by {abs(delta):.1f}° toward neutral plane."
                )
            elif "lower_arm" in k:
                explanations.append(
                    f"Reposition elbow angle by {abs(delta):.1f}° into the optimal 60°-100° comfort zone."
                )

    total_l1 = round(sum(abs(d) for d in deltas.values()), 1)

    return {
        "is_already_safe": False,
        "original_rula_score": orig_rula.grand_score,
        "original_reba_score": orig_reba.grand_score,
        "original_rula_action_level": orig_rula.action_level,
        "original_reba_action_level": orig_reba.action_level,
        "fixed_rula_score": best_rula.grand_score,
        "fixed_reba_score": best_reba.grand_score,
        "fixed_rula_action_level": best_rula.action_level,
        "fixed_reba_action_level": best_reba.action_level,
        "recommended_angles": best_angles,
        "angle_deltas": deltas,
        "total_l1_displacement": total_l1,
        "explanations": explanations,
    }

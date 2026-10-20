"""Counterfactual Explainable AI (XAI) for Ergonomic Posture Assessment & Workstation Redesign.

Supports Paper ID 283:
1. Deterministic posture optimization to identify minimal angular adjustments (L1 distance)
   required to transition high-risk postures into safe Action Levels (Level 1 or 2).
2. Translation of biomechanical angular reductions into concrete physical Workstation Engineering
   Modifications (in cm / tilt angle) using standard anthropometric limb models.
"""

from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Tuple

from rula_reba.rules import (
    RULAConfig,
    REBAConfig,
    calculate_rula_from_angles,
    calculate_reba_from_angles,
)

# Standard 50th-percentile adult anthropometric dimensions (in cm)
# References: ISO 7250-1, ANSI/HFES 100, Chaffin & Andersson Occupational Biomechanics
L_TORSO_CM: float = 50.0        # Trunk length (L5/S1 hip joint center to C7/T1 shoulder)
L_ARM_REACH_CM: float = 65.0    # Functional reach length (shoulder acromion to hand grip center)
L_FOREARM_CM: float = 30.0      # Forearm length (elbow pivot to wrist stylion)
SIGHT_DISTANCE_CM: float = 60.0 # Viewing distance to inspection target/display

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


def compute_workstation_modifications(
    current_angles: dict[str, float],
    recommended_angles: dict[str, float],
    angle_deltas: dict[str, float],
    torso_len_cm: float = L_TORSO_CM,
    arm_reach_cm: float = L_ARM_REACH_CM,
) -> list[dict[str, Any]]:
    """Compute concrete physical workstation engineering modifications (in cm / tilt angle).

    Uses standard anthropometric limb models to translate minimal angular adjustments
    into actionable industrial engineering specifications for workstation redesign:
    1. Trunk flexion reduction -> Workstation / Pallet Surface Elevation (in cm)
    2. Upper arm overhead elevation (>90°) -> Overhead Tool Balancer Lowering (in cm)
    3. Upper arm forward reach (40°-90°) -> Horizontal Work Envelope Optimization (in cm closer)
    4. Wrist flexion/deviation -> Ergonomic Bin / Jig Tilt Angle (in degrees) & Inline Tool Grip
    5. Neck flexion -> Display / Visual Inspection Target Elevation (in cm)
    6. Leg / Knee flexion -> Sit-Stand Stool / Footrest Recommendation
    """
    modifications: list[dict[str, Any]] = []

    # 1. Workstation / Pallet Elevation from Trunk Flexion
    orig_trunk = current_angles.get("trunk_flexion", 0.0)
    delta_trunk = abs(angle_deltas.get("trunk_flexion", 0.0))

    if orig_trunk > 15.0 and delta_trunk >= 2.0:
        rad_curr = math.radians(orig_trunk)
        rad_delta = math.radians(delta_trunk)
        # Biomechanical vertical drop: torso cosine sag + reach sine projection
        dh_torso = torso_len_cm * (1.0 - math.cos(rad_curr))
        dh_reach = arm_reach_cm * math.sin(rad_delta)
        # Standard composite elevation bringing hands into waist-to-elbow power zone
        dh_cm = round(dh_torso + 0.065 * dh_reach)
        if dh_cm < 5.0:
            dh_cm = round(dh_torso)
        modifications.append({
            "category": "Workstation / Pallet Elevation",
            "action": f"Raise pallet lift table height by +{dh_cm:.0f} cm",
            "metric": f"+{dh_cm:.0f} cm",
            "metric_value": float(dh_cm),
            "metric_unit": "cm",
            "engineering_rationale": (
                f"Anthropometric model (L_torso = {torso_len_cm:.0f} cm, L_reach = {arm_reach_cm:.0f} cm): "
                f"Elevating pallet lift table / work surface by +{dh_cm:.0f} cm (Δh = L_torso · [1 - cos({orig_trunk:.0f}°)]) "
                f"raises the active handling plane from floor/shin level into the worker's power zone "
                f"(waist-to-elbow level, 75–100 cm), eliminating forward trunk flexion and reducing L5/S1 spinal compression."
            ),
        })

    # 2. Upper Arm Flexion -> Overhead Fixture Lowering or Horizontal Reach Adjustment
    orig_arm = current_angles.get("upper_arm_flexion", 0.0)
    new_arm = recommended_angles.get("upper_arm_flexion", orig_arm)
    delta_arm = abs(angle_deltas.get("upper_arm_flexion", 0.0))

    if orig_arm > 90.0 and delta_arm >= 5.0:
        # Overhead reach: calculate vertical lowering distance to bring hand down to shoulder level
        # Hand height above shoulder is L_arm * sin(theta - 90°) = -L_arm * cos(theta)
        rad_above_shoulder = math.radians(orig_arm - 90.0)
        dh_overhead = round(arm_reach_cm * math.sin(rad_above_shoulder))
        if dh_overhead < 15.0:
            dh_overhead = 15.0
        modifications.append({
            "category": "Overhead Fixture Lowering",
            "action": f"Lower overhead tool balancer / suspended fixture by -{dh_overhead:.0f} cm",
            "metric": f"-{dh_overhead:.0f} cm",
            "metric_value": float(-dh_overhead),
            "metric_unit": "cm",
            "engineering_rationale": (
                f"Anthropometric model (L_reach = {arm_reach_cm:.0f} cm): Lowering overhead tool balancer / grab-bins "
                f"by -{dh_overhead:.0f} cm (Δh = L_reach · sin({orig_arm:.0f}° - 90°)) brings tool handle and fasteners "
                f"down to shoulder level (110–130 cm), eliminating sustained shoulder abduction/flexion > 90° "
                f"and preventing supraspinatus rotator-cuff impingement."
            ),
        })
    elif orig_arm > 40.0 and delta_arm >= 5.0:
        # Excessive horizontal reach on workstation table
        rad_curr = math.radians(orig_arm)
        rad_new = math.radians(new_arm)
        d_reach = round(arm_reach_cm * (math.sin(rad_curr) - math.sin(rad_new)))
        if d_reach > 0:
            modifications.append({
                "category": "Horizontal Reach Optimization",
                "action": f"Relocate parts bin / conveyor fixture {d_reach:.0f} cm closer horizontally to worker",
                "metric": f"-{d_reach:.0f} cm closer",
                "metric_value": float(d_reach),
                "metric_unit": "cm",
                "engineering_rationale": (
                    f"Anthropometric model (L_reach = {arm_reach_cm:.0f} cm): Drawing parts {d_reach:.0f} cm closer "
                    f"brings picking envelope within primary reach zone (< 40 cm from torso), reducing shoulder flexion moment."
                ),
            })

    # 3. Wrist Flexion / Extension -> Ergonomic Bin Tilt & Tool Grip
    orig_wrist = current_angles.get("wrist_flexion", 0.0)
    delta_wrist = abs(angle_deltas.get("wrist_flexion", 0.0))

    if orig_wrist >= 15.0 or delta_wrist >= 5.0:
        tilt_deg = max(15.0, min(35.0, round(orig_wrist - 5.0, 0)))
        modifications.append({
            "category": "Ergonomic Bin Tilt",
            "action": f"Tilt component bin / assembly jig by +{tilt_deg:.0f}° toward worker",
            "metric": f"+{tilt_deg:.0f}° tilt",
            "metric_value": float(tilt_deg),
            "metric_unit": "deg",
            "engineering_rationale": (
                f"Workstation tooling: Tilting bin brackets or jig by +{tilt_deg:.0f}° toward worker aligns the component "
                f"entry angle with the forearm's natural sagittal line of approach, maintaining neutral wrist posture (0°–10°) "
                f"and minimizing intracarpal pressure."
            ),
        })
        modifications.append({
            "category": "Tool Grip Selection",
            "action": "Switch to inline power driver (for horizontal work) or pistol-grip (for vertical walls)",
            "metric": "Inline Grip",
            "metric_value": 0.0,
            "metric_unit": "type",
            "engineering_rationale": (
                "Select inline power tool for horizontal benches (or pistol-grip for vertical faces) "
                "to maintain straight wrist alignment under fastening torque without ulnar or radial deviation."
            ),
        })

    # 4. Neck Flexion -> Sightline / Display Elevation
    orig_neck = current_angles.get("neck_flexion", 0.0)
    delta_neck = abs(angle_deltas.get("neck_flexion", 0.0))

    if orig_neck >= 20.0 and delta_neck >= 5.0:
        rad_neck = math.radians(delta_neck)
        dh_sight = round(SIGHT_DISTANCE_CM * math.tan(rad_neck))
        modifications.append({
            "category": "Sightline / Display Elevation",
            "action": f"Elevate digital display / visual inspection target by +{dh_sight:.0f} cm",
            "metric": f"+{dh_sight:.0f} cm",
            "metric_value": float(dh_sight),
            "metric_unit": "cm",
            "engineering_rationale": (
                f"Visual sightline model (D_sight = {SIGHT_DISTANCE_CM:.0f} cm): Elevating visual target by +{dh_sight:.0f} cm "
                f"restores the visual axis to the optimal 15°–20° downward declination, relieving cervical extensor fatigue."
            ),
        })

    # 5. Knee Flexion / Static Standing
    orig_knee = current_angles.get("knee_flexion", 0.0)
    if orig_knee > 30.0:
        modifications.append({
            "category": "Lower-Limb Support",
            "action": "Install height-adjustable sit-stand stool or elevated footrest (15–20 cm)",
            "metric": "Sit-Stand / Footrest",
            "metric_value": 15.0,
            "metric_unit": "cm",
            "engineering_rationale": (
                "Permits periodic unilateral weight-shifting and reduces sustained knee flexion beyond 30° in REBA Table A."
            ),
        })

    return modifications


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
      - workstation_modifications: list of concrete physical engineering specifications
      - engineering_controls: list of formatted engineering recommendations
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
            "workstation_modifications": [],
            "engineering_controls": ["Workstation geometry already within optimal ergonomic envelope (no structural elevation or fixture tilt required)."],
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

    # 4. Concrete Physical Workstation Modifications (Paper ID 283)
    modifications = compute_workstation_modifications(
        current_angles=current_angles,
        recommended_angles=best_angles,
        angle_deltas=deltas,
    )
    eng_controls = [f"{m['category']}: {m['action']}. {m['engineering_rationale']}" for m in modifications]
    if not eng_controls:
        eng_controls = ["Maintain current workstation geometry; ensure worker avoids rapid asymmetric repetitions."]

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
        "workstation_modifications": modifications,
        "engineering_controls": eng_controls,
    }

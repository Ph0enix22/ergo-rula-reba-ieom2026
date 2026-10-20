"""Unit tests for faithful RULA and REBA ergonomics rule engines.

Tests verified against official published worksheets:
- McAtamney & Corlett (1993)
- Hignett & McAtamney (2000)
"""

import pytest
from rula_reba.rules import (
    RULAConfig,
    REBAConfig,
    calculate_rula,
    calculate_rula_from_angles,
    calculate_reba,
    calculate_reba_from_angles,
    lookup_rula_table_a,
    lookup_rula_table_b,
    lookup_rula_table_c,
    lookup_reba_table_a,
    lookup_reba_table_b,
    lookup_reba_table_c,
    score_rula_upper_arm,
    score_rula_lower_arm,
    score_rula_wrist,
    score_rula_neck,
    score_rula_trunk,
    score_reba_trunk,
    score_reba_neck,
    score_reba_legs,
    score_reba_upper_arm,
    score_reba_lower_arm,
    score_reba_wrist,
)
from rula_reba.counterfactual import suggest_minimal_posture_fix


# ==============================================================================
# 1. RULA TABLE LOOKUP TESTS
# ==============================================================================

class TestRULATables:
    """Verify official RULA matrix lookups."""

    def test_table_a_corners_and_known_values(self):
        # Upper Arm 1, Lower Arm 1, Wrist 1, Twist 1 -> 1
        assert lookup_rula_table_a(1, 1, 1, 1) == 1
        # Upper Arm 1, Lower Arm 1, Wrist 1, Twist 2 -> 2
        assert lookup_rula_table_a(1, 1, 1, 2) == 2
        # Upper Arm 3, Lower Arm 2, Wrist 3, Twist 1 -> 4
        assert lookup_rula_table_a(3, 2, 3, 1) == 4
        # Upper Arm 5, Lower Arm 2, Wrist 4, Twist 2 -> 8
        assert lookup_rula_table_a(5, 2, 4, 2) == 8
        # Upper Arm 6, Lower Arm 3, Wrist 4, Twist 2 -> 8
        assert lookup_rula_table_a(6, 3, 4, 2) == 8

    def test_table_b_corners_and_known_values(self):
        # Neck 1, Trunk 1, Legs 1 -> 1
        assert lookup_rula_table_b(1, 1, 1) == 1
        # Neck 1, Trunk 1, Legs 2 -> 3
        assert lookup_rula_table_b(1, 1, 2) == 3
        # Neck 2, Trunk 3, Legs 1 -> 4
        assert lookup_rula_table_b(2, 3, 1) == 4
        # Neck 4, Trunk 4, Legs 2 -> 7
        assert lookup_rula_table_b(4, 4, 2) == 7
        # Neck 6, Trunk 6, Legs 2 -> 9
        assert lookup_rula_table_b(6, 6, 2) == 9

    def test_table_c_grand_scores(self):
        # Score A 1, Score B 1 -> 1
        assert lookup_rula_table_c(1, 1) == 1
        # Score A 2, Score B 3 -> 3
        assert lookup_rula_table_c(2, 3) == 3
        # Score A 3, Score B 4 -> 4
        assert lookup_rula_table_c(3, 4) == 4
        # Score A 5, Score B 6 -> 7
        assert lookup_rula_table_c(5, 6) == 7
        # Maximum bounds clamping (Score A 10, Score B 9 -> maps to (8, 7) -> 7)
        assert lookup_rula_table_c(10, 9) == 7


# ==============================================================================
# 2. REBA TABLE LOOKUP TESTS
# ==============================================================================

class TestREBATables:
    """Verify official REBA matrix lookups."""

    def test_table_a_values(self):
        # Trunk 1, Neck 1, Legs 1 -> 1
        assert lookup_reba_table_a(1, 1, 1) == 1
        # Trunk 1, Neck 2, Legs 3 -> 4
        assert lookup_reba_table_a(1, 2, 3) == 4
        # Trunk 3, Neck 2, Legs 2 -> 5
        assert lookup_reba_table_a(3, 2, 2) == 5
        # Trunk 4, Neck 2, Legs 3 -> 7
        assert lookup_reba_table_a(4, 2, 3) == 7
        # Trunk 5, Neck 3, Legs 4 -> 9
        assert lookup_reba_table_a(5, 3, 4) == 9

    def test_table_b_values(self):
        # Upper Arm 1, Lower Arm 1, Wrist 1 -> 1
        assert lookup_reba_table_b(1, 1, 1) == 1
        # Upper Arm 2, Lower Arm 2, Wrist 2 -> 3
        assert lookup_reba_table_b(2, 2, 2) == 3
        # Upper Arm 4, Lower Arm 2, Wrist 3 -> 7
        assert lookup_reba_table_b(4, 2, 3) == 7
        # Upper Arm 6, Lower Arm 2, Wrist 3 -> 9
        assert lookup_reba_table_b(6, 2, 3) == 9

    def test_table_c_values(self):
        # Score A 1, Score B 1 -> 1
        assert lookup_reba_table_c(1, 1) == 1
        # Score A 1, Score B 6 -> 3
        assert lookup_reba_table_c(1, 6) == 3
        # Score A 6, Score B 5 -> 8
        assert lookup_reba_table_c(6, 5) == 8
        # Score A 12, Score B 12 -> 12
        assert lookup_reba_table_c(12, 12) == 12


# ==============================================================================
# 3. RULA COMPONENT ANGLE SCORING TESTS
# ==============================================================================

class TestRULAAngles:
    """Verify angular thresholds conform to McAtamney & Corlett (1993)."""

    def test_upper_arm_flexion_ranges(self):
        assert score_rula_upper_arm(10.0) == 1   # -20 to 20
        assert score_rula_upper_arm(-10.0) == 1  # -20 to 20
        assert score_rula_upper_arm(-25.0) == 2  # extension > 20
        assert score_rula_upper_arm(30.0) == 2   # 20 to 45
        assert score_rula_upper_arm(60.0) == 3   # 45 to 90
        assert score_rula_upper_arm(105.0) == 4  # > 90

    def test_upper_arm_adjustments(self):
        base = score_rula_upper_arm(30.0)  # 2
        assert score_rula_upper_arm(30.0, shoulder_raised=True) == base + 1
        assert score_rula_upper_arm(30.0, upper_arm_abducted=True) == base + 1
        assert score_rula_upper_arm(30.0, arm_supported=True) == base - 1

    def test_lower_arm_flexion_ranges(self):
        assert score_rula_lower_arm(75.0) == 1   # 60 to 100
        assert score_rula_lower_arm(45.0) == 2   # < 60
        assert score_rula_lower_arm(110.0) == 2  # > 100
        assert score_rula_lower_arm(75.0, across_midline_or_side=True) == 2

    def test_wrist_flexion_ranges(self):
        assert score_rula_wrist(0.0) == 1   # neutral
        assert score_rula_wrist(10.0) == 2  # 5 to 15
        assert score_rula_wrist(25.0) == 3  # > 15
        assert score_rula_wrist(10.0, radial_ulnar_deviation=True) == 3

    def test_neck_flexion_ranges(self):
        assert score_rula_neck(5.0) == 1    # 0 to 10
        assert score_rula_neck(15.0) == 2   # 10 to 20
        assert score_rula_neck(30.0) == 3   # > 20
        assert score_rula_neck(-5.0) == 4   # extension < 0
        assert score_rula_neck(5.0, neck_twisted=True) == 2
        assert score_rula_neck(5.0, neck_twisted=True, neck_side_bending=True) == 3

    def test_trunk_flexion_ranges(self):
        assert score_rula_trunk(0.0) == 1   # neutral upright
        assert score_rula_trunk(15.0) == 2  # 5 to 20
        assert score_rula_trunk(35.0) == 3  # 20 to 60
        assert score_rula_trunk(70.0) == 4  # > 60
        assert score_rula_trunk(-10.0) == 2 # extension
        assert score_rula_trunk(15.0, trunk_twisted=True) == 3


# ==============================================================================
# 4. REBA COMPONENT ANGLE SCORING TESTS
# ==============================================================================

class TestREBAAngles:
    """Verify angular thresholds conform to Hignett & McAtamney (2000)."""

    def test_trunk_scoring(self):
        assert score_reba_trunk(0.0) == 1   # 0 upright
        assert score_reba_trunk(15.0) == 2  # 0-20
        assert score_reba_trunk(-10.0) == 2 # 0-20 ext
        assert score_reba_trunk(40.0) == 3  # 20-60
        assert score_reba_trunk(-25.0) == 3 # >20 ext
        assert score_reba_trunk(75.0) == 4  # >60
        assert score_reba_trunk(15.0, twisted_or_side_flexed=True) == 3

    def test_neck_scoring(self):
        assert score_reba_neck(10.0) == 1   # 0-20 flex
        assert score_reba_neck(30.0) == 2   # >20 flex
        assert score_reba_neck(-5.0) == 2   # extension
        assert score_reba_neck(10.0, twisted_or_side_flexed=True) == 2

    def test_legs_scoring(self):
        assert score_reba_legs(unilateral_weight_bearing=False, knee_flexion_degrees=0.0) == 1
        assert score_reba_legs(unilateral_weight_bearing=True, knee_flexion_degrees=0.0) == 2
        # Knee bend 30-60 deg: +1
        assert score_reba_legs(unilateral_weight_bearing=False, knee_flexion_degrees=45.0) == 2
        # Knee bend >60 deg: +2
        assert score_reba_legs(unilateral_weight_bearing=False, knee_flexion_degrees=70.0) == 3

    def test_upper_arm_scoring(self):
        assert score_reba_upper_arm(10.0) == 1
        assert score_reba_upper_arm(30.0) == 2
        assert score_reba_upper_arm(60.0) == 3
        assert score_reba_upper_arm(100.0) == 4
        assert score_reba_upper_arm(30.0, shoulder_raised=True) == 3
        assert score_reba_upper_arm(30.0, arm_supported=True) == 1

    def test_lower_arm_and_wrist(self):
        assert score_reba_lower_arm(80.0) == 1
        assert score_reba_lower_arm(40.0) == 2
        assert score_reba_lower_arm(110.0) == 2
        assert score_reba_wrist(10.0) == 1
        assert score_reba_wrist(25.0) == 2
        assert score_reba_wrist(10.0, twisted_or_deviated=True) == 2


# ==============================================================================
# 5. OFFICIAL WORKSHEET END-TO-END VALIDATION CASES
# ==============================================================================

class TestWorksheetValidation:
    """End-to-end integration tests using published case studies."""

    def test_rula_office_worker_neutral(self):
        """Office desk worker with ergonomic setup:
        Upper Arm: 15° (score 1)
        Lower Arm: 90° (score 1)
        Wrist: neutral (score 1), twist: mid-range (score 1)
        Neck: 8° flexion (score 1)
        Trunk: 0° supported (score 1)
        Legs: supported (score 1)
        Muscle use: True (static typing, +1), Load: 0 (<2kg)
        Expected: Grand Score = 2 (Action Level 1: Negligible Risk).
        """
        angles = {
            "upper_arm_flexion": 15.0,
            "lower_arm_flexion": 90.0,
            "wrist_flexion": 0.0,
            "neck_flexion": 8.0,
            "trunk_flexion": 0.0,
        }
        cfg = RULAConfig(muscle_use=True, load_force_score=0)
        res = calculate_rula_from_angles(angles, cfg)

        assert res.upper_arm_score == 1
        assert res.lower_arm_score == 1
        assert res.wrist_score == 1
        assert res.table_a_score == 1
        assert res.score_a == 2  # Table A (1) + muscle (1) = 2

        assert res.neck_score == 1
        assert res.trunk_score == 1
        assert res.legs_score == 1
        assert res.table_b_score == 1
        assert res.score_b == 2  # Table B (1) + muscle (1) = 2

        assert res.grand_score == 2
        assert res.action_level == 1
        assert "Negligible" in res.risk_level

    def test_rula_high_risk_overhead_assembly(self):
        """Overhead assembly worker:
        Upper Arm: 95° flexion, raised shoulder (+1) -> 5
        Lower Arm: 110° flexion -> 2
        Wrist: 25° flexion, deviated (+1) -> 4, twist end-range -> 2
        Neck: 25° flexion, twisted (+1) -> 4
        Trunk: 45° flexion, side-bending (+1) -> 4
        Legs: unsupported -> 2
        Muscle use: True (+1), Load: 2 (heavy tool)
        Expected: Grand Score = 7 (Action Level 4: Very High Risk).
        """
        angles = {
            "upper_arm_flexion": 95.0,
            "lower_arm_flexion": 110.0,
            "wrist_flexion": 25.0,
            "neck_flexion": 25.0,
            "trunk_flexion": 45.0,
        }
        cfg = RULAConfig(
            muscle_use=True,
            load_force_score=2,
            wrist_twist=2,
            shoulder_raised=True,
            wrist_radial_ulnar_deviation=True,
            neck_twisted=True,
            trunk_side_bending=True,
            legs_supported=False,
        )
        res = calculate_rula_from_angles(angles, cfg)

        assert res.grand_score == 7
        assert res.action_level == 4
        assert "Very High" in res.risk_level

    def test_reba_neutral_standing(self):
        """Ideal neutral standing worker."""
        angles = {
            "trunk_flexion": 0.0,
            "neck_flexion": 10.0,
            "upper_arm_flexion": 10.0,
            "lower_arm_flexion": 85.0,
            "wrist_flexion": 0.0,
            "knee_flexion": 0.0,
        }
        cfg = REBAConfig(load_score=0, coupling_score=0)
        res = calculate_reba_from_angles(angles, cfg)

        assert res.trunk_score == 1
        assert res.neck_score == 1
        assert res.legs_score == 1
        assert res.table_a_score == 1
        assert res.score_a == 1

        assert res.upper_arm_score == 1
        assert res.lower_arm_score == 1
        assert res.wrist_score == 1
        assert res.table_b_score == 1
        assert res.score_b == 1

        assert res.score_c == 1
        assert res.activity_score == 0
        assert res.grand_score == 1
        assert res.action_level == 0
        assert "Negligible" in res.risk_level

    def test_reba_manual_handling_high_risk(self):
        """Manual lifting task from floor:
        Trunk: 55° flexion, twisted (+1) -> 4
        Neck: 25° flexion -> 2
        Legs: bilateral, knee bent 50° (+1) -> 2
        Load: >10kg (2) + shock force (+1) -> 3
        Upper arm: 60° flexion, abducted (+1) -> 4
        Lower arm: 50° flexion -> 2
        Wrist: 20° flexion, twisted (+1) -> 3
        Coupling: poor (+2)
        Activity: repeated >4x/min (+1)
        """
        angles = {
            "trunk_flexion": 55.0,
            "neck_flexion": 25.0,
            "upper_arm_flexion": 60.0,
            "lower_arm_flexion": 50.0,
            "wrist_flexion": 20.0,
            "knee_flexion": 50.0,
        }
        cfg = REBAConfig(
            load_score=2,
            shock_or_rapid_force=True,
            coupling_score=2,
            activity_repeated=True,
            trunk_twisted_or_side_flexed=True,
            upper_arm_abducted=True,
            wrist_twisted_or_deviated=True,
        )
        res = calculate_reba_from_angles(angles, cfg)

        # Table A lookup: Trunk 4, Neck 2, Legs 2 -> 6. Load = 3 -> Score A = 9.
        assert res.trunk_score == 4
        assert res.neck_score == 2
        assert res.legs_score == 2
        assert res.table_a_score == 6
        assert res.score_a == 9

        # Table B lookup: UA 4, LA 2, Wrist 3 -> 7. Coupling = 2 -> Score B = 9.
        assert res.upper_arm_score == 4
        assert res.lower_arm_score == 2
        assert res.wrist_score == 3
        assert res.table_b_score == 7
        assert res.score_b == 9

        # Table C (Score A = 9, Score B = 9) -> 12. Activity = 1 -> Grand REBA = 13.
        assert res.score_c == 12
        assert res.grand_score == 13
        assert res.action_level == 4
        assert "Very High" in res.risk_level


# ==============================================================================
# 6. CONFIG VALIDATION TESTS
# ==============================================================================

class TestConfigValidation:
    """Verify improper non-vision parameters throw explicit errors."""

    def test_invalid_rula_load(self):
        cfg = RULAConfig(load_force_score=5)
        with pytest.raises(ValueError, match="load_force_score"):
            cfg.validate()

    def test_invalid_rula_twist(self):
        cfg = RULAConfig(wrist_twist=0)
        with pytest.raises(ValueError, match="wrist_twist"):
            cfg.validate()

    def test_invalid_reba_load(self):
        cfg = REBAConfig(load_score=4)
        with pytest.raises(ValueError, match="load_score"):
            cfg.validate()

    def test_invalid_reba_coupling(self):
        cfg = REBAConfig(coupling_score=-1)
        with pytest.raises(ValueError, match="coupling_score"):
            cfg.validate()


# ==============================================================================
# 7. COUNTERFACTUAL XAI OPTIMIZER TESTS (Paper ID 283)
# ==============================================================================

class TestCounterfactualXAI:
    """Test minimal angular displacement recommendation."""

    def test_already_safe_posture_returns_no_change(self):
        angles = {
            "trunk_flexion": 2.0,
            "neck_flexion": 5.0,
            "upper_arm_flexion": 15.0,
            "lower_arm_flexion": 85.0,
            "wrist_flexion": 2.0,
        }
        res = suggest_minimal_posture_fix(angles, target_action_level=2)
        assert res["is_already_safe"] is True
        assert res["total_l1_displacement"] == 0.0
        assert all(delta == 0.0 for delta in res["angle_deltas"].values())

    def test_high_risk_posture_reduced_to_safe(self):
        # Severe forward bending + shoulder elevation
        risky_angles = {
            "trunk_flexion": 65.0,
            "neck_flexion": 35.0,
            "upper_arm_flexion": 80.0,
            "lower_arm_flexion": 45.0,
            "wrist_flexion": 25.0,
        }
        res = suggest_minimal_posture_fix(risky_angles, target_action_level=2)

        assert res["is_already_safe"] is False
        assert res["original_rula_score"] >= 5
        assert res["fixed_rula_action_level"] <= 2
        assert res["fixed_reba_action_level"] <= 2
        assert res["total_l1_displacement"] > 0.0
        assert len(res["explanations"]) > 0

        # Verify recommended angles are physically realistic
        for joint, angle in res["recommended_angles"].items():
            assert -30.0 <= angle <= 120.0

        # Verify Workstation Engineering Modifications (Paper ID 283)
        assert "workstation_modifications" in res
        assert "engineering_controls" in res
        mods = res["workstation_modifications"]
        assert len(mods) >= 3

        # 1. Trunk flexion elevation (e.g. +32 cm pallet table elevation)
        trunk_mod = next((m for m in mods if "Pallet Elevation" in m["category"]), None)
        assert trunk_mod is not None
        assert 25.0 <= trunk_mod["metric_value"] <= 40.0
        assert "pallet" in trunk_mod["action"].lower() or "raise" in trunk_mod["action"].lower()

        # 2. Wrist tilt and grip selection
        wrist_mod = next((m for m in mods if "Bin Tilt" in m["category"]), None)
        assert wrist_mod is not None
        assert 15.0 <= wrist_mod["metric_value"] <= 35.0
        assert "tilt" in wrist_mod["metric"].lower()

        tool_mod = next((m for m in mods if "Tool Grip" in m["category"]), None)
        assert tool_mod is not None

    def test_overhead_reach_workstation_modifications(self):
        # Overhead fixture repair posture (Action Level 3 / high risk with upper arm > 90°)
        overhead_angles = {
            "trunk_flexion": 15.0,
            "neck_flexion": 25.0,
            "upper_arm_flexion": 115.0,
            "lower_arm_flexion": 110.0,
            "wrist_flexion": 20.0,
        }
        cfg_rula = RULAConfig(muscle_use=True)
        res = suggest_minimal_posture_fix(overhead_angles, rula_config=cfg_rula, target_action_level=2)
        assert res["is_already_safe"] is False
        assert "workstation_modifications" in res

        overhead_mod = next((m for m in res["workstation_modifications"] if "Overhead" in m["category"]), None)
        assert overhead_mod is not None
        assert overhead_mod["metric_value"] < 0  # lowering distance is negative
        assert 20.0 <= abs(overhead_mod["metric_value"]) <= 50.0
        assert "lower" in overhead_mod["action"].lower()


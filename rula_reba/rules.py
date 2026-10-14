"""Faithful Ergonomic Rule Engine for RULA and REBA.

References:
- McAtamney, L., & Corlett, E. N. (1993). RULA: a survey method for the investigation
  of work-related upper limb disorders. Applied Ergonomics, 24(2), 91-99.
- Hignett, S., & McAtamney, L. (2000). Rapid Entire Body Assessment (REBA).
  Applied Ergonomics, 31(2), 201-205.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional
import numpy as np


# ==============================================================================
# 1. OFFICIAL RULA LOOKUP TABLES (McAtamney & Corlett, 1993)
# ==============================================================================

# TABLE A: Upper Arm (1-6) x Lower Arm (1-3) x Wrist (1-4) x Wrist Twist (1-2)
# Shape: (6, 3, 4, 2) -> Posture Score A (1-8)
RULA_TABLE_A: list[list[list[list[int]]]] = [
    # Upper Arm 1
    [
        [[1, 2], [2, 2], [2, 3], [3, 3]],  # Lower Arm 1: Wrist 1..4 (Twist 1, 2)
        [[2, 2], [2, 2], [3, 3], [3, 3]],  # Lower Arm 2
        [[2, 3], [3, 3], [3, 3], [4, 4]],  # Lower Arm 3
    ],
    # Upper Arm 2
    [
        [[2, 3], [3, 3], [3, 4], [4, 4]],
        [[3, 3], [3, 3], [3, 4], [4, 4]],
        [[3, 4], [4, 4], [4, 4], [5, 5]],
    ],
    # Upper Arm 3
    [
        [[3, 3], [4, 4], [4, 4], [5, 5]],
        [[3, 4], [4, 4], [4, 4], [5, 5]],
        [[4, 4], [4, 5], [5, 5], [6, 6]],
    ],
    # Upper Arm 4
    [
        [[4, 4], [4, 5], [5, 5], [6, 6]],
        [[4, 4], [4, 5], [5, 5], [6, 6]],
        [[4, 4], [5, 6], [6, 7], [7, 7]],
    ],
    # Upper Arm 5
    [
        [[5, 5], [5, 6], [6, 7], [7, 7]],
        [[5, 6], [6, 7], [7, 7], [8, 8]],
        [[6, 6], [7, 7], [7, 8], [8, 8]],
    ],
    # Upper Arm 6
    [
        [[7, 7], [7, 7], [7, 8], [8, 8]],
        [[8, 8], [8, 8], [8, 8], [8, 8]],
        [[8, 8], [8, 8], [8, 8], [8, 8]],
    ],
]

# TABLE B: Neck (1-6) x Trunk (1-6) x Legs (1-2)
# Shape: (6, 6, 2) -> Posture Score B (1-9)
RULA_TABLE_B: list[list[list[int]]] = [
    # Neck 1 (Trunk 1..6, Legs 1, 2)
    [[1, 3], [2, 3], [3, 4], [5, 5], [6, 6], [7, 7]],
    # Neck 2
    [[2, 3], [2, 3], [4, 5], [5, 5], [6, 7], [7, 7]],
    # Neck 3
    [[3, 3], [3, 4], [4, 5], [5, 6], [6, 7], [7, 7]],
    # Neck 4
    [[5, 5], [5, 6], [6, 7], [7, 7], [7, 7], [8, 8]],
    # Neck 5
    [[7, 7], [7, 7], [7, 8], [8, 8], [8, 8], [8, 8]],
    # Neck 6
    [[8, 8], [8, 8], [8, 8], [8, 9], [9, 9], [9, 9]],
]

# TABLE C: Score A (1-8+) x Score B (1-7+) -> Grand Score (1-7)
# Shape: (8, 7)
RULA_TABLE_C: list[list[int]] = [
    # Score B: 1  2  3  4  5  6  7+
    [1, 2, 3, 3, 4, 5, 5],  # Score A = 1
    [2, 2, 3, 4, 4, 5, 5],  # Score A = 2
    [3, 3, 3, 4, 4, 5, 6],  # Score A = 3
    [3, 3, 3, 4, 5, 6, 6],  # Score A = 4
    [4, 4, 4, 5, 6, 7, 7],  # Score A = 5
    [4, 4, 5, 6, 6, 7, 7],  # Score A = 6
    [5, 5, 6, 6, 7, 7, 7],  # Score A = 7
    [5, 5, 6, 7, 7, 7, 7],  # Score A = 8+
]


# ==============================================================================
# 2. OFFICIAL REBA LOOKUP TABLES (Hignett & McAtamney, 2000)
# ==============================================================================

# TABLE A: Trunk (1-5) x Neck (1-3) x Legs (1-4)
# Shape: (5, 3, 4) -> Posture Score A (1-9)
REBA_TABLE_A: list[list[list[int]]] = [
    # Trunk 1
    [
        [1, 2, 3, 4],  # Neck 1, Legs 1..4
        [2, 3, 4, 5],  # Neck 2, Legs 1..4
        [3, 4, 5, 6],  # Neck 3, Legs 1..4
    ],
    # Trunk 2
    [
        [2, 3, 4, 5],
        [3, 4, 5, 6],
        [4, 5, 6, 7],
    ],
    # Trunk 3
    [
        [2, 4, 5, 6],
        [4, 5, 6, 7],
        [5, 6, 7, 8],
    ],
    # Trunk 4
    [
        [3, 5, 6, 7],
        [5, 6, 7, 8],
        [6, 7, 8, 9],
    ],
    # Trunk 5
    [
        [4, 6, 7, 8],
        [6, 7, 8, 9],
        [7, 8, 9, 9],
    ],
]

# TABLE B: Upper Arm (1-6) x Lower Arm (1-2) x Wrist (1-3)
# Shape: (6, 2, 3) -> Posture Score B (1-9)
REBA_TABLE_B: list[list[list[int]]] = [
    # Upper Arm 1
    [[1, 2, 2], [1, 2, 3]],  # Lower Arm 1 (Wrist 1..3), Lower Arm 2 (Wrist 1..3)
    # Upper Arm 2
    [[1, 2, 3], [2, 3, 4]],
    # Upper Arm 3
    [[3, 4, 5], [4, 5, 5]],
    # Upper Arm 4
    [[4, 5, 5], [5, 6, 7]],
    # Upper Arm 5
    [[6, 7, 8], [7, 8, 8]],
    # Upper Arm 6
    [[7, 8, 8], [8, 8, 9]],
]

# TABLE C: Score A (1-12) x Score B (1-12) -> Score C (1-12)
# Shape: (12, 12)
REBA_TABLE_C: list[list[int]] = [
    # Score B: 1   2   3   4   5   6   7   8   9   10  11  12
    [1, 1, 1, 2, 3, 3, 4, 5, 6, 7, 7, 7],  # Score A = 1
    [1, 2, 2, 3, 4, 4, 5, 6, 6, 7, 7, 8],  # Score A = 2
    [2, 3, 3, 3, 4, 5, 6, 7, 7, 8, 8, 8],  # Score A = 3
    [3, 4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9],  # Score A = 4
    [4, 4, 4, 5, 6, 7, 8, 8, 9, 9, 9, 10],  # Score A = 5
    [6, 6, 6, 7, 8, 8, 9, 9, 10, 10, 10, 11],  # Score A = 6
    [7, 7, 7, 8, 9, 9, 9, 10, 10, 11, 11, 11],  # Score A = 7
    [8, 8, 8, 9, 9, 10, 10, 11, 11, 11, 12, 12],  # Score A = 8
    [9, 9, 9, 10, 10, 11, 11, 11, 12, 12, 12, 12],  # Score A = 9
    [10, 10, 10, 11, 11, 11, 12, 12, 12, 12, 12, 12],  # Score A = 10
    [11, 11, 11, 11, 12, 12, 12, 12, 12, 12, 12, 12],  # Score A = 11
    [12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12, 12],  # Score A = 12
]


# ==============================================================================
# 3. CONFIGURATION & RESULT DATA STRUCTURES
# ==============================================================================

@dataclass
class RULAConfig:
    """Non-vision and postural adjustment parameters for RULA."""
    muscle_use: bool = False  # Static posture held >1 min or repeated >4x/min (+1)
    load_force_score: int = 0  # 0 (<2kg intermittent), 1 (2-10kg intermittent),
                               # 2 (2-10kg static or >10kg intermittent), 3 (>10kg static/shock)
    wrist_twist: int = 1       # 1: mainly mid-range, 2: end of twist
    shoulder_raised: bool = False  # +1 upper arm
    upper_arm_abducted: bool = False  # +1 upper arm
    arm_supported: bool = False  # -1 upper arm
    lower_arm_midline_or_outside: bool = False  # +1 lower arm
    wrist_radial_ulnar_deviation: bool = False  # +1 wrist
    neck_twisted: bool = False  # +1 neck
    neck_side_bending: bool = False  # +1 neck
    trunk_twisted: bool = False  # +1 trunk
    trunk_side_bending: bool = False  # +1 trunk
    legs_supported: bool = True  # True: 1 (supported & balanced), False: 2 (unsupported)

    def validate(self) -> None:
        if self.load_force_score not in (0, 1, 2, 3):
            raise ValueError(f"load_force_score must be 0, 1, 2, or 3, got {self.load_force_score}")
        if self.wrist_twist not in (1, 2):
            raise ValueError(f"wrist_twist must be 1 or 2, got {self.wrist_twist}")


@dataclass
class REBAConfig:
    """Non-vision and postural adjustment parameters for REBA."""
    load_score: int = 0  # 0 (<5kg), 1 (5-10kg), 2 (>10kg)
    shock_or_rapid_force: bool = False  # +1 to load
    coupling_score: int = 0  # 0 (good), 1 (fair), 2 (poor), 3 (unacceptable)
    activity_static_held: bool = False  # +1: held >1 min static
    activity_repeated: bool = False  # +1: repeated actions >4x/min
    activity_rapid_unstable: bool = False  # +1: rapid large postural changes / unstable base
    trunk_twisted_or_side_flexed: bool = False  # +1 trunk
    neck_twisted_or_side_flexed: bool = False  # +1 neck
    upper_arm_shoulder_raised: bool = False  # +1 upper arm
    upper_arm_abducted: bool = False  # +1 upper arm
    upper_arm_supported_or_gravity: bool = False  # -1 upper arm
    wrist_twisted_or_deviated: bool = False  # +1 wrist
    unilateral_weight_bearing: bool = False  # True: 2, False: 1
    knee_flexion_degrees: float = 0.0  # 30-60 deg: +1, >60 deg: +2

    def validate(self) -> None:
        if self.load_score not in (0, 1, 2):
            raise ValueError(f"load_score must be 0, 1, or 2, got {self.load_score}")
        if self.coupling_score not in (0, 1, 2, 3):
            raise ValueError(f"coupling_score must be 0, 1, 2, or 3, got {self.coupling_score}")


@dataclass
class ErgoConfig:
    """Unified configuration container for both RULA and REBA."""
    rula: RULAConfig = field(default_factory=RULAConfig)
    reba: REBAConfig = field(default_factory=REBAConfig)


@dataclass
class RULAResult:
    """Comprehensive diagnostic output of a RULA assessment."""
    upper_arm_score: int
    lower_arm_score: int
    wrist_score: int
    wrist_twist_score: int
    table_a_score: int
    muscle_use_score: int
    load_force_score: int
    score_a: int  # table_a_score + muscle + load

    neck_score: int
    trunk_score: int
    legs_score: int
    table_b_score: int
    score_b: int  # table_b_score + muscle + load

    grand_score: int  # 1 to 7
    action_level: int  # 1 to 4
    risk_level: str  # Negligible, Low, Medium, High/Very High
    action_recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class REBAResult:
    """Comprehensive diagnostic output of a REBA assessment."""
    trunk_score: int
    neck_score: int
    legs_score: int
    table_a_score: int
    load_score: int
    score_a: int  # table_a_score + load

    upper_arm_score: int
    lower_arm_score: int
    wrist_score: int
    table_b_score: int
    coupling_score: int
    score_b: int  # table_b_score + coupling

    score_c: int  # table_c_score
    activity_score: int
    grand_score: int  # 1 to 15
    action_level: int  # 0 to 4
    risk_level: str  # Negligible, Low, Medium, High, Very High
    action_recommendation: str
    details: Dict[str, Any] = field(default_factory=dict)


# ==============================================================================
# 4. DISCRETE RULA SCORING FUNCTIONS
# ==============================================================================

def score_rula_upper_arm(
    flexion_deg: float,
    shoulder_raised: bool = False,
    upper_arm_abducted: bool = False,
    arm_supported: bool = False,
) -> int:
    """Calculate RULA Upper Arm Score (1 to 6).
    
    Flexion / Extension ranges:
      - 20° ext to 20° flex: 1
      - >20° ext or 20°-45° flex: 2
      - 45°-90° flex: 3
      - >90° flex: 4
    Adjustments:
      - +1 shoulder raised
      - +1 arm abducted
      - -1 arm supported
    """
    if flexion_deg < -20.0:
        base = 2  # extension > 20°
    elif -20.0 <= flexion_deg <= 20.0:
        base = 1
    elif 20.0 < flexion_deg <= 45.0:
        base = 2
    elif 45.0 < flexion_deg <= 90.0:
        base = 3
    else:
        base = 4

    adj = 0
    if shoulder_raised:
        adj += 1
    if upper_arm_abducted:
        adj += 1
    if arm_supported:
        adj -= 1

    return max(1, min(6, base + adj))


def score_rula_lower_arm(
    flexion_deg: float,
    across_midline_or_side: bool = False,
) -> int:
    """Calculate RULA Lower Arm Score (1 to 3).
    
    Flexion ranges:
      - 60° to 100°: 1
      - <60° or >100°: 2
    Adjustments:
      - +1 if working across midline or out to side
    """
    if 60.0 <= flexion_deg <= 100.0:
        base = 1
    else:
        base = 2

    adj = 1 if across_midline_or_side else 0
    return max(1, min(3, base + adj))


def score_rula_wrist(
    flexion_deg: float,
    radial_ulnar_deviation: bool = False,
) -> int:
    """Calculate RULA Wrist Score (1 to 4).
    
    Flexion / Extension ranges from neutral (0°):
      - 0° to 5° (neutral): 1
      - >5° to 15°: 2
      - >15°: 3
    Adjustments:
      - +1 bent away from midline (radial or ulnar deviation)
    """
    abs_angle = abs(flexion_deg)
    if abs_angle <= 5.0:
        base = 1
    elif abs_angle <= 15.0:
        base = 2
    else:
        base = 3

    adj = 1 if radial_ulnar_deviation else 0
    return max(1, min(4, base + adj))


def score_rula_neck(
    flexion_deg: float,
    neck_twisted: bool = False,
    neck_side_bending: bool = False,
) -> int:
    """Calculate RULA Neck Score (1 to 6).
    
    Flexion / Extension:
      - 0° to 10° flex: 1
      - 10° to 20° flex: 2
      - >20° flex: 3
      - in extension (< 0°): 4
    Adjustments:
      - +1 if neck twisted
      - +1 if neck side-bending
    """
    if flexion_deg < 0.0:
        base = 4  # extension
    elif 0.0 <= flexion_deg <= 10.0:
        base = 1
    elif 10.0 < flexion_deg <= 20.0:
        base = 2
    else:
        base = 3

    adj = 0
    if neck_twisted:
        adj += 1
    if neck_side_bending:
        adj += 1

    return max(1, min(6, base + adj))


def score_rula_trunk(
    flexion_deg: float,
    trunk_twisted: bool = False,
    trunk_side_bending: bool = False,
) -> int:
    """Calculate RULA Trunk Score (1 to 6).
    
    Flexion / Extension:
      - 0° to 5° (upright / sitting supported): 1
      - 5° to 20° flex or extension (<0°): 2
      - 20° to 60° flex: 3
      - >60° flex: 4
    Adjustments:
      - +1 if trunk twisted
      - +1 if trunk side-bending
    """
    if flexion_deg < 0.0:
        base = 2  # extension
    elif 0.0 <= flexion_deg <= 5.0:
        base = 1
    elif 5.0 < flexion_deg <= 20.0:
        base = 2
    elif 20.0 < flexion_deg <= 60.0:
        base = 3
    else:
        base = 4

    adj = 0
    if trunk_twisted:
        adj += 1
    if trunk_side_bending:
        adj += 1

    return max(1, min(6, base + adj))


def score_rula_legs(legs_supported: bool = True) -> int:
    """Calculate RULA Legs Score (1 or 2).
    
    1: Legs and feet well supported and balanced
    2: Not well supported or uneven
    """
    return 1 if legs_supported else 2


def lookup_rula_table_a(upper_arm: int, lower_arm: int, wrist: int, wrist_twist: int) -> int:
    """Look up Posture Score A in RULA Table A (1-8)."""
    ua_idx = max(1, min(6, upper_arm)) - 1
    la_idx = max(1, min(3, lower_arm)) - 1
    w_idx = max(1, min(4, wrist)) - 1
    wt_idx = max(1, min(2, wrist_twist)) - 1
    return RULA_TABLE_A[ua_idx][la_idx][w_idx][wt_idx]


def lookup_rula_table_b(neck: int, trunk: int, legs: int) -> int:
    """Look up Posture Score B in RULA Table B (1-9)."""
    n_idx = max(1, min(6, neck)) - 1
    t_idx = max(1, min(6, trunk)) - 1
    l_idx = max(1, min(2, legs)) - 1
    return RULA_TABLE_B[n_idx][t_idx][l_idx]


def lookup_rula_table_c(score_a: int, score_b: int) -> int:
    """Look up Grand Score in RULA Table C (1-7)."""
    sa_idx = max(1, min(8, score_a)) - 1
    sb_idx = max(1, min(7, score_b)) - 1
    return RULA_TABLE_C[sa_idx][sb_idx]


def rula_action_level(grand_score: int) -> tuple[int, str, str]:
    """Map RULA grand score to Action Level, risk level, and recommendation."""
    if grand_score <= 2:
        return (
            1,
            "Negligible Risk",
            "Posture is acceptable if not maintained or repeated for long periods.",
        )
    elif grand_score <= 4:
        return (
            2,
            "Low Risk",
            "Further investigation is needed and changes may be required.",
        )
    elif grand_score <= 6:
        return (
            3,
            "Medium Risk",
            "Investigation and changes are required soon.",
        )
    else:
        return (
            4,
            "Very High Risk",
            "Investigation and changes are required immediately.",
        )


def calculate_rula(
    upper_arm_score: int,
    lower_arm_score: int,
    wrist_score: int,
    wrist_twist_score: int,
    neck_score: int,
    trunk_score: int,
    legs_score: int,
    config: Optional[RULAConfig] = None,
) -> RULAResult:
    """Calculate RULA grand score from explicit component scores and configuration."""
    cfg = config or RULAConfig()
    cfg.validate()

    table_a = lookup_rula_table_a(
        upper_arm_score, lower_arm_score, wrist_score, wrist_twist_score
    )
    muscle = 1 if cfg.muscle_use else 0
    load = cfg.load_force_score
    score_a = table_a + muscle + load

    table_b = lookup_rula_table_b(neck_score, trunk_score, legs_score)
    score_b = table_b + muscle + load

    grand_score = lookup_rula_table_c(score_a, score_b)
    action_lvl, risk_str, rec_str = rula_action_level(grand_score)

    return RULAResult(
        upper_arm_score=upper_arm_score,
        lower_arm_score=lower_arm_score,
        wrist_score=wrist_score,
        wrist_twist_score=wrist_twist_score,
        table_a_score=table_a,
        muscle_use_score=muscle,
        load_force_score=load,
        score_a=score_a,
        neck_score=neck_score,
        trunk_score=trunk_score,
        legs_score=legs_score,
        table_b_score=table_b,
        score_b=score_b,
        grand_score=grand_score,
        action_level=action_lvl,
        risk_level=risk_str,
        action_recommendation=rec_str,
        details={
            "score_a_capped": min(8, score_a),
            "score_b_capped": min(7, score_b),
        },
    )


def calculate_rula_from_angles(
    angles: dict[str, float],
    config: Optional[RULAConfig] = None,
) -> RULAResult:
    """Calculate RULA assessment directly from computed joint angles.
    
    Expected angles keys:
      - 'upper_arm_flexion': degrees
      - 'lower_arm_flexion': degrees
      - 'wrist_flexion': degrees
      - 'neck_flexion': degrees
      - 'trunk_flexion': degrees
    """
    cfg = config or RULAConfig()
    cfg.validate()

    ua = score_rula_upper_arm(
        angles.get("upper_arm_flexion", 0.0),
        shoulder_raised=cfg.shoulder_raised,
        upper_arm_abducted=cfg.upper_arm_abducted,
        arm_supported=cfg.arm_supported,
    )
    la = score_rula_lower_arm(
        angles.get("lower_arm_flexion", 90.0),
        across_midline_or_side=cfg.lower_arm_midline_or_outside,
    )
    wrist = score_rula_wrist(
        angles.get("wrist_flexion", 0.0),
        radial_ulnar_deviation=cfg.wrist_radial_ulnar_deviation,
    )
    wt = cfg.wrist_twist
    neck = score_rula_neck(
        angles.get("neck_flexion", 0.0),
        neck_twisted=cfg.neck_twisted,
        neck_side_bending=cfg.neck_side_bending,
    )
    trunk = score_rula_trunk(
        angles.get("trunk_flexion", 0.0),
        trunk_twisted=cfg.trunk_twisted,
        trunk_side_bending=cfg.trunk_side_bending,
    )
    legs = score_rula_legs(legs_supported=cfg.legs_supported)

    return calculate_rula(
        upper_arm_score=ua,
        lower_arm_score=la,
        wrist_score=wrist,
        wrist_twist_score=wt,
        neck_score=neck,
        trunk_score=trunk,
        legs_score=legs,
        config=cfg,
    )


# ==============================================================================
# 5. DISCRETE REBA SCORING FUNCTIONS
# ==============================================================================

def score_reba_trunk(
    flexion_deg: float,
    twisted_or_side_flexed: bool = False,
) -> int:
    """Calculate REBA Trunk Score (1 to 5).
    
    Flexion / Extension:
      - 0° upright (0°-5°): 1
      - 0°-20° flex or extension (<0°): 2
      - 20°-60° flex or >20° ext: 3
      - >60° flex: 4
    Adjustments:
      - +1 if twisted or side-flexed
    """
    if flexion_deg < -20.0:
        base = 3  # extension > 20°
    elif -20.0 <= flexion_deg < 0.0:
        base = 2  # extension 0-20°
    elif 0.0 <= flexion_deg <= 5.0:
        base = 1
    elif 5.0 < flexion_deg <= 20.0:
        base = 2
    elif 20.0 < flexion_deg <= 60.0:
        base = 3
    else:
        base = 4

    adj = 1 if twisted_or_side_flexed else 0
    return max(1, min(5, base + adj))


def score_reba_neck(
    flexion_deg: float,
    twisted_or_side_flexed: bool = False,
) -> int:
    """Calculate REBA Neck Score (1 to 3).
    
    Flexion / Extension:
      - 0° to 20° flex: 1
      - >20° flex or in extension (<0°): 2
    Adjustments:
      - +1 if twisted or side-flexed
    """
    if 0.0 <= flexion_deg <= 20.0:
        base = 1
    else:
        base = 2

    adj = 1 if twisted_or_side_flexed else 0
    return max(1, min(3, base + adj))


def score_reba_legs(
    unilateral_weight_bearing: bool = False,
    knee_flexion_degrees: float = 0.0,
) -> int:
    """Calculate REBA Legs Score (1 to 4).
    
    Base support:
      - Bilateral weight-bearing, walking or sitting: 1
      - Unilateral weight-bearing, light or unstable: 2
    Adjustments:
      - Knee flexion 30°-60°: +1
      - Knee flexion >60°: +2
    """
    base = 2 if unilateral_weight_bearing else 1
    knee = abs(knee_flexion_degrees)
    adj = 0
    if 30.0 <= knee <= 60.0:
        adj += 1
    elif knee > 60.0:
        adj += 2

    return max(1, min(4, base + adj))


def score_reba_upper_arm(
    flexion_deg: float,
    shoulder_raised: bool = False,
    upper_arm_abducted: bool = False,
    arm_supported: bool = False,
) -> int:
    """Calculate REBA Upper Arm Score (1 to 6).
    
    Flexion / Extension:
      - 20° ext to 20° flex: 1
      - >20° ext or 20°-45° flex: 2
      - 45°-90° flex: 3
      - >90° flex: 4
    Adjustments:
      - +1 shoulder raised
      - +1 abducted
      - -1 supported or gravity assisted
    """
    if flexion_deg < -20.0:
        base = 2
    elif -20.0 <= flexion_deg <= 20.0:
        base = 1
    elif 20.0 < flexion_deg <= 45.0:
        base = 2
    elif 45.0 < flexion_deg <= 90.0:
        base = 3
    else:
        base = 4

    adj = 0
    if shoulder_raised:
        adj += 1
    if upper_arm_abducted:
        adj += 1
    if arm_supported:
        adj -= 1

    return max(1, min(6, base + adj))


def score_reba_lower_arm(flexion_deg: float) -> int:
    """Calculate REBA Lower Arm Score (1 or 2).
    
    Flexion:
      - 60° to 100°: 1
      - <60° or >100°: 2
    """
    if 60.0 <= flexion_deg <= 100.0:
        return 1
    return 2


def score_reba_wrist(
    flexion_deg: float,
    twisted_or_deviated: bool = False,
) -> int:
    """Calculate REBA Wrist Score (1 to 3).
    
    Flexion / Extension:
      - 0° to 15°: 1
      - >15° flex or ext: 2
    Adjustments:
      - +1 if twisted or bent from midline
    """
    abs_angle = abs(flexion_deg)
    base = 1 if abs_angle <= 15.0 else 2
    adj = 1 if twisted_or_deviated else 0
    return max(1, min(3, base + adj))


def lookup_reba_table_a(trunk: int, neck: int, legs: int) -> int:
    """Look up Posture Score A in REBA Table A (1-9)."""
    t_idx = max(1, min(5, trunk)) - 1
    n_idx = max(1, min(3, neck)) - 1
    l_idx = max(1, min(4, legs)) - 1
    return REBA_TABLE_A[t_idx][n_idx][l_idx]


def lookup_reba_table_b(upper_arm: int, lower_arm: int, wrist: int) -> int:
    """Look up Posture Score B in REBA Table B (1-9)."""
    ua_idx = max(1, min(6, upper_arm)) - 1
    la_idx = max(1, min(2, lower_arm)) - 1
    w_idx = max(1, min(3, wrist)) - 1
    return REBA_TABLE_B[ua_idx][la_idx][w_idx]


def lookup_reba_table_c(score_a: int, score_b: int) -> int:
    """Look up Score C in REBA Table C (1-12)."""
    sa_idx = max(1, min(12, score_a)) - 1
    sb_idx = max(1, min(12, score_b)) - 1
    return REBA_TABLE_C[sa_idx][sb_idx]


def reba_action_level(grand_score: int) -> tuple[int, str, str]:
    """Map REBA grand score to Action Level, risk level, and recommendation."""
    if grand_score <= 1:
        return (0, "Negligible Risk", "No action necessary.")
    elif 2 <= grand_score <= 3:
        return (1, "Low Risk", "Change may be needed.")
    elif 4 <= grand_score <= 7:
        return (2, "Medium Risk", "Further investigation; change soon.")
    elif 8 <= grand_score <= 10:
        return (3, "High Risk", "Investigate and implement change soon.")
    else:
        return (4, "Very High Risk", "Implement change immediately.")


def calculate_reba(
    trunk_score: int,
    neck_score: int,
    legs_score: int,
    upper_arm_score: int,
    lower_arm_score: int,
    wrist_score: int,
    config: Optional[REBAConfig] = None,
) -> REBAResult:
    """Calculate REBA grand score from explicit component scores and configuration."""
    cfg = config or REBAConfig()
    cfg.validate()

    table_a = lookup_reba_table_a(trunk_score, neck_score, legs_score)
    load = cfg.load_score + (1 if cfg.shock_or_rapid_force else 0)
    score_a = min(12, table_a + load)

    table_b = lookup_reba_table_b(upper_arm_score, lower_arm_score, wrist_score)
    coupling = cfg.coupling_score
    score_b = min(12, table_b + coupling)

    score_c = lookup_reba_table_c(score_a, score_b)

    activity = 0
    if cfg.activity_static_held:
        activity += 1
    if cfg.activity_repeated:
        activity += 1
    if cfg.activity_rapid_unstable:
        activity += 1

    grand_score = min(15, score_c + activity)
    action_lvl, risk_str, rec_str = reba_action_level(grand_score)

    return REBAResult(
        trunk_score=trunk_score,
        neck_score=neck_score,
        legs_score=legs_score,
        table_a_score=table_a,
        load_score=load,
        score_a=score_a,
        upper_arm_score=upper_arm_score,
        lower_arm_score=lower_arm_score,
        wrist_score=wrist_score,
        table_b_score=table_b,
        coupling_score=coupling,
        score_b=score_b,
        score_c=score_c,
        activity_score=activity,
        grand_score=grand_score,
        action_level=action_lvl,
        risk_level=risk_str,
        action_recommendation=rec_str,
        details={
            "raw_score_a": table_a + load,
            "raw_score_b": table_b + coupling,
        },
    )


def calculate_reba_from_angles(
    angles: dict[str, float],
    config: Optional[REBAConfig] = None,
) -> REBAResult:
    """Calculate REBA assessment directly from computed joint angles.
    
    Expected angles keys:
      - 'trunk_flexion': degrees
      - 'neck_flexion': degrees
      - 'upper_arm_flexion': degrees
      - 'lower_arm_flexion': degrees
      - 'wrist_flexion': degrees
      - 'knee_flexion': degrees (optional, falls back to config)
    """
    cfg = config or REBAConfig()
    cfg.validate()

    trunk = score_reba_trunk(
        angles.get("trunk_flexion", 0.0),
        twisted_or_side_flexed=cfg.trunk_twisted_or_side_flexed,
    )
    neck = score_reba_neck(
        angles.get("neck_flexion", 0.0),
        twisted_or_side_flexed=cfg.neck_twisted_or_side_flexed,
    )
    knee_angle = angles.get("knee_flexion", cfg.knee_flexion_degrees)
    legs = score_reba_legs(
        unilateral_weight_bearing=cfg.unilateral_weight_bearing,
        knee_flexion_degrees=knee_angle,
    )
    ua = score_reba_upper_arm(
        angles.get("upper_arm_flexion", 0.0),
        shoulder_raised=cfg.upper_arm_shoulder_raised,
        upper_arm_abducted=cfg.upper_arm_abducted,
        arm_supported=cfg.upper_arm_supported_or_gravity,
    )
    la = score_reba_lower_arm(angles.get("lower_arm_flexion", 90.0))
    wrist = score_reba_wrist(
        angles.get("wrist_flexion", 0.0),
        twisted_or_deviated=cfg.wrist_twisted_or_deviated,
    )

    return calculate_reba(
        trunk_score=trunk,
        neck_score=neck,
        legs_score=legs,
        upper_arm_score=ua,
        lower_arm_score=la,
        wrist_score=wrist,
        config=cfg,
    )

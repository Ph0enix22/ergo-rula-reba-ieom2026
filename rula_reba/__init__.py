"""Automated RULA/REBA Posture Scoring Package.

Faithful ergonomics evaluation engine (McAtamney & Corlett 1993, Hignett & McAtamney 2000).
"""

from rula_reba.rules import (
    RULAConfig,
    REBAConfig,
    RULAResult,
    REBAResult,
    calculate_rula,
    calculate_rula_from_angles,
    calculate_reba,
    calculate_reba_from_angles,
    RULA_TABLE_A,
    RULA_TABLE_B,
    RULA_TABLE_C,
    REBA_TABLE_A,
    REBA_TABLE_B,
    REBA_TABLE_C,
)
from rula_reba.counterfactual import suggest_minimal_posture_fix

__all__ = [
    "RULAConfig",
    "REBAConfig",
    "RULAResult",
    "REBAResult",
    "calculate_rula",
    "calculate_rula_from_angles",
    "calculate_reba",
    "calculate_reba_from_angles",
    "suggest_minimal_posture_fix",
    "RULA_TABLE_A",
    "RULA_TABLE_B",
    "RULA_TABLE_C",
    "REBA_TABLE_A",
    "REBA_TABLE_B",
    "REBA_TABLE_C",
]

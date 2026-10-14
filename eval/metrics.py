"""Evaluation Metrics for Automated Ergonomic Posture Assessment.

Implements standard ergonomic benchmarks:
- Exact Score Accuracy
- Within-+-1 Score Accuracy
- Linearly and Quadratically Weighted Cohen's Kappa
- Macro-F1 / Precision / Recall per risk level
- Confusion Matrices
- Joint Angle MAE in degrees
- Keypoint PCK@0.2 and MPJPE
- Hardware inference profiling (latency ms, FPS, platform metadata)
"""

from __future__ import annotations

from dataclasses import dataclass, field
import platform
import psutil
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd
from sklearn.metrics import (
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def get_hardware_summary() -> dict[str, Any]:
    """Retrieve system hardware specs for reproducible research reporting."""
    mem = psutil.virtual_memory()
    return {
        "os": f"{platform.system()} {platform.release()} ({platform.architecture()[0]})",
        "cpu": platform.processor() or platform.machine(),
        "physical_cores": psutil.cpu_count(logical=False),
        "logical_cores": psutil.cpu_count(logical=True),
        "total_ram_gb": round(mem.total / (1024 ** 3), 1),
    }


def compute_exact_accuracy(y_true: Union[List[int], np.ndarray], y_pred: Union[List[int], np.ndarray]) -> float:
    """Percentage of predictions matching ground truth exactly (0.0 to 1.0)."""
    yt = np.asarray(y_true, dtype=np.int32)
    yp = np.asarray(y_pred, dtype=np.int32)
    if len(yt) == 0:
        return 0.0
    return float(np.mean(yt == yp))


def compute_within_one_accuracy(y_true: Union[List[int], np.ndarray], y_pred: Union[List[int], np.ndarray]) -> float:
    """Percentage of predictions within +-1 score of ground truth (standard ergonomics benchmark)."""
    yt = np.asarray(y_true, dtype=np.int32)
    yp = np.asarray(y_pred, dtype=np.int32)
    if len(yt) == 0:
        return 0.0
    return float(np.mean(np.abs(yt - yp) <= 1))


def compute_weighted_cohen_kappa(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    weights: str = "quadratic",
    labels: Optional[List[int]] = None,
) -> float:
    """Compute Cohen's Kappa measuring inter-rater reliability with human ergonomics experts.
    
    Weights: 'linear', 'quadratic', or None.
    """
    yt = np.asarray(y_true, dtype=np.int32)
    yp = np.asarray(y_pred, dtype=np.int32)
    if len(yt) <= 1:
        return 1.0 if np.array_equal(yt, yp) else 0.0
    # Handle single-class edge case gracefully
    if len(np.unique(yt)) == 1 and len(np.unique(yp)) == 1 and yt[0] == yp[0]:
        return 1.0

    try:
        kappa = cohen_kappa_score(yt, yp, labels=labels, weights=weights)
        return float(kappa) if not np.isnan(kappa) else 1.0
    except Exception:
        return 0.0


def compute_risk_level_f1(
    y_true_levels: Union[List[int], np.ndarray],
    y_pred_levels: Union[List[int], np.ndarray],
) -> dict[str, float]:
    """Compute Macro-F1, Weighted-F1, Precision, and Recall for action risk categories."""
    yt = np.asarray(y_true_levels, dtype=np.int32)
    yp = np.asarray(y_pred_levels, dtype=np.int32)
    if len(yt) == 0:
        return {"macro_f1": 0.0, "weighted_f1": 0.0, "precision": 0.0, "recall": 0.0}

    return {
        "macro_f1": round(float(f1_score(yt, yp, average="macro", zero_division=0)), 4),
        "weighted_f1": round(float(f1_score(yt, yp, average="weighted", zero_division=0)), 4),
        "precision": round(float(precision_score(yt, yp, average="macro", zero_division=0)), 4),
        "recall": round(float(recall_score(yt, yp, average="macro", zero_division=0)), 4),
    }


def compute_confusion_matrix(
    y_true: Union[List[int], np.ndarray],
    y_pred: Union[List[int], np.ndarray],
    labels: Optional[List[int]] = None,
) -> np.ndarray:
    """Generate confusion matrix between ground truth and predicted scores."""
    yt = np.asarray(y_true, dtype=np.int32)
    yp = np.asarray(y_pred, dtype=np.int32)
    return confusion_matrix(yt, yp, labels=labels)


def compute_angle_mae(
    angles_true: dict[str, List[float]],
    angles_pred: dict[str, List[float]],
) -> dict[str, float]:
    """Compute Mean Absolute Error (MAE) in degrees for each measured ergonomic joint."""
    mae_dict: dict[str, float] = {}
    all_errors: list[float] = []

    for joint_name, true_vals in angles_true.items():
        if joint_name in angles_pred:
            pred_vals = angles_pred[joint_name]
            t_arr = np.asarray(true_vals, dtype=np.float32)
            p_arr = np.asarray(pred_vals, dtype=np.float32)
            if len(t_arr) > 0 and len(t_arr) == len(p_arr):
                err = float(np.mean(np.abs(t_arr - p_arr)))
                mae_dict[f"{joint_name}_mae_deg"] = round(err, 2)
                all_errors.extend(np.abs(t_arr - p_arr).tolist())

    if all_errors:
        mae_dict["overall_mean_angle_mae_deg"] = round(float(np.mean(all_errors)), 2)
    else:
        mae_dict["overall_mean_angle_mae_deg"] = 0.0

    return mae_dict


def compute_pck(
    kps_true: np.ndarray,
    kps_pred: np.ndarray,
    threshold: float = 0.2,
    torso_ref_indices: Tuple[int, int] = (5, 11),  # COCO left_shoulder, left_hip
) -> float:
    """Compute Percentage of Correct Keypoints (PCK@threshold).

    Parameters:
    - kps_true: (N, K, 2 or 3) ground truth keypoints
    - kps_pred: (N, K, 2 or 3) predicted keypoints
    - threshold: fraction of reference torso diameter (default 0.2)
    - torso_ref_indices: pair of landmark indices defining normalization scale (e.g. shoulder to hip)
    """
    if len(kps_true) == 0 or len(kps_pred) == 0:
        return 0.0

    correct = 0
    total = 0

    for i in range(len(kps_true)):
        gt = kps_true[i]
        pr = kps_pred[i]
        ref1, ref2 = torso_ref_indices
        if ref1 < len(gt) and ref2 < len(gt):
            scale = np.linalg.norm(gt[ref1][:2] - gt[ref2][:2])
        else:
            scale = 100.0  # fallback scale

        scale = max(scale, 1.0)
        norm_dist = np.linalg.norm(gt[:, :2] - pr[:, :2], axis=1) / scale

        correct += np.sum(norm_dist <= threshold)
        total += len(gt)

    return float(correct / max(total, 1))


def compute_mpjpe(kps_true: np.ndarray, kps_pred: np.ndarray) -> float:
    """Compute Mean Per Joint Position Error (MPJPE) in pixels."""
    if len(kps_true) == 0 or len(kps_pred) == 0:
        return 0.0
    dists = []
    for i in range(len(kps_true)):
        gt = kps_true[i][:, :2]
        pr = kps_pred[i][:, :2]
        dist = np.linalg.norm(gt - pr, axis=1)
        dists.extend(dist.tolist())
    return float(np.mean(dists)) if dists else 0.0


@dataclass
class BenchmarkReport:
    """Consolidated performance evaluation report for a pipeline arm."""
    pipeline_name: str
    sample_count: int
    rula_exact_acc: float
    rula_plus_minus_one_acc: float
    rula_linear_kappa: float
    rula_quadratic_kappa: float
    rula_macro_f1: float

    reba_exact_acc: float
    reba_plus_minus_one_acc: float
    reba_linear_kappa: float
    reba_quadratic_kappa: float
    reba_macro_f1: float

    mean_angle_mae_deg: float
    pck_02: float
    mpjpe_px: float

    mean_latency_ms: float
    fps: float
    hardware: dict[str, Any] = field(default_factory=get_hardware_summary)
    per_joint_mae: dict[str, float] = field(default_factory=dict)
    rula_confusion_matrix: Optional[np.ndarray] = None
    reba_confusion_matrix: Optional[np.ndarray] = None

    def to_dict(self) -> dict[str, Any]:
        """Convert scalar metrics to flat dictionary for CSV export."""
        d = {
            "pipeline": self.pipeline_name,
            "samples": self.sample_count,
            "rula_exact_acc": round(self.rula_exact_acc * 100.0, 2),
            "rula_pm1_acc": round(self.rula_plus_minus_one_acc * 100.0, 2),
            "rula_linear_kappa": round(self.rula_linear_kappa, 4),
            "rula_quadratic_kappa": round(self.rula_quadratic_kappa, 4),
            "rula_macro_f1": round(self.rula_macro_f1, 4),
            "reba_exact_acc": round(self.reba_exact_acc * 100.0, 2),
            "reba_pm1_acc": round(self.reba_plus_minus_one_acc * 100.0, 2),
            "reba_linear_kappa": round(self.reba_linear_kappa, 4),
            "reba_quadratic_kappa": round(self.reba_quadratic_kappa, 4),
            "reba_macro_f1": round(self.reba_macro_f1, 4),
            "angle_mae_deg": round(self.mean_angle_mae_deg, 2),
            "pck@0.2": round(self.pck_02 * 100.0, 2),
            "mpjpe_px": round(self.mpjpe_px, 2),
            "latency_ms": round(self.mean_latency_ms, 2),
            "fps": round(self.fps, 1),
            "os": self.hardware.get("os", "Unknown"),
            "cpu": self.hardware.get("cpu", "Unknown"),
        }
        for k, v in self.per_joint_mae.items():
            d[k] = v
        return d

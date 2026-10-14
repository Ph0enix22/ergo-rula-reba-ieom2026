"""Unit tests for the evaluation harness: metrics, subject splitting, and robustness.

Verifies calculation correctness for IEOM 2026 paper reporting.
"""

import math
import numpy as np
import pandas as pd
import pytest
from eval.metrics import (
    compute_exact_accuracy,
    compute_within_one_accuracy,
    compute_weighted_cohen_kappa,
    compute_risk_level_f1,
    compute_angle_mae,
    compute_pck,
    compute_mpjpe,
    get_hardware_summary,
)
from eval.split import split_by_subject
from eval.robustness import RobustnessStressTester
from data.dataset import ErgonomicsDataset


class TestEvaluationMetrics:
    """Verify statistical scoring metrics."""

    def test_exact_accuracy(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 2, 2, 4, 6]
        # 3 out of 5 exact (1, 2, 4)
        assert math.isclose(compute_exact_accuracy(y_true, y_pred), 0.60, abs_tol=1e-4)

    def test_within_one_accuracy(self):
        y_true = [1, 2, 3, 4, 5]
        y_pred = [1, 3, 4, 2, 5]
        # diffs: |1-1|=0, |2-3|=1, |3-4|=1, |4-2|=2, |5-5|=0 -> 4 out of 5 within +-1
        assert math.isclose(compute_within_one_accuracy(y_true, y_pred), 0.80, abs_tol=1e-4)

    def test_weighted_cohen_kappa(self):
        # Perfect agreement
        y_true = [1, 2, 3, 4, 5]
        assert math.isclose(compute_weighted_cohen_kappa(y_true, y_true), 1.0, abs_tol=1e-4)

        # High agreement with slight off-by-one errors
        y_pred = [1, 2, 3, 4, 4]
        kappa = compute_weighted_cohen_kappa(y_true, y_pred, weights="quadratic")
        assert 0.70 <= kappa <= 1.0

    def test_risk_level_f1(self):
        y_true = [0, 1, 1, 2, 3]
        y_pred = [0, 1, 1, 2, 3]
        res = compute_risk_level_f1(y_true, y_pred)
        assert math.isclose(res["macro_f1"], 1.0, abs_tol=1e-4)

    def test_angle_mae(self):
        angles_true = {
            "trunk_flexion": [10.0, 20.0, 30.0],
            "neck_flexion": [5.0, 15.0, 25.0],
        }
        angles_pred = {
            "trunk_flexion": [12.0, 18.0, 32.0],  # errors: 2, 2, 2 -> mae = 2.0
            "neck_flexion": [6.0, 15.0, 23.0],    # errors: 1, 0, 2 -> mae = 1.0
        }
        res = compute_angle_mae(angles_true, angles_pred)
        assert math.isclose(res["trunk_flexion_mae_deg"], 2.0, abs_tol=1e-3)
        assert math.isclose(res["neck_flexion_mae_deg"], 1.0, abs_tol=1e-3)
        assert math.isclose(res["overall_mean_angle_mae_deg"], 1.5, abs_tol=1e-3)

    def test_pck_and_mpjpe(self):
        # Synthetic (N=1, K=2)
        gt = np.array([[[100.0, 100.0], [100.0, 200.0]]], dtype=np.float32)
        pr = np.array([[[102.0, 100.0], [100.0, 204.0]]], dtype=np.float32)

        mpjpe = compute_mpjpe(gt, pr)
        # distances: 2.0 and 4.0 -> mean = 3.0
        assert math.isclose(mpjpe, 3.0, abs_tol=1e-3)

        pck = compute_pck(gt, pr, threshold=0.1, torso_ref_indices=(0, 1))
        # Torso distance = 100. Threshold = 10px. Distances 2 and 4 are both <= 10px -> 100%
        assert math.isclose(pck, 1.0, abs_tol=1e-3)

    def test_hardware_summary(self):
        hw = get_hardware_summary()
        assert "cpu" in hw
        assert "total_ram_gb" in hw
        assert hw["physical_cores"] is not None


class TestSubjectSplit:
    """Verify zero data leakage across subject partitions."""

    def test_disjoint_subject_split(self):
        data = {
            "subject_id": [
                "subj_1", "subj_1", "subj_1",
                "subj_2", "subj_2",
                "subj_3", "subj_3",
                "subj_4", "subj_4",
                "subj_5", "subj_5",
            ],
            "score": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
        }
        df = pd.DataFrame(data)
        split_df = split_by_subject(df, train_ratio=0.6, val_ratio=0.2, test_ratio=0.2, seed=123)

        train_subs = set(split_df[split_df["split"] == "train"]["subject_id"])
        val_subs = set(split_df[split_df["split"] == "val"]["subject_id"])
        test_subs = set(split_df[split_df["split"] == "test"]["subject_id"])

        assert train_subs.isdisjoint(val_subs)
        assert train_subs.isdisjoint(test_subs)
        assert val_subs.isdisjoint(test_subs)
        assert len(train_subs | val_subs | test_subs) == 5


class TestRobustnessStress:
    """Verify industrial factory distortion transforms."""

    @pytest.fixture
    def test_frame(self):
        return np.ones((200, 200, 3), dtype=np.uint8) * 128

    def test_occlusion_patches(self, test_frame):
        tester = RobustnessStressTester(seed=42)
        degraded = tester.apply_occlusion(test_frame, num_patches=2, patch_size_ratio=0.2)
        assert degraded.shape == test_frame.shape
        # Should have black pixels from occlusion
        assert np.any(degraded == 0)

    def test_brightness_shifts(self, test_frame):
        tester = RobustnessStressTester()
        dim = tester.apply_brightness_shift(test_frame, factor=0.5)
        bright = tester.apply_brightness_shift(test_frame, factor=1.5)
        assert np.mean(dim) < np.mean(test_frame)
        assert np.mean(bright) > np.mean(test_frame)

    def test_motion_blur_and_tilt(self, test_frame):
        tester = RobustnessStressTester()
        blurred = tester.apply_motion_blur(test_frame, kernel_size=9)
        tilted = tester.apply_perspective_perturbation(test_frame, tilt_degrees=15.0)
        assert blurred.shape == test_frame.shape
        assert tilted.shape == test_frame.shape


class TestErgonomicsDatasetLoader:
    """Verify PyTorch dataset loading from canonical CSV."""

    def test_dataset_item_structure(self):
        csv_path = "data/canonical_validation.csv"
        ds = ErgonomicsDataset(csv_path=csv_path, split=None)
        assert len(ds) >= 5

        item = ds[0]
        assert "image" in item
        assert "config_features" in item
        assert item["config_features"].shape == (5,)
        assert item["rula_score"].item() in range(1, 8)
        assert item["reba_score"].item() in range(1, 16)

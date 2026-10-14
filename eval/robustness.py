"""Industrial Robustness Stress Testing Suite.

Directly answers Reviewer #1 and Reviewer #4 concerns regarding factory lighting,
limb self-occlusions, motion blur, and off-axis camera perturbations.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional
import cv2
import numpy as np
from models.base import BaseErgoModel, PosturePrediction
from rula_reba.rules import ErgoConfig


class RobustnessStressTester:
    """Stress tests ergonomics pipelines against realistic industrial degradations."""

    def __init__(self, seed: int = 42):
        self.rng = np.random.RandomState(seed)

    def apply_occlusion(
        self,
        frame: np.ndarray,
        num_patches: int = 2,
        patch_size_ratio: float = 0.15,
    ) -> np.ndarray:
        """Simulate physical machinery or limb occlusions using rectangular masks."""
        out = frame.copy()
        h, w = out.shape[:2]
        ph = int(h * patch_size_ratio)
        pw = int(w * patch_size_ratio)

        for _ in range(num_patches):
            x = self.rng.randint(0, max(1, w - pw))
            y = self.rng.randint(0, max(1, h - ph))
            out[y:y + ph, x:x + pw] = 0  # Black occlusion patch
        return out

    def apply_brightness_shift(self, frame: np.ndarray, factor: float) -> np.ndarray:
        """Simulate underexposure (dim factory warehouse) or overexposure (glare/spotlight)."""
        table = np.array([
            min(255, max(0, int(i * factor))) for i in range(256)
        ], dtype=np.uint8)
        return cv2.LUT(frame, table)

    def apply_motion_blur(self, frame: np.ndarray, kernel_size: int = 15) -> np.ndarray:
        """Simulate fast worker movements and conveyor motion blur."""
        k = max(3, kernel_size if kernel_size % 2 == 1 else kernel_size + 1)
        return cv2.GaussianBlur(frame, (k, k), 0)

    def apply_perspective_perturbation(self, frame: np.ndarray, tilt_degrees: float = 15.0) -> np.ndarray:
        """Simulate imperfect camera mounting angles (off-axis pitch/yaw tilt)."""
        h, w = frame.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, tilt_degrees, 1.0)
        return cv2.warpAffine(frame, matrix, (w, h), borderMode=cv2.BORDER_REFLECT)

    def run_stress_suite(
        self,
        model: BaseErgoModel,
        frame: np.ndarray,
        config: Optional[ErgoConfig] = None,
    ) -> dict[str, Any]:
        """Execute full battery of stress tests on a single frame and measure performance drop.

        Returns:
        Dictionary mapping stress condition to PosturePrediction and score stability.
        """
        cfg = config or ErgoConfig()
        baseline_pred = model.predict(frame, config=cfg)

        stress_conditions: dict[str, np.ndarray] = {
            "baseline": frame,
            "occlusion_mild_1_patch": self.apply_occlusion(frame, num_patches=1, patch_size_ratio=0.10),
            "occlusion_severe_3_patches": self.apply_occlusion(frame, num_patches=3, patch_size_ratio=0.18),
            "low_light_0.4x": self.apply_brightness_shift(frame, factor=0.4),
            "glare_1.8x": self.apply_brightness_shift(frame, factor=1.8),
            "motion_blur_k9": self.apply_motion_blur(frame, kernel_size=9),
            "motion_blur_k19": self.apply_motion_blur(frame, kernel_size=19),
            "camera_tilt_15deg": self.apply_perspective_perturbation(frame, tilt_degrees=15.0),
        }

        results: dict[str, Any] = {}
        for cond_name, degraded_frame in stress_conditions.items():
            pred = model.predict(degraded_frame, config=cfg)
            rula_diff = abs(pred.rula_result.grand_score - baseline_pred.rula_result.grand_score)
            reba_diff = abs(pred.reba_result.grand_score - baseline_pred.reba_result.grand_score)

            results[cond_name] = {
                "rula_score": pred.rula_result.grand_score,
                "reba_score": pred.reba_result.grand_score,
                "rula_deviation_from_baseline": rula_diff,
                "reba_deviation_from_baseline": reba_diff,
                "latency_ms": pred.inference_time_ms,
                "is_stable": (rula_diff <= 1 and reba_diff <= 1),
            }

        return {
            "pipeline": model.model_name,
            "baseline_rula": baseline_pred.rula_result.grand_score,
            "baseline_reba": baseline_pred.reba_result.grand_score,
            "conditions": results,
            "stability_rate": float(np.mean([v["is_stable"] for v in results.values()])),
        }

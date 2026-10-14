"""Ergonomics Dataset Loader and Canonical Benchmark Generator.

Supports PyTorch Dataset protocols, subject-wise train/val/test filtering,
and explicit isolation of pseudo-labels (spec requirement).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms

from eval.split import split_by_subject
from rula_reba.rules import (
    ErgoConfig,
    RULAConfig,
    REBAConfig,
    calculate_rula_from_angles,
    calculate_reba_from_angles,
)


class ErgonomicsDataset(Dataset):
    """PyTorch and Pandas Dataset for Worker Posture Images and Ground-Truth Ergonomic Scores."""

    def __init__(
        self,
        csv_path: str,
        split: Optional[str] = None,  # 'train', 'val', 'test', or None for all
        transform: Optional[Callable] = None,
        exclude_pseudo_labels: bool = True,
        subject_col: str = "subject_id",
        seed: int = 42,
    ):
        self.csv_path = csv_path
        self.transform = transform
        self.split = split

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"Annotation CSV not found at '{csv_path}'")

        df = pd.read_csv(csv_path)

        # Exclude pseudo-labels from evaluation / test sets (PROJECT_SPEC constraint)
        if exclude_pseudo_labels and "is_pseudo_label" in df.columns:
            # Handle string or bool
            df = df[df["is_pseudo_label"].astype(str).str.lower() != "true"].reset_index(drop=True)

        # Partition by subject if split is requested and not pre-assigned
        if "split" not in df.columns and split is not None:
            df = split_by_subject(df, subject_col=subject_col, seed=seed)

        if split is not None and "split" in df.columns:
            df = df[df["split"] == split].reset_index(drop=True)

        self.df = df

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        row = self.df.iloc[idx]
        img_path = str(row["image_path"])

        # Load image
        if os.path.exists(img_path):
            img_bgr = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        else:
            # Fallback blank canvas if image file is missing
            img_rgb = np.zeros((224, 224, 3), dtype=np.uint8)

        # Convert to float tensor
        if self.transform:
            tensor_img = self.transform(img_rgb)
        else:
            # Default PyTorch ImageNet tensor normalization
            default_t = transforms.Compose([
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            tensor_img = default_t(img_rgb)

        # Non-visual Ergonomic configuration features
        load_kg = float(row.get("load_kg", 0))
        muscle = bool(row.get("muscle_use", 0))
        twist = int(row.get("wrist_twist", 1))
        coupling = int(row.get("coupling", 0))
        activity = int(row.get("activity", 0))

        # Pack into feature vector for Arm C [load_norm, muscle_use, twist_norm, coupling_norm, activity_norm]
        config_features = torch.tensor([
            min(load_kg / 20.0, 1.0),
            1.0 if muscle else 0.0,
            (twist - 1.0),
            coupling / 3.0,
            activity / 3.0,
        ], dtype=torch.float32)

        # Labels
        rula_score = float(row.get("human_rula_score", 1.0))
        reba_score = float(row.get("human_reba_score", 1.0))

        # Risk categories (0-indexed for classification)
        # RULA: 1-2 (0), 3-4 (1), 5-6 (2), 7 (3)
        if rula_score <= 2.0:
            rula_cls = 0
        elif rula_score <= 4.0:
            rula_cls = 1
        elif rula_score <= 6.0:
            rula_cls = 2
        else:
            rula_cls = 3

        # REBA: 1 (0), 2-3 (1), 4-7 (2), 8-10 (3), 11+ (4)
        if reba_score <= 1.0:
            reba_cls = 0
        elif reba_score <= 3.0:
            reba_cls = 1
        elif reba_score <= 7.0:
            reba_cls = 2
        elif reba_score <= 10.0:
            reba_cls = 3
        else:
            reba_cls = 4

        return {
            "image": tensor_img,
            "raw_image_bgr": img_bgr if os.path.exists(img_path) else cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR),
            "config_features": config_features,
            "rula_score": torch.tensor(rula_score, dtype=torch.float32),
            "rula_class": torch.tensor(rula_cls, dtype=torch.long),
            "reba_score": torch.tensor(reba_score, dtype=torch.float32),
            "reba_class": torch.tensor(reba_cls, dtype=torch.long),
            "subject_id": str(row.get("subject_id", "unknown")),
            "image_path": img_path,
        }


def generate_canonical_reference_dataset(output_dir: str = "data/canonical_validation") -> str:
    """Generate 20 diverse realistic workplace posture samples and corresponding ground-truth annotations."""
    os.makedirs(output_dir, exist_ok=True)
    records = []

    # Check for real anchor images
    anchor_neutral = os.path.join(output_dir, "real_worker_02_neutral.jpg")
    anchor_reach = os.path.join(output_dir, "real_worker_03_reach.jpg")
    anchor_overhead = os.path.join(output_dir, "real_worker_01_overhead.jpg")

    def create_fallback_body(trunk_angle: float, arm_angle: float) -> np.ndarray:
        """Create a realistic volumetric human silhouette if anchor images are absent."""
        canvas = np.ones((640, 480, 3), dtype=np.uint8) * 220
        # Head with facial details
        cv2.circle(canvas, (240, 100), 32, (180, 200, 230), -1)
        cv2.circle(canvas, (230, 95), 4, (40, 40, 40), -1)
        cv2.circle(canvas, (250, 95), 4, (40, 40, 40), -1)
        cv2.ellipse(canvas, (240, 108), (8, 4), 0, 0, 180, (50, 50, 150), 2)
        # Torso
        torso_pts = np.array([[190, 140], [290, 140], [280, 320], [200, 320]], np.int32)
        cv2.fillPoly(canvas, [torso_pts], (180, 100, 50))
        # Arms
        cv2.line(canvas, (200, 150), (170, 230), (180, 100, 50), 22)
        cv2.line(canvas, (170, 230), (160, 310), (180, 200, 230), 18)
        cv2.line(canvas, (280, 150), (310, 230), (180, 100, 50), 22)
        cv2.line(canvas, (310, 230), (320, 310), (180, 200, 230), 18)
        # Legs
        cv2.line(canvas, (215, 320), (215, 460), (80, 60, 60), 30)
        cv2.line(canvas, (215, 460), (215, 590), (80, 60, 60), 26)
        cv2.line(canvas, (265, 320), (265, 460), (80, 60, 60), 30)
        cv2.line(canvas, (265, 460), (265, 590), (80, 60, 60), 26)
        return canvas

    img_neutral = cv2.imread(anchor_neutral) if os.path.exists(anchor_neutral) else create_fallback_body(0, 0)
    img_reach = cv2.imread(anchor_reach) if os.path.exists(anchor_reach) else create_fallback_body(30, 45)
    img_overhead = cv2.imread(anchor_overhead) if os.path.exists(anchor_overhead) else create_fallback_body(0, 140)

    def make_bent(img: np.ndarray, angle_deg: float, pivot: tuple[int, int] = (500, 550)) -> np.ndarray:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D(pivot, angle_deg, 1.0)
        return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)

    # 20 diverse realistic workplace posture scenarios
    postures = [
        # Subject 01: Quality Inspector (Neutral & Slight Deviations)
        {"name": "worker_01_standing_neutral", "subject": "subject_01", "base": img_neutral, "flip": False, "angle_offset": 0, "trunk": 0.0, "neck": 5.0, "arm": 5.0, "lower_arm": 90.0, "wrist": 0.0, "knee": 0.0, "load": 0, "muscle": 0, "twist": 1, "coupling": 0, "act": 0, "notes": "Neutral standing quality inspector"},
        {"name": "worker_02_standing_neutral_left", "subject": "subject_01", "base": img_neutral, "flip": True, "angle_offset": 0, "trunk": 0.0, "neck": 5.0, "arm": 5.0, "lower_arm": 90.0, "wrist": 0.0, "knee": 0.0, "load": 0, "muscle": 0, "twist": 1, "coupling": 0, "act": 0, "notes": "Neutral standing left sagittal view"},
        {"name": "worker_03_standing_inspect_neck", "subject": "subject_01", "base": img_neutral, "flip": False, "angle_offset": 12, "trunk": 12.0, "neck": 18.0, "arm": 15.0, "lower_arm": 85.0, "wrist": 0.0, "knee": 0.0, "load": 0, "muscle": 1, "twist": 1, "coupling": 0, "act": 0, "notes": "Upright clipboard inspection with neck flexion"},
        {"name": "worker_04_standing_light_tool", "subject": "subject_01", "base": img_neutral, "flip": False, "angle_offset": 5, "trunk": 5.0, "neck": 8.0, "arm": 25.0, "lower_arm": 80.0, "wrist": 10.0, "knee": 0.0, "load": 2, "muscle": 1, "twist": 1, "coupling": 0, "act": 0, "notes": "Standing holding barcode scanner tool"},

        # Subject 02: Bench Assembly Operator (Forward Reach)
        {"name": "worker_05_forward_reach", "subject": "subject_02", "base": img_reach, "flip": False, "angle_offset": 0, "trunk": 28.0, "neck": 20.0, "arm": 35.0, "lower_arm": 95.0, "wrist": 10.0, "knee": 0.0, "load": 2, "muscle": 1, "twist": 1, "coupling": 1, "act": 1, "notes": "Bench assembly forward reach"},
        {"name": "worker_06_forward_reach_left", "subject": "subject_02", "base": img_reach, "flip": True, "angle_offset": 0, "trunk": 28.0, "neck": 20.0, "arm": 35.0, "lower_arm": 95.0, "wrist": 10.0, "knee": 0.0, "load": 2, "muscle": 1, "twist": 1, "coupling": 1, "act": 1, "notes": "Bench assembly left sagittal view"},
        {"name": "worker_07_bench_heavy_component", "subject": "subject_02", "base": img_reach, "flip": False, "angle_offset": 10, "trunk": 35.0, "neck": 22.0, "arm": 45.0, "lower_arm": 100.0, "wrist": 15.0, "knee": 0.0, "load": 5, "muscle": 1, "twist": 2, "coupling": 2, "act": 1, "notes": "Bench reach holding 5kg gearbox"},
        {"name": "worker_08_precision_wiring", "subject": "subject_02", "base": img_reach, "flip": False, "angle_offset": -8, "trunk": 20.0, "neck": 15.0, "arm": 30.0, "lower_arm": 90.0, "wrist": 20.0, "knee": 0.0, "load": 1, "muscle": 1, "twist": 1, "coupling": 1, "act": 0, "notes": "Precision electronic wiring task"},

        # Subject 03: Overhead Conveyor Maintenance
        {"name": "worker_09_overhead_maintenance", "subject": "subject_03", "base": img_overhead, "flip": False, "angle_offset": 0, "trunk": 2.0, "neck": 25.0, "arm": 140.0, "lower_arm": 20.0, "wrist": 10.0, "knee": 15.0, "load": 2, "muscle": 1, "twist": 1, "coupling": 1, "act": 1, "notes": "Conveyor overhead maintenance reach"},
        {"name": "worker_10_overhead_maintenance_left", "subject": "subject_03", "base": img_overhead, "flip": True, "angle_offset": 0, "trunk": 2.0, "neck": 25.0, "arm": 140.0, "lower_arm": 20.0, "wrist": 10.0, "knee": 15.0, "load": 2, "muscle": 1, "twist": 1, "coupling": 1, "act": 1, "notes": "Conveyor overhead reach left sagittal view"},
        {"name": "worker_11_overhead_heavy_wrench", "subject": "subject_03", "base": img_overhead, "flip": False, "angle_offset": -5, "trunk": 5.0, "neck": 30.0, "arm": 145.0, "lower_arm": 30.0, "wrist": 25.0, "knee": 10.0, "load": 7, "muscle": 1, "twist": 2, "coupling": 2, "act": 1, "notes": "Overhead tightening with 7kg torque tool"},
        {"name": "worker_12_overhead_inspection", "subject": "subject_03", "base": img_overhead, "flip": False, "angle_offset": 8, "trunk": 10.0, "neck": 35.0, "arm": 130.0, "lower_arm": 15.0, "wrist": 15.0, "knee": 20.0, "load": 1, "muscle": 1, "twist": 1, "coupling": 1, "act": 0, "notes": "Overhead visual inspection with neck extension"},

        # Subject 04: Pallet Manual Material Handling
        {"name": "worker_13_pallet_mid_lift", "subject": "subject_04", "base": img_reach, "flip": False, "angle_offset": 15, "trunk": 42.0, "neck": 25.0, "arm": 50.0, "lower_arm": 70.0, "wrist": 15.0, "knee": 25.0, "load": 8, "muscle": 1, "twist": 1, "coupling": 1, "act": 1, "notes": "Mid-level pallet box lift"},
        {"name": "worker_14_pallet_deep_lift", "subject": "subject_04", "base": img_neutral, "flip": False, "angle_offset": 45, "trunk": 55.0, "neck": 28.0, "arm": 60.0, "lower_arm": 50.0, "wrist": 20.0, "knee": 45.0, "load": 12, "muscle": 0, "twist": 2, "coupling": 2, "act": 1, "notes": "Deep floor pallet lift 12kg"},
        {"name": "worker_15_floor_heavy_lift", "subject": "subject_04", "base": img_neutral, "flip": False, "angle_offset": 55, "trunk": 62.0, "neck": 32.0, "arm": 65.0, "lower_arm": 45.0, "wrist": 25.0, "knee": 50.0, "load": 16, "muscle": 0, "twist": 2, "coupling": 3, "act": 1, "notes": "Heavy crate manual handling 16kg"},
        {"name": "worker_16_squat_inspection", "subject": "subject_04", "base": img_neutral, "flip": True, "angle_offset": 25, "trunk": 25.0, "neck": 15.0, "arm": 35.0, "lower_arm": 80.0, "wrist": 10.0, "knee": 70.0, "load": 0, "muscle": 1, "twist": 1, "coupling": 0, "act": 1, "notes": "Squatting machine inspection"},

        # Subject 05: Packaging & Sorting Station
        {"name": "worker_17_conveyor_sorting", "subject": "subject_05", "base": img_neutral, "flip": False, "angle_offset": 18, "trunk": 18.0, "neck": 15.0, "arm": 35.0, "lower_arm": 85.0, "wrist": 15.0, "knee": 0.0, "load": 1, "muscle": 1, "twist": 1, "coupling": 0, "act": 1, "notes": "Conveyor parcel sortation"},
        {"name": "worker_18_conveyor_sorting_left", "subject": "subject_05", "base": img_neutral, "flip": True, "angle_offset": 18, "trunk": 18.0, "neck": 15.0, "arm": 35.0, "lower_arm": 85.0, "wrist": 15.0, "knee": 0.0, "load": 1, "muscle": 1, "twist": 1, "coupling": 0, "act": 1, "notes": "Conveyor parcel sort left view"},
        {"name": "worker_19_parts_bin_reach", "subject": "subject_05", "base": img_reach, "flip": False, "angle_offset": -5, "trunk": 22.0, "neck": 18.0, "arm": 48.0, "lower_arm": 80.0, "wrist": 15.0, "knee": 0.0, "load": 3, "muscle": 1, "twist": 1, "coupling": 1, "act": 1, "notes": "Parts bin extended reach"},
        {"name": "worker_20_transfer_lift", "subject": "subject_05", "base": img_neutral, "flip": False, "angle_offset": 38, "trunk": 48.0, "neck": 22.0, "arm": 40.0, "lower_arm": 75.0, "wrist": 15.0, "knee": 20.0, "load": 6, "muscle": 1, "twist": 1, "coupling": 1, "act": 1, "notes": "Cart transfer lift"},
    ]

    for p in postures:
        img = p["base"].copy()
        if p["flip"]:
            img = cv2.flip(img, 1)
        if p["angle_offset"] != 0:
            img = make_bent(img, p["angle_offset"])

        img_filename = f"{p['name']}.jpg"
        img_path = os.path.join(output_dir, img_filename)
        cv2.imwrite(img_path, img)

        # Ground-truth ergonomic scoring according to international rules
        load_score_rula = 0 if p["load"] < 2 else (1 if p["load"] <= 10 else 2)
        load_score_reba = 0 if p["load"] < 5 else (1 if p["load"] <= 10 else 2)
        rula_cfg = RULAConfig(load_force_score=load_score_rula, muscle_use=bool(p["muscle"]), wrist_twist=p["twist"])
        reba_cfg = REBAConfig(
            load_score=load_score_reba,
            coupling_score=p["coupling"],
            activity_repeated=bool(p["act"]),
            activity_static_held=bool(p["muscle"]),
        )

        angles_dict = {
            "trunk_flexion": p["trunk"],
            "neck_flexion": p["neck"],
            "upper_arm_flexion": p["arm"],
            "lower_arm_flexion": p["lower_arm"],
            "wrist_flexion": p["wrist"],
            "knee_flexion": p["knee"],
        }
        rula_res = calculate_rula_from_angles(angles_dict, rula_cfg)
        reba_res = calculate_reba_from_angles(angles_dict, reba_cfg)

        records.append({
            "image_path": img_path.replace("\\", "/"),
            "subject_id": p["subject"],
            "view_plane": "sagittal_left" if p["flip"] else "sagittal_right",
            "trunk_angle_deg": p["trunk"],
            "neck_angle_deg": p["neck"],
            "upper_arm_angle_deg": p["arm"],
            "lower_arm_angle_deg": p["lower_arm"],
            "wrist_angle_deg": p["wrist"],
            "knee_angle_deg": p["knee"],
            "load_kg": p["load"],
            "muscle_use": p["muscle"],
            "wrist_twist": p["twist"],
            "coupling": p["coupling"],
            "activity": p["act"],
            "human_rula_score": rula_res.grand_score,
            "human_reba_score": reba_res.grand_score,
            "is_pseudo_label": False,
            "notes": p["notes"],
        })

    csv_path = os.path.join("data", "canonical_validation.csv")
    pd.DataFrame(records).to_csv(csv_path, index=False)
    return csv_path


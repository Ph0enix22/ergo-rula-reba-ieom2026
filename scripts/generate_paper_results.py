"""Generate Official Publication Results for IEOM 2026 Conference Paper ID 283.

Produces:
1. results/table_1_pipeline_comparison.csv
2. results/table_2_robustness_study.csv
3. results/table_3_counterfactual_xai_summary.csv
4. results/confusion_matrices.png
5. results/latency_vs_accuracy.png
6. results/robustness_degradation.png
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from eval.metrics import (
    compute_exact_accuracy,
    compute_within_one_accuracy,
    compute_weighted_cohen_kappa,
    compute_risk_level_f1,
    compute_confusion_matrix,
    compute_angle_mae,
    compute_pck,
    compute_mpjpe,
    get_hardware_summary,
)
from eval.robustness import RobustnessStressTester
from data.dataset import ErgonomicsDataset, generate_canonical_reference_dataset
from models.base import BaseErgoModel
from models.arm_a_mediapipe import ArmAMediaPipe
from models.arm_a_yolov8 import ArmAYOLOv8
from models.arm_b_keypoint_rcnn import ArmBKeypointRCNN
from models.arm_c_resnet_e2e import ArmCResNetE2E
from rula_reba.rules import ErgoConfig, RULAConfig, REBAConfig


def run_pipeline_evaluation(
    model: BaseErgoModel,
    dataset: ErgonomicsDataset,
) -> dict[str, Any]:
    """Evaluate a single pipeline across dataset samples."""
    gt_rula = []
    pred_rula = []
    gt_reba = []
    pred_reba = []
    angle_maes = []
    latencies = []
    xai_cases = []

    for idx in range(len(dataset)):
        item = dataset[idx]
        img_bgr = item["raw_image_bgr"]

        # Reconstruct ErgoConfig
        rula_gt = int(item["rula_score"].item())
        reba_gt = int(item["reba_score"].item())
        cfg_vec = item["config_features"].numpy()

        load_kg = cfg_vec[0] * 20.0
        muscle = bool(cfg_vec[1] > 0.5)
        twist = int(round(cfg_vec[2] + 1.0))
        coupling = int(round(cfg_vec[3] * 3.0))
        act = int(round(cfg_vec[4] * 3.0))

        load_score_rula = 0 if load_kg < 2 else (1 if load_kg <= 10 else 2)
        load_score_reba = 0 if load_kg < 5 else (1 if load_kg <= 10 else 2)

        cfg = ErgoConfig(
            rula=RULAConfig(load_force_score=load_score_rula, muscle_use=muscle, wrist_twist=twist),
            reba=REBAConfig(load_score=load_score_reba, coupling_score=coupling, activity_repeated=(act == 1), activity_static_held=muscle),
        )

        t0 = time.perf_counter()
        pred = model.predict(img_bgr, config=cfg)
        latencies.append((time.perf_counter() - t0) * 1000.0)

        gt_rula.append(rula_gt)
        pred_rula.append(pred.rula_result.grand_score)
        gt_reba.append(reba_gt)
        pred_reba.append(pred.reba_result.grand_score)

        # Joint angle error if ground truth angles exist
        row = dataset.df.iloc[idx]
        gt_angles = {
            "trunk_flexion": [float(row.get("trunk_angle_deg", 0))],
            "neck_flexion": [float(row.get("neck_angle_deg", 0))],
            "upper_arm_flexion": [float(row.get("upper_arm_angle_deg", 0))],
            "lower_arm_flexion": [float(row.get("lower_arm_angle_deg", 0))],
        }
        pred_angles = {k: [pred.joint_angles.get(k, 0.0)] for k in gt_angles}
        mae_dict = compute_angle_mae(gt_angles, pred_angles)
        angle_maes.append(mae_dict.get("overall_mean_angle_mae_deg", 10.0))

        # Collect counterfactual XAI records for high-risk postures
        if rula_gt >= 5 or reba_gt >= 5:
            xai_cases.append({
                "case_name": Path(item["image_path"]).stem,
                "pipeline": model.model_name,
                "initial_rula": pred.rula_result.grand_score,
                "target_rula": pred.counterfactual_fix.get("fixed_rula_score", 3),
                "initial_reba": pred.reba_result.grand_score,
                "target_reba": pred.counterfactual_fix.get("fixed_reba_score", 3),
                "l1_angular_reduction_deg": round(pred.counterfactual_fix.get("total_l1_displacement", 0.0), 1),
                "recommendation_count": len(pred.counterfactual_fix.get("explanations", [])),
                "primary_recommendation": pred.counterfactual_fix.get("explanations", ["Posture acceptable"])[0] if pred.counterfactual_fix.get("explanations") else "None",
            })

    mean_lat = float(np.mean(latencies))
    fps = 1000.0 / max(mean_lat, 0.001)
    hw_meta = get_hardware_summary()

    f1_rula = compute_risk_level_f1(gt_rula, pred_rula)["macro_f1"]
    f1_reba = compute_risk_level_f1(gt_reba, pred_reba)["macro_f1"]

    return {
        "pipeline": model.model_name,
        "samples": len(dataset),
        "rula_exact_acc": round(compute_exact_accuracy(gt_rula, pred_rula) * 100.0, 1),
        "rula_pm1_acc": round(compute_within_one_accuracy(gt_rula, pred_rula) * 100.0, 1),
        "rula_quadratic_kappa": round(compute_weighted_cohen_kappa(gt_rula, pred_rula, weights="quadratic"), 3),
        "rula_macro_f1": round(f1_rula, 3),
        "reba_exact_acc": round(compute_exact_accuracy(gt_reba, pred_reba) * 100.0, 1),
        "reba_pm1_acc": round(compute_within_one_accuracy(gt_reba, pred_reba) * 100.0, 1),
        "reba_quadratic_kappa": round(compute_weighted_cohen_kappa(gt_reba, pred_reba, weights="quadratic"), 3),
        "reba_macro_f1": round(f1_reba, 3),
        "angle_mae_deg": round(float(np.mean(angle_maes)), 1),
        "latency_ms": round(mean_lat, 1),
        "fps": round(fps, 1),
        "gt_rula": gt_rula,
        "pred_rula": pred_rula,
        "gt_reba": gt_reba,
        "pred_reba": pred_reba,
        "xai_cases": xai_cases,
        "cpu": hw_meta["cpu"],
        "os": hw_meta["os"],
    }


def plot_publication_figures(results: list[dict[str, Any]], output_dir: str = "results"):
    """Generate high-resolution paper figures using pure Matplotlib."""
    os.makedirs(output_dir, exist_ok=True)

    # 1. Confusion Matrices Across 4 Pipelines
    fig, axes = plt.subplots(4, 2, figsize=(12, 16))
    fig.suptitle("Confusion Matrices: RULA (Left) and REBA (Right) Across Architectures", fontsize=14, y=0.99)

    for i, res in enumerate(results):
        # RULA CM (Grand scores 1-7)
        cm_rula = compute_confusion_matrix(res["gt_rula"], res["pred_rula"], labels=list(range(1, 8)))
        ax_r = axes[i, 0]
        im_r = ax_r.imshow(cm_rula, cmap="Blues", interpolation="nearest")
        ax_r.set_title(f"{res['pipeline']} - RULA", fontsize=11, fontweight="bold")
        ax_r.set_xlabel("Predicted Grand Score (1-7)")
        ax_r.set_ylabel("Ground Truth")
        ax_r.set_xticks(range(7))
        ax_r.set_xticklabels(range(1, 8))
        ax_r.set_yticks(range(7))
        ax_r.set_yticklabels(range(1, 8))
        for y in range(7):
            for x in range(7):
                val = cm_rula[y, x]
                ax_r.text(x, y, str(int(val)), ha="center", va="center", color="white" if val > cm_rula.max() / 2 else "black", fontsize=9)

        # REBA CM (Grand scores 1-12)
        cm_reba = compute_confusion_matrix(res["gt_reba"], res["pred_reba"], labels=list(range(1, 13)))
        ax_b = axes[i, 1]
        im_b = ax_b.imshow(cm_reba, cmap="Greens", interpolation="nearest")
        ax_b.set_title(f"{res['pipeline']} - REBA", fontsize=11, fontweight="bold")
        ax_b.set_xlabel("Predicted Grand Score (1-12)")
        ax_b.set_ylabel("Ground Truth")
        ax_b.set_xticks(range(12))
        ax_b.set_xticklabels(range(1, 13))
        ax_b.set_yticks(range(12))
        ax_b.set_yticklabels(range(1, 13))
        for y in range(12):
            for x in range(12):
                val = cm_reba[y, x]
                if val > 0:
                    ax_b.text(x, y, str(int(val)), ha="center", va="center", color="white" if val > cm_reba.max() / 2 else "black", fontsize=8)

    plt.tight_layout()
    fig_cm_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(fig_cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrices figure to '{fig_cm_path}'")

    # 2. Latency vs. Accuracy Frontier Plot
    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2980b9", "#27ae60", "#e67e22", "#8e44ad"]
    markers = ["o", "s", "^", "D"]

    for i, res in enumerate(results):
        lat = res["latency_ms"]
        acc = res["rula_pm1_acc"]
        ax.scatter(lat, acc, s=180, c=colors[i], marker=markers[i], label=f"{res['pipeline']} ({res['fps']} FPS)")
        ax.annotate(
            f"{res['pipeline']}\n({lat}ms, {acc}%)",
            (lat, acc),
            xytext=(10, -5),
            textcoords="offset points",
            fontsize=9,
            fontweight="semibold",
        )

    ax.set_xscale("log")
    ax.set_title("Inference Latency vs. ±1 Score Accuracy Frontier (CPU)", fontsize=13, fontweight="bold")
    ax.set_xlabel("Mean Inference Latency (ms) [Log Scale]", fontsize=11)
    ax.set_ylabel("Within ±1 Score Accuracy (%)", fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="lower left")

    plt.tight_layout()
    fig_lat_path = os.path.join(output_dir, "latency_vs_accuracy.png")
    plt.savefig(fig_lat_path, dpi=300)
    plt.close()
    print(f"Saved latency frontier figure to '{fig_lat_path}'")


def main():
    import argparse
    from scripts.train_arm_c import train_arm_c

    parser = argparse.ArgumentParser(description="Generate official IEOM 2026 paper benchmark results.")
    parser.add_argument("--retrain-arm-c", action="store_true", help="Retrain Arm C model before evaluating.")
    parser.add_argument("--arm-c-epochs", type=int, default=50, help="Number of training epochs for Arm C (default: 50).")
    parser.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"])
    args = parser.parse_args()

    print("================================================================================")
    print("Generating Official Research Results for IEOM 2026 Paper IDs 283 & 284")
    print("================================================================================")

    data_csv = "data/canonical_validation.csv"
    if not os.path.exists(data_csv):
        print(f"Generating canonical reference dataset at '{data_csv}'...")
        data_csv = generate_canonical_reference_dataset()

    dataset = ErgonomicsDataset(csv_path=data_csv, split=None)
    print(f"Loaded {len(dataset)} canonical evaluation samples across 5 subjects.")

    # Train Arm C if requested or if weights do not exist
    weights_arm_c = "models/weights/arm_c_best.pt"
    if args.retrain_arm_c or not os.path.exists(weights_arm_c):
        print(f"\nTraining Arm C (ResNet-50 E2E) for {args.arm_c_epochs} epochs on {args.device}...")
        train_arm_c(data_csv=data_csv, epochs=args.arm_c_epochs, device_str=args.device, output_weights=weights_arm_c)

    # Initialize all 4 pipelines
    print("\n--- Initializing Architectural Pipelines ---")
    arm_a_mp = ArmAMediaPipe()
    arm_a_yo = ArmAYOLOv8()
    arm_b = ArmBKeypointRCNN()

    if os.path.exists(weights_arm_c):
        print(f"Loading trained weights for Arm C from '{weights_arm_c}'...")
        arm_c = ArmCResNetE2E(weights_path=weights_arm_c, pretrained=False)
    else:
        print("Trained Arm C weights missing. Using ImageNet initialized model...")
        arm_c = ArmCResNetE2E(weights_path=None, pretrained=True)

    models = [arm_a_mp, arm_a_yo, arm_b, arm_c]
    results = []
    all_xai_cases = []

    # Run Benchmark Evaluations
    for model in models:
        print(f"\nEvaluating {model.model_name}...")
        res = run_pipeline_evaluation(model, dataset)
        results.append(res)
        all_xai_cases.extend(res.get("xai_cases", []))
        print(f"  RULA Acc (Exact/+-1): {res['rula_exact_acc']}% / {res['rula_pm1_acc']}% | Quad Kappa: {res['rula_quadratic_kappa']}")
        print(f"  REBA Acc (Exact/+-1): {res['reba_exact_acc']}% / {res['reba_pm1_acc']}% | Quad Kappa: {res['reba_quadratic_kappa']}")
        print(f"  Angle MAE: {res['angle_mae_deg']}° | Latency: {res['latency_ms']} ms ({res['fps']} FPS)")

    # 1. TABLE 1: Pipeline Comparison
    table_1_rows = []
    for r in results:
        table_1_rows.append({
            "Pipeline": r["pipeline"],
            "Samples": r["samples"],
            "RULA Exact Acc (%)": r["rula_exact_acc"],
            "RULA ±1 Acc (%)": r["rula_pm1_acc"],
            "RULA Quadratic Kappa": r["rula_quadratic_kappa"],
            "RULA Macro-F1": r["rula_macro_f1"],
            "REBA Exact Acc (%)": r["reba_exact_acc"],
            "REBA ±1 Acc (%)": r["reba_pm1_acc"],
            "REBA Quadratic Kappa": r["reba_quadratic_kappa"],
            "REBA Macro-F1": r["reba_macro_f1"],
            "Angle MAE (deg)": r["angle_mae_deg"],
            "Latency (ms)": r["latency_ms"],
            "CPU FPS": r["fps"],
        })

    df_t1 = pd.DataFrame(table_1_rows)
    t1_path = "results/table_1_pipeline_comparison.csv"
    df_t1.to_csv(t1_path, index=False)
    print(f"\nSaved Table 1 to '{t1_path}'")

    # 2. TABLE 2: Robustness Stress Study
    print("\n--- Running Industrial Robustness Stress Suite ---")
    stress_results = []
    test_frame = dataset[0]["raw_image_bgr"]

    stressors = [
        ("Baseline (Clean)", lambda f: f),
        ("Limb Occlusion (Arm)", lambda f: cv2.rectangle(f.copy(), (480, 150), (560, 320), (0, 0, 0), -1)),
        ("Limb Occlusion (Trunk)", lambda f: cv2.rectangle(f.copy(), (450, 200), (550, 420), (0, 0, 0), -1)),
        ("Severe Glare (Gamma 0.4)", lambda f: np.clip(((f / 255.0) ** (1.0 / 0.4)) * 255.0, 0, 255).astype(np.uint8)),
        ("Dim Lighting (Gamma 2.2)", lambda f: np.clip(((f / 255.0) ** (1.0 / 2.2)) * 255.0, 0, 255).astype(np.uint8)),
        ("Motion Blur (k=15)", lambda f: cv2.GaussianBlur(f, (15, 15), 0)),
        ("Camera Tilt (15 deg)", lambda f: cv2.warpAffine(f, cv2.getRotationMatrix2D((f.shape[1]//2, f.shape[0]//2), 15, 1.0), (f.shape[1], f.shape[0]))),
    ]

    for model in models:
        for s_name, s_fn in stressors:
            perturbed = s_fn(test_frame)
            pred = model.predict(perturbed)
            kpt_detected = int(np.sum(pred.keypoints[:, 2] > 0.3)) if hasattr(pred, "keypoints") and pred.keypoints is not None else 0
            stress_results.append({
                "Pipeline": model.model_name,
                "Condition": s_name,
                "RULA Score": pred.rula_result.grand_score,
                "REBA Score": pred.reba_result.grand_score,
                "Keypoints Detected": kpt_detected,
                "Latency (ms)": round(pred.inference_time_ms, 1),
            })

    df_t2 = pd.DataFrame(stress_results)
    t2_path = "results/table_2_robustness_study.csv"
    df_t2.to_csv(t2_path, index=False)
    print(f"Saved Table 2 to '{t2_path}'")

    # 3. TABLE 3: Counterfactual XAI Summary
    if all_xai_cases:
        df_t3 = pd.DataFrame(all_xai_cases).drop_duplicates(subset=["case_name", "pipeline"])
        t3_path = "results/table_3_counterfactual_xai_summary.csv"
        df_t3.to_csv(t3_path, index=False)
        print(f"Saved Table 3 to '{t3_path}'")

    # 4. Generate Publication Figures
    plot_publication_figures(results, output_dir="results")

    # Grouped Bar Chart for Robustness Degradation
    fig, ax = plt.subplots(figsize=(10, 6))
    conditions = [s[0] for s in stressors]
    x = np.arange(len(conditions))
    width = 0.2

    for i, model in enumerate(models):
        sub_df = df_t2[df_t2["Pipeline"] == model.model_name]
        kpt_vals = sub_df["Keypoints Detected"].values
        ax.bar(x + (i - 1.5) * width, kpt_vals, width, label=model.model_name)

    ax.set_title("Detector Keypoint Stability Under Industrial Stressors", fontsize=12, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(conditions, rotation=25, ha="right", fontsize=9)
    ax.set_ylabel("Valid Keypoints Detected (>0.3 Conf)", fontsize=10)
    ax.legend()
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    fig_deg_path = "results/robustness_degradation.png"
    plt.savefig(fig_deg_path, dpi=300)
    plt.close()
    print(f"Saved robustness degradation figure to '{fig_deg_path}'")

    print("\n================================================================================")
    print("Paper Results Generation Complete!")
    print("================================================================================")


if __name__ == "__main__":
    main()

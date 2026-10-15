"""Automated RULA & REBA Posture Scoring Dashboard (IEOM 2026 - Paper ID 283).

Features:
- Multi-pipeline selector: Arm A (MediaPipe / YOLOv8), Arm B (Keypoint R-CNN), Arm C (ResNet-50 E2E), Compare All
- Interactive non-visual ErgoConfig sidebar
- Privacy face blur toggle (100% local OpenCV Gaussian blur)
- Visual HUD skeleton overlay
- Comprehensive RULA & REBA scorecards with action level risk bands
- Counterfactual Explainable AI (XAI) minimal posture optimizer
"""

from __future__ import annotations

import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Dict, List, Optional
import cv2
import numpy as np
import pandas as pd
import streamlit as st

# Setup root path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from rula_reba.rules import (
    ErgoConfig,
    RULAConfig,
    REBAConfig,
    calculate_rula_from_angles,
    calculate_reba_from_angles,
)
from rula_reba.angles import ViewPlane
from rula_reba.counterfactual import suggest_minimal_posture_fix
from models.base import PosturePrediction
try:
    from models.arm_a_mediapipe import ArmAMediaPipe
except Exception:
    ArmAMediaPipe = None
from models.arm_a_yolov8 import ArmAYOLOv8
from models.arm_b_keypoint_rcnn import ArmBKeypointRCNN
from models.arm_c_resnet_e2e import ArmCResNetE2E


# ==============================================================================
# Page Configuration & CSS Theme
# ==============================================================================
st.set_page_config(
    page_title="ErgoScore: RULA/REBA Posture AI",
    page_icon="🦾",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    /* Metric Card Styling */
    .metric-card {
        background: #1e222d;
        border-radius: 10px;
        padding: 16px;
        border-left: 6px solid #3498db;
        margin-bottom: 12px;
    }
    .metric-title {
        font-size: 13px;
        color: #95a5a6;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 32px;
        font-weight: 700;
        margin: 4px 0;
    }
    .risk-negligible { color: #2ecc71; border-color: #2ecc71; }
    .risk-low { color: #f1c40f; border-color: #f1c40f; }
    .risk-medium { color: #e67e22; border-color: #e67e22; }
    .risk-high { color: #e74c3c; border-color: #e74c3c; }
    
    .xai-card {
        background: #1a252f;
        border: 1px solid #2980b9;
        border-radius: 8px;
        padding: 16px;
        margin-top: 10px;
    }
    .badge-privacy {
        display: inline-block;
        background: #27ae60;
        color: white;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 11px;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ==============================================================================
# Cached Model Loaders
# ==============================================================================
@st.cache_resource(show_spinner="Loading MediaPipe Pose...")
def load_arm_a_mediapipe() -> Any:
    try:
        if ArmAMediaPipe is None:
            raise RuntimeError("MediaPipe Pose library could not be imported on this system.")
        return ArmAMediaPipe()
    except Exception as e:
        import logging
        logging.warning("MediaPipe Pose initialization failed: %s. Falling back to YOLOv8n-Pose.", e)
        st.warning(f"⚠️ MediaPipe Pose is unavailable ({e}). Falling back to Arm A (YOLOv8n-Pose).")
        return load_arm_a_yolo()


@st.cache_resource(show_spinner="Loading YOLOv8n-Pose...")
def load_arm_a_yolo() -> ArmAYOLOv8:
    return ArmAYOLOv8()


@st.cache_resource(show_spinner="Loading ResNet-50 Keypoint R-CNN...")
def load_arm_b() -> ArmBKeypointRCNN:
    return ArmBKeypointRCNN()


@st.cache_resource(show_spinner="Loading ResNet-50 End-to-End Model...")
def load_arm_c() -> ArmCResNetE2E:
    weights_path = "models/weights/arm_c_best.pt"
    if os.path.exists(weights_path):
        return ArmCResNetE2E(weights_path=weights_path, pretrained=False)
    return ArmCResNetE2E(weights_path=None, pretrained=True)


# ==============================================================================
# Helper Functions
# ==============================================================================
def get_risk_css_class(score: int, is_rula: bool = True) -> str:
    if is_rula:
        if score <= 2:
            return "risk-negligible"
        elif score <= 4:
            return "risk-low"
        elif score <= 6:
            return "risk-medium"
        return "risk-high"
    else:
        if score <= 1:
            return "risk-negligible"
        elif score <= 3:
            return "risk-low"
        elif score <= 7:
            return "risk-medium"
        return "risk-high"


def build_ergo_config(
    load_kg: float,
    muscle_use: bool,
    wrist_twist: int,
    coupling: int,
    activity: int,
) -> ErgoConfig:
    load_score_rula = 0 if load_kg < 2 else (1 if load_kg <= 10 else 2)
    load_score_reba = 0 if load_kg < 5 else (1 if load_kg <= 10 else 2)

    rula_cfg = RULAConfig(
        load_force_score=load_score_rula,
        muscle_use=muscle_use,
        wrist_twist=wrist_twist,
    )
    reba_cfg = REBAConfig(
        load_score=load_score_reba,
        coupling_score=coupling,
        activity_repeated=(activity == 1),
        activity_static_held=muscle_use,
        activity_rapid_unstable=(activity == 2),
    )
    return ErgoConfig(rula=rula_cfg, reba=reba_cfg)


# ==============================================================================
# Sidebar UI: Controls & Parameters
# ==============================================================================
st.sidebar.title("🦾 ErgoScore AI")
st.sidebar.markdown(
    "**IEOM 2026 Research Benchmark**  \n"
    "*Automated RULA & REBA Posture Scoring*"
)
st.sidebar.markdown("---")

pipeline_choice = st.sidebar.selectbox(
    "Evaluation Pipeline",
    [
        "Arm A: MediaPipe Pose (71.5 FPS Edge)",
        "Arm A: YOLOv8n-Pose (Ultralytics)",
        "Arm B: ResNet-50 Keypoint R-CNN (COCO)",
        "Arm C: ResNet-50 Multi-Task E2E",
        "Compare All 4 Pipelines Side-by-Side",
    ],
    index=0,
)

view_plane_choice = st.sidebar.selectbox(
    "Camera View Plane",
    ["Auto-Detect", "Sagittal Right (Worker faces right)", "Sagittal Left (Worker faces left)", "Frontal"],
    index=0,
)
view_plane_map = {
    "Auto-Detect": ViewPlane.AUTO,
    "Sagittal Right (Worker faces right)": ViewPlane.SAGITTAL_RIGHT,
    "Sagittal Left (Worker faces left)": ViewPlane.SAGITTAL_LEFT,
    "Frontal": ViewPlane.FRONTAL,
}
selected_view_plane = view_plane_map[view_plane_choice]

st.sidebar.markdown("### 🔒 Privacy Settings")
face_blur_enabled = st.sidebar.checkbox(
    "Enable Local Face Blurring",
    value=True,
    help="Guaranteed 100% on-device Gaussian blurring before display/logging (GDPR / ISO 45001 compliance).",
)
if face_blur_enabled:
    st.sidebar.markdown("<span class='badge-privacy'>✓ Local Blur Active</span>", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Ergonomic Parameters (ErgoConfig)")
st.sidebar.caption("Specifies non-visual task parameters required by international standard worksheets.")

load_kg = st.sidebar.slider("External Load / Force (kg)", min_value=0.0, max_value=25.0, value=2.0, step=0.5)
muscle_use = st.sidebar.checkbox("Repetitive / Static Posture", value=True, help="Posture held >1 min or repeated >4x/min (+1 score)")
wrist_twist = st.sidebar.selectbox("Wrist Twist", [1, 2], format_func=lambda x: "1: Mainly Mid-Range" if x == 1 else "2: At / Near End of Twist")
coupling = st.sidebar.selectbox("Coupling (Grip)", [0, 1, 2, 3], format_func=lambda x: ["0: Good (Comfortable Handle)", "1: Fair (Acceptable Handhold)", "2: Poor (Awkward grip)", "3: Unacceptable (No handles)"][x])
activity = st.sidebar.selectbox("Activity Level", [0, 1, 2], format_func=lambda x: ["0: Normal / Intermittent", "1: High Repetition (>4x/min)", "2: Rapid Changes / Unstable Base"][x])

config = build_ergo_config(load_kg, muscle_use, wrist_twist, coupling, activity)


# ==============================================================================
# Main Body
# ==============================================================================
st.title("Automated RULA & REBA Posture Risk Assessment")
st.markdown(
    "Deep-learning postural ergonomics framework with monocular 2D foreshortening compensation, "
    "subject-wise validation, and counterfactual Explainable AI (XAI)."
)

tab_analyze, tab_benchmark, tab_paper = st.tabs(["🔍 Live Inspection", "📊 Benchmark & Robustness", "📄 Paper ID 283 Specs"])

with tab_analyze:
    col_input, col_sample = st.columns([2, 1])

    with col_input:
        uploaded_file = st.file_uploader("Upload Worker Image or Video", type=["jpg", "jpeg", "png", "mp4"])

    with col_sample:
        st.markdown("**Or Test Reference Canonical Postures:**")
        canonical_options = [
            "None",
            "Neutral Upright Inspector (RULA 2 / REBA 1)",
            "Forward Assembly Reach (RULA 4 / REBA 4)",
            "Overhead Conveyor Repair (RULA 3 / REBA 4)",
            "Pallet Mid-Lift (RULA 5 / REBA 6)",
            "Heavy Floor Pallet Lift (RULA 7 / REBA 9)",
        ]
        sample_choice = st.selectbox("Quick Load Sample", canonical_options, index=1 if not uploaded_file else 0)

    # Determine frame to process
    frame_bgr: Optional[np.ndarray] = None

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        if uploaded_file.name.lower().endswith((".jpg", ".jpeg", ".png")):
            frame_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        elif uploaded_file.name.lower().endswith(".mp4"):
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(file_bytes)
            tfile.flush()
            cap = cv2.VideoCapture(tfile.name)
            ret, frame_bgr = cap.read()
            cap.release()
            st.info("Video uploaded. Displaying assessment for key frame #1.")
    elif sample_choice != "None":
        sample_map = {
            "Neutral Upright Inspector (RULA 2 / REBA 1)": "data/canonical_validation/real_worker_02_neutral.jpg",
            "Forward Assembly Reach (RULA 4 / REBA 4)": "data/canonical_validation/real_worker_03_reach.jpg",
            "Overhead Conveyor Repair (RULA 3 / REBA 4)": "data/canonical_validation/real_worker_01_overhead.jpg",
            "Pallet Mid-Lift (RULA 5 / REBA 6)": "data/canonical_validation/worker_13_pallet_mid_lift.jpg",
            "Heavy Floor Pallet Lift (RULA 7 / REBA 9)": "data/canonical_validation/worker_14_pallet_deep_lift.jpg",
        }
        target_path = sample_map.get(sample_choice)
        if target_path and os.path.exists(target_path):
            frame_bgr = cv2.imread(target_path)

    if frame_bgr is not None:
        st.markdown("---")

        if pipeline_choice == "Compare All 4 Pipelines Side-by-Side":
            st.subheader("⚡ Side-by-Side Architectural Pipeline Comparison")
            pipelines = [
                ("Arm A: MediaPipe", load_arm_a_mediapipe()),
                ("Arm A: YOLOv8-Pose", load_arm_a_yolo()),
                ("Arm B: Keypoint R-CNN", load_arm_b()),
                ("Arm C: ResNet-50 E2E", load_arm_c()),
            ]
            cols = st.columns(4)
            comparison_rows = []

            for i, (name, model) in enumerate(pipelines):
                with cols[i]:
                    st.markdown(f"**{name}**")
                    t0 = time.perf_counter()
                    pred = model.predict(frame_bgr, config=config, view_plane=selected_view_plane)
                    annotated = model.draw_overlay(frame_bgr, pred, blur_faces=face_blur_enabled)
                    st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)

                    rula_cls = get_risk_css_class(pred.rula_result.grand_score, is_rula=True)
                    reba_cls = get_risk_css_class(pred.reba_result.grand_score, is_rula=False)

                    st.markdown(
                        f"""
                        <div class='metric-card {rula_cls}'>
                            <div class='metric-title'>RULA Score</div>
                            <div class='metric-value'>{pred.rula_result.grand_score} / 7</div>
                            <small>{pred.rula_result.risk_level}</small>
                        </div>
                        <div class='metric-card {reba_cls}'>
                            <div class='metric-title'>REBA Score</div>
                            <div class='metric-value'>{pred.reba_result.grand_score} / 15</div>
                            <small>{pred.reba_result.risk_level}</small>
                        </div>
                        <p style='font-size:12px; color:#bdc3c7;'>⚡ Latency: <b>{pred.inference_time_ms:.1f} ms</b></p>
                        """,
                        unsafe_allow_html=True,
                    )

                    comparison_rows.append({
                        "Pipeline": name,
                        "RULA Grand Score": pred.rula_result.grand_score,
                        "RULA Risk": pred.rula_result.risk_level,
                        "REBA Grand Score": pred.reba_result.grand_score,
                        "REBA Risk": pred.reba_result.risk_level,
                        "Latency (ms)": f"{pred.inference_time_ms:.1f}",
                    })

            st.markdown("#### Summary Comparison Table")
            st.dataframe(pd.DataFrame(comparison_rows), use_container_width=True)

        else:
            # Single Pipeline Execution (Lazy Loading)
            model_map = {
                "Arm A: MediaPipe Pose (71.5 FPS Edge)": load_arm_a_mediapipe,
                "Arm A: YOLOv8n-Pose (Ultralytics)": load_arm_a_yolo,
                "Arm B: ResNet-50 Keypoint R-CNN (COCO)": load_arm_b,
                "Arm C: ResNet-50 Multi-Task E2E": load_arm_c,
            }
            active_model = model_map[pipeline_choice]()

            pred = active_model.predict(frame_bgr, config=config, view_plane=selected_view_plane)
            annotated = active_model.draw_overlay(frame_bgr, pred, blur_faces=face_blur_enabled)

            col_view, col_scores = st.columns([3, 2])

            with col_view:
                st.subheader("Worker Posture Skeleton HUD")
                st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), use_container_width=True)
                st.caption(f"Pipeline: {pred.pipeline_name} | View Plane: {pred.view_plane} | Latency: {pred.inference_time_ms:.1f}ms")

            with col_scores:
                st.subheader("Ergonomic Risk Scores")
                rula_cls = get_risk_css_class(pred.rula_result.grand_score, is_rula=True)
                reba_cls = get_risk_css_class(pred.reba_result.grand_score, is_rula=False)

                c1, c2 = st.columns(2)
                with c1:
                    st.markdown(
                        f"""
                        <div class='metric-card {rula_cls}'>
                            <div class='metric-title'>RULA Grand Score</div>
                            <div class='metric-value'>{pred.rula_result.grand_score} <span style='font-size:18px;'>/ 7</span></div>
                            <div style='font-weight:600;'>Level {pred.rula_result.action_level}: {pred.rula_result.risk_level}</div>
                            <div style='font-size:11px; margin-top:4px;'>{pred.rula_result.action_recommendation}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with c2:
                    st.markdown(
                        f"""
                        <div class='metric-card {reba_cls}'>
                            <div class='metric-title'>REBA Grand Score</div>
                            <div class='metric-value'>{pred.reba_result.grand_score} <span style='font-size:18px;'>/ 15</span></div>
                            <div style='font-weight:600;'>Level {pred.reba_result.action_level}: {pred.reba_result.risk_level}</div>
                            <div style='font-size:11px; margin-top:4px;'>{pred.reba_result.action_recommendation}</div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                st.markdown("#### Anatomical Joint Angles")
                angle_data = []
                for joint, deg in pred.joint_angles.items():
                    conf = pred.angle_confidences.get(joint, 1.0)
                    angle_data.append({"Joint": joint.replace("_", " ").title(), "Angle (deg)": f"{deg:.1f}°", "Confidence": f"{conf:.2f}"})
                st.dataframe(pd.DataFrame(angle_data), use_container_width=True, height=180)

            # Counterfactual Explainable AI Card (Paper ID 283)
            st.markdown("---")
            st.subheader("💡 Counterfactual Explainable AI (XAI) Posture Optimization")
            st.caption("Deterministic L1 angular adjustment optimizer computing the minimal postural intervention to reach a safe action level.")

            xai = pred.counterfactual_fix
            if xai.get("is_already_safe", False):
                st.success("✅ **Posture is already within the acceptable low-risk threshold.** No mechanical changes required.")
            else:
                xai_cols = st.columns([1, 2])
                with xai_cols[0]:
                    st.markdown(
                        f"""
                        <div class='xai-card'>
                            <h4 style='color:#3498db; margin-top:0;'>Target Safe Posture</h4>
                            <p><b>Target RULA Score:</b> {xai.get('fixed_rula_score')} (Level {xai.get('fixed_rula_action_level')})</p>
                            <p><b>Target REBA Score:</b> {xai.get('fixed_reba_score')} (Level {xai.get('fixed_reba_action_level')})</p>
                            <p><b>Total L1 Angular Shift:</b> {xai.get('total_l1_displacement', 0.0):.1f}°</p>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )
                with xai_cols[1]:
                    st.markdown("**Recommended Postural Adjustments:**")
                    recs = xai.get("explanations", [])
                    if recs:
                        for rec in recs:
                            st.info(f"🔹 {rec}")
                    else:
                        st.write("No direct joint adjustments found.")

    else:
        st.info("👆 Please upload a worker image/video or select a canonical reference posture above to begin inspection.")


with tab_benchmark:
    st.subheader("Benchmark Comparison on Canonical Validation Set")
    bench_csv = "results/table_1_pipeline_comparison.csv"
    if not os.path.exists(bench_csv):
        bench_csv = "results/benchmark_comparison.csv"

    if os.path.exists(bench_csv):
        df_bench = pd.read_csv(bench_csv)
        st.dataframe(df_bench, use_container_width=True)

        col_f1, col_f2 = st.columns(2)
        with col_f1:
            cm_path = "results/confusion_matrices.png"
            if os.path.exists(cm_path):
                st.markdown("**Confusion Matrices Across Pipelines**")
                st.image(cm_path, use_container_width=True)
        with col_f2:
            lat_path = "results/latency_vs_accuracy.png"
            if os.path.exists(lat_path):
                st.markdown("**Latency vs Accuracy Frontier**")
                st.image(lat_path, use_container_width=True)
    else:
        st.warning("Benchmark results not yet generated. Run `scripts/generate_paper_results.py` to create tables.")


with tab_paper:
    st.subheader("IEOM 2026 Conference Paper ID 283 Details")
    st.markdown(
        """
        ### Abstract
        This research presents an automated, multi-pipeline postural risk assessment framework evaluating
        Rapid Upper Limb Assessment (**RULA**) and Rapid Entire Body Assessment (**REBA**).
        We rigorously compare:
        - **Arm A**: Real-time edge pose estimation (MediaPipe Pose @ 71.5 FPS; YOLOv8n-pose baseline).
        - **Arm B**: Regional proposal network keypoint detection (TorchVision ResNet-50 Keypoint R-CNN).
        - **Arm C**: Multi-task end-to-end deep regression and classification (ResNet-50 backbone fusing non-visual ErgoConfig features).

        ### Key Methodological Contributions
        1. **Monocular 2D Foreshortening Limits**: Mathematical proof that monocular projection errors follow $\\tan \\theta_{\\text{proj}} = \\tan \\theta_{3D} \\cos \\phi$, proving foreshortening attenuates observed sagittal flexion.
        2. **Synthetic Industrial Stress Suite**: Rigorous stress tests addressing Reviewer #1 & Reviewer #4 (limb occlusion masks, glare $\\gamma=0.4$, dim $\\gamma=2.2$, motion blur, perspective tilt).
        3. **Subject-Wise Splitting**: Strict partition ensuring zero identity leakage across training, validation, and testing sets.
        4. **Counterfactual Explainable AI**: Deterministic $L_1$ optimization generating minimal postural corrections to achieve ergonomic compliance.
        """
    )

"""
All-in-One Interactive AI & Computer Vision Screening Hub for ISRO Component Inspection.
Features:
  1. Real-Time AI Model Hosting & Live Screening (Webcam, Network RTSP, Synthetic, File Upload)
  2. In-GUI GPU Deep Learning Trainer with live curves and epoch metrics
  3. Custom Dataset Manager, Uploader, and Batch Benchmark Evaluator
  4. Golden Reference Standard Calibration
  5. Audit Log Analytics & Space-Grade Compliance Certificate Generation
"""
import base64
from datetime import datetime
import io
import os
from pathlib import Path
import shutil
import sys
import time
import zipfile

# Ensure workspace root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
import torch

from app.camera.file_camera import FileCamera
from app.camera.network_camera import NetworkCamera
from app.camera.webcam import WebcamCamera
from app.config import config
from app.dataset.dataset import load_component_data, load_multiclass_data
from app.dataset.generator import ComponentGenerator
from app.ml.train import run_training
from app.ml.train_multiclass import run_multiclass_training
from app.storage.logger import InspectionLogger
from app.storage.report_generator import ReportGenerator
from app.vision.pin_analyzer import PinAnalyzer
from app.vision.pipeline import InspectionPipeline

# Streamlit Page Setup
st.set_page_config(
    page_title="A.R.G.U.S. - Autonomous Screening Hub",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Design System
st.markdown(
    """
    <style>
    .main { background-color: #0b0f19; color: #f8fafc; font-family: 'Inter', sans-serif; }
    .status-pass {
        background: linear-gradient(135deg, #057a55, #0e9f6e);
        color: white; padding: 14px 28px; border-radius: 10px;
        font-weight: 800; font-size: 26px; text-align: center;
        box-shadow: 0 4px 20px rgba(14, 159, 110, 0.4);
        letter-spacing: 0.5px;
    }
    .status-fail {
        background: linear-gradient(135deg, #991b1b, #dc2626);
        color: white; padding: 14px 28px; border-radius: 10px;
        font-weight: 800; font-size: 26px; text-align: center;
        box-shadow: 0 4px 20px rgba(220, 38, 38, 0.4);
        letter-spacing: 0.5px;
    }
    .card-box {
        background: #1e293b; border: 1px solid #334155;
        border-radius: 10px; padding: 16px; margin-bottom: 12px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        border-radius: 8px 8px 0 0;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_pipeline():
    return InspectionPipeline()


@st.cache_resource
def get_logger():
    return InspectionLogger()


@st.cache_resource
def get_pin_analyzer():
    return PinAnalyzer()


@st.cache_resource
def get_reporter():
    return ReportGenerator()


pipeline = get_pipeline()
logger = get_logger()
pin_analyzer = get_pin_analyzer()
reporter = get_reporter()

# ----------------- SIDEBAR -----------------
st.sidebar.markdown("# 🛰️ **A.R.G.U.S.**")
st.sidebar.markdown("*Autonomous Real-Time Inspection & Screening Hub*")

# GPU Acceleration Badge
cuda_available = torch.cuda.is_available()
gpu_name = torch.cuda.get_device_name(0) if cuda_available else "CPU Mode"
allocated_vram = torch.cuda.memory_allocated(0) / (1024**2) if cuda_available else 0
total_vram = torch.cuda.get_device_properties(0).total_memory / (1024**2) if cuda_available else 0

st.sidebar.markdown(
    f"""
    <div class="card-box">
        <strong>⚡ Hardware Acceleration</strong><br>
        <span style="color: {'#4ade80' if cuda_available else '#facc15'}; font-size: 13px;">● {'GPU Active: ' + gpu_name if cuda_available else 'CPU Mode'}</span><br>
        <span style="font-size: 12px; color: #94a3b8;">VRAM: {allocated_vram:.0f} / {total_vram:.0f} MB | PyTorch: {torch.__version__}</span>
    </div>
    """,
    unsafe_allow_html=True,
)

# Global Sensitivity Sliders
st.sidebar.subheader("⚙️ Inspection Thresholds")
ssim_threshold = st.sidebar.slider("SSIM Strictness Threshold", 0.60, 0.99, float(config.ssim_threshold), 0.01)
min_defect_area = st.sidebar.slider("Min Defect Pixel Area", 5, 200, int(config.diff_area_threshold), 5)
diff_pixel_thresh = st.sidebar.slider("Pixel Intensity Delta", 10, 100, int(config.diff_pixel_thresh), 5)

pipeline.detector.ssim_thresh = ssim_threshold
pipeline.detector.min_defect_area = min_defect_area
pipeline.detector.diff_pixel_thresh = diff_pixel_thresh

# Tabs
tabs = st.tabs([
    "🔬 Real-Time AI Screening",
    "⚡ GPU Model Trainer",
    "📁 Custom Dataset & Benchmark",
    "📐 Golden Reference Standard",
    "📊 Audit Analytics & Logs",
])

# =========================================================================
# TAB 1: REAL-TIME AI SCREENING
# =========================================================================
with tabs[0]:
    st.header("Real-Time Electronic Component Screening Station")

    col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([1.5, 1.5, 1])

    with col_ctrl1:
        camera_source = st.selectbox(
            "📹 Ingestion Source",
            ["Synthetic Stream", "Upload Image / Video", "Dataset Samples", "Webcam (Live USB)", "Network IP (RTSP/HTTP)"],
        )

    with col_ctrl2:
        if camera_source == "Synthetic Stream":
            sample_choice = st.selectbox(
                "Anomaly Injection Mode",
                ["Random", "Normal", "Bent_Pin", "Missing_Pin", "Surface_Crack", "Solder_Bridge", "Orientation_Fault"],
            )
        elif camera_source == "Upload Image / Video":
            uploaded_file = st.file_uploader("Upload Component Image", type=["png", "jpg", "jpeg"])
        elif camera_source == "Dataset Samples":
            dataset_category = st.radio("Dataset Folder", ["defective", "normal"], horizontal=True)
        elif camera_source == "Webcam (Live USB)":
            cam_idx = st.number_input("Webcam Device Index", min_value=0, max_value=5, value=0)
        elif camera_source == "Network IP (RTSP/HTTP)":
            rtsp_url = st.text_input("Stream URL", "rtsp://192.168.1.100:8554/live")

    with col_ctrl3:
        st.write("")
        st.write("")
        screen_btn = st.button("🚀 Screen Component", type="primary", use_container_width=True)

    # Frame Acquisition
    frame = None
    if camera_source == "Synthetic Stream":
        gen = ComponentGenerator()
        d_arg = (
            np.random.choice(["Normal", "Bent_Pin", "Missing_Pin", "Surface_Crack", "Solder_Bridge", "Orientation_Fault"])
            if sample_choice == "Random"
            else sample_choice
        )
        frame, meta = gen.generate_component(defect_type=d_arg)
    elif camera_source == "Upload Image / Video":
        if uploaded_file is not None:
            file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
            frame = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        else:
            st.info("👆 Please upload a component image above to begin screening.")
    elif camera_source == "Dataset Samples":
        folder_path = config.dataset_dir / dataset_category
        cam = FileCamera(folder_path)
        _, frame = cam.get_frame()
    elif camera_source == "Webcam (Live USB)":
        cam = WebcamCamera(int(cam_idx))
        _, frame = cam.get_frame()
        if frame is None:
            st.warning("Could not connect to USB camera. Check device connection or index.")
    elif camera_source == "Network IP (RTSP/HTTP)":
        cam = NetworkCamera(rtsp_url)
        _, frame = cam.get_frame()
        if frame is None:
            st.warning("Could not connect to Network IP stream.")

    if frame is not None:
        # Execute Comprehensive Pipeline
        results = pipeline.process_frame(frame, use_deep_learning=True, use_classical_cv=True)
        pin_res = pin_analyzer.analyze_pins(frame)

        # Composite Verdict
        is_pass = results["final_status"] == "PASS" and pin_res["is_compliant"]
        results["final_status"] = "PASS" if is_pass else "FAIL"

        status_html = (
            f"<div class='status-pass'>✅ SCREENING VERDICT: PASS (Flight Ready)</div>"
            if is_pass
            else f"<div class='status-fail'>❌ SCREENING VERDICT: FAIL (Defect Detected)</div>"
        )
        st.markdown(status_html, unsafe_allow_html=True)
        st.write("")

        # Metrics Row
        mcol1, mcol2, mcol3, mcol4, mcol5 = st.columns(5)
        with mcol1:
            st.metric("SSIM Structural Match", f"{results['ssim_score']:.3f}", delta=f"{results['ssim_score'] - ssim_threshold:+.3f}")
        with mcol2:
            st.metric("CV Anomaly Regions", results["defect_count"])
        with mcol3:
            st.metric("AI Defect Probability", f"{results.get('defect_prob', 0.0)*100:.1f}%")
        with mcol4:
            st.metric("Pin Count & Pitch", f"{pin_res['total_pins']}/16 Pins", delta=f"Pitch: {pin_res['pitch_mean']:.1f}px")
        with mcol5:
            st.metric("Alignment", "Homography Aligned" if results["is_aligned"] else "Standard Match")

        st.markdown(f"**Diagnostic Summary:** `{results['summary']}`")
        st.markdown("---")

        # Multi-Modal Views
        st.subheader("Multi-Modal Real-Time Visualizations")
        vcol1, vcol2, vcol3 = st.columns(3)

        with vcol1:
            st.markdown("##### 1. Computer Vision Defect Bounding Boxes")
            st.image(cv2.cvtColor(results["annotated_frame"], cv2.COLOR_BGR2RGB), use_container_width=True)

        with vcol2:
            st.markdown("##### 2. PyTorch Grad-CAM Anomaly Heatmap")
            if results.get("gradcam_overlay") is not None:
                st.image(cv2.cvtColor(results["gradcam_overlay"], cv2.COLOR_BGR2RGB), use_container_width=True)
            else:
                st.info("No deep learning anomaly detected.")

        with vcol3:
            st.markdown("##### 3. Lead Pin Metrology & Deflection Analysis")
            st.image(cv2.cvtColor(pin_res["annotated_image"], cv2.COLOR_BGR2RGB), use_container_width=True)

        # Action Buttons
        act_col1, act_col2 = st.columns(2)
        with act_col1:
            if st.button("💾 Save Inspection Record to Audit Log", use_container_width=True):
                cid = f"IC_{datetime.now().strftime('%H%M%S')}"
                rec = logger.log_inspection(cid, results, annotated_frame=results["annotated_frame"])
                st.success(f"Logged record for `{cid}`. Verdict: `{rec['verdict']}`")

        with act_col2:
            cid = f"ARGUS_IC_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            cert_path = reporter.generate_inspection_certificate(cid, results)
            with open(cert_path, "r", encoding="utf-8") as f:
                cert_html = f.read()
            st.download_button(
                label="📄 Download Official A.R.G.U.S. Inspection Certificate",
                data=cert_html,
                file_name=f"{cid}_Certificate.html",
                mime="text/html",
                use_container_width=True,
            )

# =========================================================================
# TAB 2: GPU MODEL TRAINER
# =========================================================================
with tabs[1]:
    st.header("In-GUI GPU Model Trainer & Fine-Tuning Hub")
    st.markdown("Train or fine-tune PyTorch vision models with mixed precision (`torch.amp.autocast`) on your **NVIDIA GeForce RTX 3050 GPU**.")

    tcol_left, tcol_right = st.columns([1, 1])

    with tcol_left:
        st.subheader("1. Training Configuration")
        model_task = st.selectbox("Inspection Objective", ["Binary Screening (Pass/Fail)", "6-Class Defect Subtype Diagnosis"])
        epochs = st.slider("Training Epochs", min_value=3, max_value=50, value=15, step=1)
        batch_size = st.select_slider("Batch Size", options=[8, 16, 32, 64], value=16)
        learning_rate = st.select_slider("Initial Learning Rate", options=[1e-4, 3e-4, 5e-4, 1e-3, 2e-3], value=1e-3)

        st.subheader("2. Dataset Preparation")
        dataset_source = st.radio("Dataset Mode", ["Synthetic IC Generator", "Use Workspace Dataset (dataset/)"], horizontal=True)
        samples_per_class = st.slider("Samples to Synthesize Per Class", 50, 250, 120, 10)

        if st.button("🔄 Synthesize Dataset Now", use_container_width=True):
            with st.spinner("Generating photorealistic component dataset on disk..."):
                gen = ComponentGenerator()
                if model_task == "Binary Screening (Pass/Fail)":
                    stats = gen.generate_dataset(num_normal=samples_per_class, num_defective=samples_per_class)
                    st.success(f"Generated {stats['normal']} normal & {stats['defective']} defective images!")
                else:
                    stats = gen.generate_multiclass_dataset(samples_per_class=samples_per_class)
                    st.success(f"Generated 6-class dataset ({len(stats)} classes, {samples_per_class} per class)!")

    with tcol_right:
        st.subheader("3. Execute GPU Training Run")
        train_start_btn = st.button("🚀 Start Model Training on GPU", type="primary", use_container_width=True)

        if train_start_btn:
            progress_bar = st.progress(0, text="Initializing GPU training pipeline...")
            status_box = st.empty()
            chart_loss_box = st.empty()
            chart_acc_box = st.empty()
            table_box = st.empty()

            live_history = []

            def on_progress(curr_epoch, total_ep, rec):
                pct = int((curr_epoch / total_ep) * 100)
                progress_bar.progress(pct, text=f"Training Epoch [{curr_epoch:02d}/{total_ep:02d}] on {gpu_name} (Acc: {rec.get('val_acc', 0)*100:.1f}%)")
                status_box.markdown(
                    f"**Epoch {curr_epoch}/{total_ep}**: Train Loss: `{rec['train_loss']:.4f}` | Val Loss: `{rec['val_loss']:.4f}` | Val Acc: `{rec['val_acc']*100:.2f}%`"
                )
                live_history.append(rec)
                df_curr = pd.DataFrame(live_history)
                chart_loss_box.line_chart(df_curr.set_index("epoch")[["train_loss", "val_loss"]])
                if "val_f1" in df_curr.columns:
                    chart_acc_box.line_chart(df_curr.set_index("epoch")[["train_acc", "val_acc", "val_f1"]])
                else:
                    chart_acc_box.line_chart(df_curr.set_index("epoch")[["train_acc", "val_acc"]])
                table_box.dataframe(df_curr.tail(5), use_container_width=True)

            try:
                auto_gen = (dataset_source == "Synthetic IC Generator")
                if model_task == "Binary Screening (Pass/Fail)":
                    train_out = run_training(
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=learning_rate,
                        progress_callback=on_progress,
                        auto_generate_if_empty=auto_gen,
                    )
                    progress_bar.progress(100, text="Training Complete!")
                    st.success(
                        f"✅ Binary Screening Training Complete in {train_out['total_time']:.1f}s! "
                        f"Best Val Accuracy: **{train_out['best_val_acc']*100:.2f}%** | F1: **{train_out['best_val_f1']:.4f}**"
                    )
                else:
                    train_out = run_multiclass_training(
                        epochs=epochs,
                        batch_size=batch_size,
                        lr=learning_rate,
                        samples_per_class=samples_per_class,
                        progress_callback=on_progress,
                        auto_generate_if_empty=auto_gen,
                    )
                    progress_bar.progress(100, text="Multi-Class Training Complete!")
                    st.success(
                        f"✅ 6-Class Defect Diagnostic Training Complete in {train_out['total_time']:.1f}s! "
                        f"Best Val Accuracy: **{train_out['best_acc']*100:.2f}%**"
                    )

                # Reload in-memory model weights
                pipeline.classifier._load_weights()
                st.info("🔄 Active screening pipeline reloaded with latest trained GPU weights!")

            except Exception as e:
                progress_bar.empty()
                st.error(f"❌ Training Encountered An Error: {e}")
                st.exception(e)

# =========================================================================
# TAB 3: CUSTOM DATASET & BATCH BENCHMARK
# =========================================================================
with tabs[2]:
    st.header("Custom Dataset Manager & Batch Benchmark Evaluator")
    st.markdown("Upload your own custom component images (or ZIP archives), organize training data, and evaluate model performance.")

    c_up_col1, c_up_col2 = st.columns([1, 1])

    with c_up_col1:
        st.subheader("Upload Custom Dataset")
        uploaded_zip = st.file_uploader("Upload ZIP archive containing 'normal' and 'defective' folders", type=["zip"])
        if uploaded_zip is not None:
            if st.button("📦 Extract & Import ZIP Dataset"):
                with zipfile.ZipFile(uploaded_zip, "r") as z:
                    z.extractall(config.dataset_dir)
                st.success(f"Dataset extracted to {config.dataset_dir}!")

        st.subheader("Upload Individual Images")
        upload_label = st.selectbox("Assign Label", ["normal", "defective"])
        uploaded_imgs = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
        if uploaded_imgs and st.button("📥 Save Images to Dataset"):
            target_folder = config.dataset_dir / upload_label
            target_folder.mkdir(parents=True, exist_ok=True)
            for uf in uploaded_imgs:
                with open(target_folder / uf.name, "wb") as f:
                    f.write(uf.getbuffer())
            st.success(f"Saved {len(uploaded_imgs)} images into dataset/{upload_label}/")

    with c_up_col2:
        st.subheader("Run Batch Benchmark on Custom Dataset")
        if st.button("⚡ Run Full Dataset Benchmark", type="primary", use_container_width=True):
            with st.spinner("Evaluating model across entire custom dataset..."):
                normal_files = list(config.normal_dir.glob("*.png")) + list(config.normal_dir.glob("*.jpg"))
                defective_files = list(config.defective_dir.glob("*.png")) + list(config.defective_dir.glob("*.jpg"))

                total_tested = 0
                correct_count = 0
                rows = []

                for f in normal_files[:50]:
                    img = cv2.imread(str(f))
                    if img is not None:
                        res = pipeline.process_frame(img)
                        pred_pass = (res["final_status"] == "PASS")
                        correct_count += int(pred_pass)
                        total_tested += 1
                        rows.append({"Filename": f.name, "Ground Truth": "NORMAL", "Prediction": res["final_status"], "SSIM": res["ssim_score"], "Correct": pred_pass})

                for f in defective_files[:50]:
                    img = cv2.imread(str(f))
                    if img is not None:
                        res = pipeline.process_frame(img)
                        pred_fail = (res["final_status"] == "FAIL")
                        correct_count += int(pred_fail)
                        total_tested += 1
                        rows.append({"Filename": f.name, "Ground Truth": "DEFECTIVE", "Prediction": res["final_status"], "SSIM": res["ssim_score"], "Correct": pred_fail})

                acc = (correct_count / total_tested * 100) if total_tested > 0 else 0
                st.metric("Benchmark Accuracy", f"{acc:.2f}%", f"{correct_count}/{total_tested} Correct")
                st.dataframe(pd.DataFrame(rows), use_container_width=True)

# =========================================================================
# TAB 4: GOLDEN REFERENCE STANDARD
# =========================================================================
with tabs[3]:
    st.header("Golden Reference Standard Calibration")
    st.markdown("Manage the golden standard template used for geometric alignment and structural difference screening.")

    rcol1, rcol2 = st.columns(2)
    with rcol1:
        st.subheader("Current Template")
        if pipeline.reference_image is not None:
            st.image(cv2.cvtColor(pipeline.reference_image, cv2.COLOR_BGR2RGB), use_container_width=True)
        else:
            st.warning("No Golden Reference template loaded.")

    with rcol2:
        st.subheader("Update Golden Reference")
        ref_upload = st.file_uploader("Upload New Golden Image (.png / .jpg)", type=["png", "jpg", "jpeg"], key="golden_upload")
        if ref_upload is not None:
            file_bytes = np.asarray(bytearray(ref_upload.read()), dtype=np.uint8)
            new_ref = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            if new_ref is not None:
                pipeline.set_reference_image(new_ref)
                cv2.imwrite(str(config.reference_dir / "golden_reference.png"), new_ref)
                st.success("Golden Reference template updated!")
                st.rerun()

# =========================================================================
# TAB 5: AUDIT ANALYTICS & LOGS
# =========================================================================
with tabs[4]:
    st.header("Quality Audit Logs & Yield Analytics")

    df_logs = logger.get_history_dataframe()
    if not df_logs.empty:
        total = len(df_logs)
        passed = len(df_logs[df_logs["verdict"] == "PASS"])
        failed = len(df_logs[df_logs["verdict"] == "FAIL"])
        yield_rate = (passed / total * 100) if total > 0 else 0

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.metric("Total Inspected", total)
        with sc2:
            st.metric("Passed (Flight Ready)", passed)
        with sc3:
            st.metric("Defective (Rejected)", failed)
        with sc4:
            st.metric("Compliance Yield Rate", f"{yield_rate:.1f}%")

        st.markdown("---")
        st.subheader("Historical Inspection Log Records")
        st.dataframe(df_logs.sort_values(by="timestamp", ascending=False), use_container_width=True)
    else:
        st.info("No inspection records logged yet. Run screening in the Live Screening tab to generate audit logs.")

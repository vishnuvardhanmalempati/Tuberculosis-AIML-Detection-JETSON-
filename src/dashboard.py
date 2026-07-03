import os
import sys
from pathlib import Path
import datetime
import pandas as pd
from PIL import Image
import streamlit as st
import torch
from torchvision import transforms

# Add root folder to path to allow absolute imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import get_dataloaders
from src.model import get_tuberculosis_model
from src.explain import generate_and_save_gradcam
from src.database import init_db, add_patient, add_scan, get_all_scans, update_doctor_notes
from src.report_generator import generate_pdf_report
from src.benchmark import run_benchmarks

# Page Setup
st.set_page_config(
    page_title="Tuberculosis Medical AI Workstation",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Human-Designed Professional Aesthetics
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;500&display=swap');
    
    /* Base font override */
    html, body, [data-testid="stAppViewContainer"], .stWidgetFormContainer {
        font-family: 'Outfit', sans-serif;
        background-color: #F8FAFC;
    }
    
    /* Top Masthead Header */
    .masthead {
        background: linear-gradient(135deg, #1E293B, #0F172A);
        color: #F8FAFC;
        padding: 30px;
        border-radius: 16px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1), 0 2px 4px -2px rgba(0, 0, 0, 0.1);
        border: 1px solid #334155;
    }
    
    .masthead-title {
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 0;
        background: linear-gradient(to right, #38BDF8, #818CF8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    .masthead-subtitle {
        color: #94A3B8;
        font-size: 1.05rem;
        margin: 6px 0 0 0;
        font-weight: 300;
    }
    
    /* Card Container Wrapper */
    .clinical-card {
        background-color: #FFFFFF;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid #E2E8F0;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.05), 0 1px 2px -1px rgba(0, 0, 0, 0.05);
    }
    
    .clinical-card h3 {
        color: #0F172A;
        font-weight: 600;
        font-size: 1.25rem;
        margin-top: 0;
        margin-bottom: 16px;
        border-bottom: 1px solid #F1F5F9;
        padding-bottom: 10px;
    }
    
    /* Radiology Viewport Lightbox style */
    .lightbox-title {
        color: #94A3B8;
        font-size: 0.8rem;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    /* Clinical Triage Banner */
    .triage-card {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        border-left: 6px solid;
    }
    
    .triage-positive {
        background: linear-gradient(to right, #FEF2F2, #FEE2E2);
        color: #991B1B;
        border-left-color: #EF4444;
        border: 1px solid #FCA5A5;
    }
    
    .triage-negative {
        background: linear-gradient(to right, #F0FDF4, #DCFCE7);
        color: #166534;
        border-left-color: #22C55E;
        border: 1px solid #86EFAC;
    }
    
    .triage-card h4 {
        margin: 0 0 6px 0;
        font-weight: 700;
        font-size: 1.3rem;
    }
    
    .triage-card p {
        margin: 0;
        font-size: 0.95rem;
        font-weight: 400;
    }

    /* Override standard tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: #E2E8F0;
        padding: 6px;
        border-radius: 12px;
        margin-bottom: 20px;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 8px;
        background-color: transparent;
        border: none;
        color: #475569;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #FFFFFF;
        color: #0F172A;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
</style>
""", unsafe_allow_html=True)

# Initialize database on startup
init_db()

# Cached Model Loader
@st.cache_resource
def load_cached_model(model_name, weight_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_tuberculosis_model(model_name, pretrained=True)
    if weight_path and os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model, device

# --- SIDEBAR CONTROLS (TIGHT CONTROL SYSTEM) ---
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-top: 15px;'><img src='https://cdn-icons-png.flaticon.com/512/2870/2870638.png' width='70'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #1E293B; font-weight: 700; margin-bottom: 20px;'>Control Desk</h2>", unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("<h4 style='color: #2563EB; font-weight: 600;'>AI Engine Configuration</h4>", unsafe_allow_html=True)
    model_choice = st.selectbox("Model Architecture", ["efficientnet_b0", "mobilenet_v3"])
    
    # Check for weights file
    model_weight_file = f"models/best_tb_model_{model_choice}.pth"
    weight_status = "Baseline (Pretrained ImageNet)"
    weight_path = None
    
    if os.path.exists(model_weight_file):
        weight_status = "Trained Weights Loaded"
        weight_path = model_weight_file
        st.success(f"✔️ {weight_status}")
    else:
        st.warning(f"⚠️ {weight_status}")
        
    st.divider()
    
    st.markdown("<h4 style='color: #2563EB; font-weight: 600;'>Decision Sensitivity</h4>", unsafe_allow_html=True)
    threshold = st.slider(
        "Triage Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.05,
        help="Adjusting this changes the threshold for TB classification. Lower values increase screening sensitivity."
    )
    
    st.divider()
    st.info("💡 **Clinical Tip:** Lower the threshold to `0.40` to catch early-stage consolidations, improving safety margins.")

# Load the configured model
model, device = load_cached_model(model_choice, weight_path)

# --- TOP MASTHEAD ---
st.markdown("""
<div class="masthead">
    <h1 class="masthead-title">🫁 Tuberculosis AI Medical Workstation</h1>
    <p class="masthead-subtitle">Triage Diagnostic Portal & PACS Imaging Saliency Analyzer</p>
</div>
""", unsafe_allow_html=True)

# --- WORKSPACE TABS (PACS Metaphor) ---
diagnostic_tab, database_tab, analytics_tab = st.tabs([
    "🔬 Diagnostic Workstation", 
    "📂 Patient Registry & Archive", 
    "📊 Performance Analytics"
])

# ==========================================
# 🔬 TAB 1: DIAGNOSTIC WORKSTATION
# ==========================================
with diagnostic_tab:
    col_input, col_view = st.columns([1, 1])
    
    with col_input:
        st.markdown("""
        <div class="clinical-card">
            <h3>1. Patient Demographics & Intake</h3>
        </div>
        """, unsafe_allow_html=True)
        
        # Group inputs within a clean styled intake form
        with st.container():
            patient_id = st.text_input("Patient ID*", value="P-" + datetime.datetime.now().strftime("%y%m%d%H%M"))
            patient_name = st.text_input("Full Name*", placeholder="Enter patient's name")
            
            col_a, col_b = st.columns(2)
            with col_a:
                patient_age = st.number_input("Age*", min_value=0, max_value=120, value=30)
            with col_b:
                patient_gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
                
        st.markdown("""
        <div class="clinical-card" style="margin-top: 15px;">
            <h3>2. Acquire Chest Radiograph</h3>
        </div>
        """, unsafe_allow_html=True)
        
        uploaded_file = st.file_uploader("Upload Frontal Chest Radiograph (CXR)...", type=["png", "jpg", "jpeg"])
        
        if uploaded_file:
            st.image(uploaded_file, caption="Acquired Scan Preview", use_container_width=True)
            
    with col_view:
        st.markdown("""
        <div class="clinical-card">
            <h3>3. Diagnostic Interpretation & Triage</h3>
        </div>
        """, unsafe_allow_html=True)
        
        if uploaded_file and patient_name:
            # Setup directories
            temp_dir = Path("data/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_image_path = temp_dir / f"temp_{patient_id}.png"
            
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            run_diag = st.button("🔬 Execute AI Screening & Visual Mapping", type="primary", use_container_width=True)
            
            if run_diag:
                with st.spinner("Processing radiograph and computing saliency vectors..."):
                    # Inference pipeline
                    img = Image.open(temp_image_path).convert('RGB')
                    preprocess = transforms.Compose([
                        transforms.Resize((224, 224)),
                        transforms.ToTensor(),
                        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    ])
                    input_tensor = preprocess(img).unsqueeze(0).to(device)
                    
                    with torch.no_grad():
                        logits = model(input_tensor)
                        prob = torch.sigmoid(logits).item()
                        
                    predicted_label = "Tuberculosis" if prob >= threshold else "Normal"
                    
                    # Generate Grad-CAM Saliency Overlay
                    gradcam_save_path = temp_dir / f"gradcam_{patient_id}.png"
                    generate_and_save_gradcam(model, model_choice, temp_image_path, gradcam_save_path, device)
                    
                    # Store to session state to prevent state loss
                    st.session_state['diag_run'] = True
                    st.session_state['prob'] = prob
                    st.session_state['label'] = predicted_label
                    st.session_state['temp_img'] = str(temp_image_path)
                    st.session_state['gradcam_img'] = str(gradcam_save_path)
                    st.session_state['doctor_notes'] = f"Radiographical features are {'suggestive of active Tuberculosis' if predicted_label == 'Tuberculosis' else 'within normal limits'}. Grad-CAM maps confirm region focus."
            
            if st.session_state.get('diag_run'):
                prob = st.session_state['prob']
                predicted_label = st.session_state['label']
                temp_image_path = st.session_state['temp_img']
                gradcam_save_path = st.session_state['gradcam_img']
                
                # Clinical triage outcomes
                if predicted_label == "Tuberculosis":
                    st.markdown(f"""
                    <div class="triage-card triage-positive">
                        <h4>⚠️ Tuberculosis Suspicion: HIGH</h4>
                        <p>Inference score: <strong>{prob*100:.1f}%</strong> (Decision threshold set at {threshold*100:.0f}%)</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="triage-card triage-negative">
                        <h4>✅ Screening Normal: CLEAR</h4>
                        <p>Inference score: <strong>{(1-prob)*100:.1f}%</strong> confidence normal</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                # Side-by-Side PACS Lightbox Viewing
                st.markdown("<p class='lightbox-title'>PACS Image Viewport</p>", unsafe_allow_html=True)
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(temp_image_path, caption="Original CXR", use_container_width=True)
                with img_col2:
                    st.image(gradcam_save_path, caption="Grad-CAM Saliency Overlay", use_container_width=True)
                
                st.divider()
                
                # Doctor Remarks Form
                st.markdown("##### 🩺 Clinical Remarks & Database Sync")
                doc_notes = st.text_area("Observations / Follow-up Plan", value=st.session_state['doctor_notes'])
                
                col_sync, col_pdf = st.columns(2)
                with col_sync:
                    if st.button("📝 Log Scan and Metadata", use_container_width=True):
                        add_patient(patient_id, patient_name, patient_age, patient_gender)
                        add_scan(patient_id, temp_image_path, prob, predicted_label, str(gradcam_save_path), doc_notes)
                        st.success("Record synced with SQLite.")
                        
                with col_pdf:
                    p_info = {"patient_id": patient_id, "name": patient_name, "age": patient_age, "gender": patient_gender}
                    s_info = {
                        "scan_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "image_path": temp_image_path,
                        "prediction_score": prob,
                        "prediction_label": predicted_label,
                        "gradcam_path": gradcam_save_path,
                        "doctor_notes": doc_notes
                    }
                    pdf_report_path = f"data/reports/report_{patient_id}.pdf"
                    generate_pdf_report(p_info, s_info, pdf_report_path)
                    
                    with open(pdf_report_path, "rb") as f:
                        pdf_data = f.read()
                    st.download_button(
                        label="📥 Download Clinical PDF Report",
                        data=pdf_data,
                        file_name=f"TB_Report_{patient_id}.pdf",
                        mime="application/pdf",
                        use_container_width=True
                    )
        else:
            st.warning("Please complete the patient demographics and upload a chest radiograph to activate the workstation.")

# ==========================================
# 📂 TAB 2: PATIENT REGISTRY & ARCHIVE
# ==========================================
with database_tab:
    st.markdown("""
    <div class="clinical-card">
        <h3>Patient Scan Registry</h3>
    </div>
    """, unsafe_allow_html=True)
    
    scans = get_all_scans()
    
    if len(scans) == 0:
        st.info("Registry is currently empty. Run scans in the Diagnostic Workstation to compile database records.")
    else:
        df = pd.DataFrame(scans)
        
        # Registry Search Filter
        search_query = st.text_input("🔍 Search registry by Patient Name or ID:")
        if search_query:
            df = df[df['patient_id'].str.contains(search_query, case=False) | df['name'].str.contains(search_query, case=False)]
            
        df_display = df[['scan_date', 'patient_id', 'name', 'age', 'gender', 'prediction_label', 'prediction_score']].copy()
        df_display['prediction_score'] = df_display['prediction_score'].map(lambda x: f"{x*100:.1f}%")
        
        st.dataframe(df_display, use_container_width=True)
        
        st.divider()
        
        # Clinical Archive Inspector
        st.markdown("### 🔍 Case History Inspector")
        selected_scan_id = st.selectbox(
            "Select Registry ID to Inspect", 
            df['id'].tolist(), 
            format_func=lambda x: f"Scan #{x} - {df[df['id']==x]['name'].values[0]} ({df[df['id']==x]['patient_id'].values[0]})"
        )
        
        if selected_scan_id:
            row = df[df['id'] == selected_scan_id].iloc[0]
            
            ins1, ins2 = st.columns([1, 1])
            with ins1:
                st.markdown(f"""
                <div style='background-color: white; padding: 20px; border-radius: 12px; border: 1px solid #E2E8F0;'>
                    <h5 style='margin-top: 0; color: #1E293B;'>Clinical Details</h5>
                    <p><strong>Patient Name:</strong> {row['name']}</p>
                    <p><strong>Patient ID:</strong> {row['patient_id']}</p>
                    <p><strong>Demographics:</strong> {row['age']} yrs / {row['gender']}</p>
                    <p><strong>Screening Date:</strong> {row['scan_date']}</p>
                    <p><strong>Diagnostic Outcome:</strong> {row['prediction_label']} ({row['prediction_score']*100:.1f}%)</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                new_notes = st.text_area("Observations Log", value=row['doctor_notes'], key=f"notes_{selected_scan_id}")
                
                col_up, col_dn = st.columns(2)
                with col_up:
                    if st.button("💾 Update Notes Log", key=f"btn_up_{selected_scan_id}", use_container_width=True):
                        update_doctor_notes(selected_scan_id, new_notes)
                        st.success("Notes synchronized.")
                        st.rerun()
                        
                with col_dn:
                    p_info = {"patient_id": row['patient_id'], "name": row['name'], "age": row['age'], "gender": row['gender']}
                    s_info = {
                        "scan_date": row['scan_date'],
                        "image_path": row['image_path'],
                        "prediction_score": row['prediction_score'],
                        "prediction_label": row['prediction_label'],
                        "gradcam_path": row['gradcam_path'],
                        "doctor_notes": new_notes
                    }
                    pdf_report_path = f"data/reports/report_{row['patient_id']}.pdf"
                    generate_pdf_report(p_info, s_info, pdf_report_path)
                    
                    with open(pdf_report_path, "rb") as f:
                        pdf_data = f.read()
                    st.download_button(
                        label="📥 Download Updated PDF Report",
                        data=pdf_data,
                        file_name=f"TB_Report_{row['patient_id']}.pdf",
                        mime="application/pdf",
                        key=f"dl_pdf_{selected_scan_id}",
                        use_container_width=True
                    )
                    
            with ins2:
                img_col_a, img_col_b = st.columns(2)
                with img_col_a:
                    if os.path.exists(row['image_path']):
                        st.image(row['image_path'], caption="Original CXR", use_container_width=True)
                with img_col_b:
                    if row['gradcam_path'] and os.path.exists(row['gradcam_path']):
                        st.image(row['gradcam_path'], caption="Grad-CAM Overlay", use_container_width=True)

# ==========================================
# 📊 TAB 3: PERFORMANCE ANALYTICS
# ==========================================
with analytics_tab:
    metrics_tab, bench_tab = st.tabs(["Clinical Validation Metrics", "Hardware Edge Benchmarking"])
    
    with metrics_tab:
        st.markdown("### Clinical Performance Analytics (Test Validation Set)")
        
        metrics_file = f"models/metrics_{model_choice}.json"
        
        if os.path.exists(metrics_file):
            import json
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
                
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Validation Accuracy", f"{metrics_data['accuracy']*100:.2f}%")
            col_m2.metric("Sensitivity / Recall (Safety)", f"{metrics_data['sensitivity_recall']*100:.2f}%")
            col_m3.metric("Specificity", f"{metrics_data['specificity']*100:.2f}%")
            col_m4.metric("AUC-ROC", f"{metrics_data['auc_roc']:.4f}")
            
            st.divider()
            
            st.markdown("#### Validation Curves & Performance Plots")
            img_p1, img_p2, img_p3 = st.columns(3)
            
            hist_plot = f"models/training_history_{model_choice}.png"
            cm_plot = f"models/confusion_matrix_{model_choice}.png"
            roc_plot = f"models/roc_curve_{model_choice}.png"
            
            with img_p1:
                if os.path.exists(hist_plot):
                    st.image(hist_plot, caption="Training History Curves", use_container_width=True)
            with img_p2:
                if os.path.exists(cm_plot):
                    st.image(cm_plot, caption="Test Confusion Matrix", use_container_width=True)
            with img_p3:
                if os.path.exists(roc_plot):
                    st.image(roc_plot, caption="Test ROC Curve", use_container_width=True)
        else:
            st.info("Metrics not found. Run model training (`main.py`) to generate validation curves and final test performance scores.")
            
    with bench_tab:
        st.markdown("### Live Hardware Benchmarking (Edge Performance)")
        st.markdown("Execute live tests on this server/device to calculate execution latency and system resource requirements:")
        
        onnx_file = f"models/best_tb_model_{model_choice}.onnx"
        onnx_status = "Available" if os.path.exists(onnx_file) else "Not Compiled"
        st.text(f"ONNX Model Status: {onnx_status}")
        
        if st.button("🚀 Run Live Latency & Memory Benchmark", use_container_width=True):
            with st.spinner("Measuring inference latency..."):
                results = run_benchmarks(
                    pytorch_model=model,
                    model_name=model_choice,
                    onnx_path=onnx_file if os.path.exists(onnx_file) else None
                )
                
                # Render results in table
                table_rows = []
                for engine, data in results.items():
                    table_rows.append({
                        "Inference Engine / Device": data['device'],
                        "Avg Latency (ms)": f"{data['avg_latency_ms']:.2f} ms",
                        "Throughput (FPS)": f"{data['throughput_fps']:.2f} FPS",
                        "RAM footprint (MB)": f"{data['ram_process_mb']:.1f} MB"
                    })
                    
                st.table(pd.DataFrame(table_rows))
                st.success("Benchmarks completed successfully.")

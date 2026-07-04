import os
import sys
import json
import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import cv2
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

# Page Configuration (PACS Viewport Layout)
st.set_page_config(
    page_title="PACS Tuberculosis AI Workstation",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for the "Clinical Radiology" Lightbox Theme (Dark, High-Contrast Cyan/Amber Accent)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=JetBrains+Mono:wght@400;600;700&display=swap');
    
    /* Global App Container Override */
    html, body, [data-testid="stAppViewContainer"], .stWidgetFormContainer {
        background-color: #0B0F19 !important;
        color: #E2E8F0 !important;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Sidebar Overrides */
    [data-testid="stSidebar"] {
        background-color: #111827 !important;
        border-right: 1px solid #1F2937;
    }
    
    /* Clinical Triage Disclaimer Header */
    .clinical-disclaimer {
        background: linear-gradient(135deg, #1E293B, #0F172A);
        border: 1px solid #3B82F6;
        border-left: 6px solid #3B82F6;
        border-radius: 12px;
        padding: 16px 20px;
        margin-bottom: 24px;
        color: #93C5FD;
    }
    
    .disclaimer-title {
        font-weight: 700;
        font-size: 1.05rem;
        margin-top: 0;
        margin-bottom: 4px;
        color: #60A5FA;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    
    .disclaimer-text {
        font-size: 0.9rem;
        line-height: 1.4;
        margin: 0;
    }
    
    /* PACS Workstation Title Block */
    .pacs-title-block {
        background: #111827;
        border: 1px solid #1F2937;
        border-radius: 16px;
        padding: 24px 30px;
        margin-bottom: 25px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.3);
    }
    
    .pacs-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.03em;
        margin: 0;
        color: #F8FAFC;
    }
    
    .pacs-title span {
        color: #06B6D4; /* Diagnostic Cyan */
    }
    
    .pacs-subtitle {
        color: #9CA3AF;
        font-size: 1rem;
        margin: 4px 0 0 0;
        font-weight: 300;
        font-family: 'JetBrains Mono', monospace;
    }
    
    /* Radiology Diagnostic Card */
    .radiology-card {
        background-color: #111827;
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        border: 1px solid #1F2937;
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.3);
    }
    
    .radiology-card h3 {
        color: #F8FAFC;
        font-weight: 600;
        font-size: 1.2rem;
        margin-top: 0;
        margin-bottom: 16px;
        border-bottom: 1px solid #1F2937;
        padding-bottom: 10px;
        letter-spacing: 0.02em;
    }
    
    /* Diagnostic Lightbox viewport */
    .lightbox-viewport {
        background-color: #030712;
        border: 3px solid #1F2937;
        border-radius: 12px;
        padding: 16px;
        box-shadow: inset 0 2px 8px rgba(0, 0, 0, 0.8);
        text-align: center;
    }
    
    .lightbox-tag {
        color: #6B7280;
        font-size: 0.75rem;
        font-family: 'JetBrains Mono', monospace;
        margin-bottom: 8px;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    /* Clinical Risk Banners */
    .risk-banner {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 24px;
        border: 1px solid;
    }
    
    .risk-positive {
        background: linear-gradient(135deg, #7F1D1D, #450A0A);
        color: #FCA5A5;
        border-color: #EF4444;
        border-left: 6px solid #EF4444;
    }
    
    .risk-negative {
        background: linear-gradient(135deg, #064E3B, #022C22);
        color: #A7F3D0;
        border-color: #10B981;
        border-left: 6px solid #10B981;
    }
    
    .risk-banner h4 {
        margin: 0 0 6px 0;
        font-weight: 700;
        font-size: 1.25rem;
        letter-spacing: 0.02em;
    }
    
    .risk-banner p {
        margin: 0;
        font-size: 0.95rem;
    }
    
    /* Monospaced Metrics display */
    .diagnostic-val {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        font-size: 1.4rem;
        color: #06B6D4;
    }
    
    /* Override standard tab styling for Radiology theme */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        background-color: #111827;
        padding: 6px;
        border-radius: 12px;
        margin-bottom: 20px;
        border: 1px solid #1F2937;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 10px 24px;
        border-radius: 8px;
        background-color: transparent;
        border: none;
        color: #9CA3AF;
        font-weight: 600;
        transition: all 0.2s ease-in-out;
    }
    
    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background-color: #1F2937;
        color: #06B6D4;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Initialize database registry
init_db()

# Clinical Preprocessing Helpers
def apply_clahe_enhancement(image_path, save_path):
    """
    Applies Contrast Limited Adaptive Histogram Equalization (CLAHE)
    to enhance X-ray image contrast (standard clinical practice).
    """
    img_gray = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl_img = clahe.apply(img_gray)
    cl_rgb = cv2.cvtColor(cl_img, cv2.COLOR_GRAY2RGB)
    cv2.imwrite(str(save_path), cl_rgb)

def validate_chest_xray(image_path):
    """
    Performs lightweight structural check on the image to verify 
    if it matches standard grayscale chest radiograph characteristics.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return False, "Unable to read image file."
        
    h, w, c = img.shape
    aspect_ratio = w / h
    
    # 1. Check aspect ratio (CXR PA views are typically square/vertical: aspect ratio between 0.65 and 1.35)
    if aspect_ratio < 0.60 or aspect_ratio > 1.40:
        return False, f"Aspect ratio ({aspect_ratio:.2f}) does not match standard PA chest radiograph structure."
        
    # 2. Check channel color uniformity (X-rays are grayscale, meaning R, G, B channels should have minimal variance)
    b, g, r = cv2.split(img)
    diff_rg = np.mean(np.abs(r.astype(np.int16) - g.astype(np.int16)))
    diff_gb = np.mean(np.abs(g.astype(np.int16) - b.astype(np.int16)))
    
    if diff_rg > 15.0 or diff_gb > 15.0:
        return False, "Image contains high color variance. Grayscale chest radiograph expected."
        
    return True, "Valid CXR format."

# Cached Model Loader
@st.cache_resource
def load_cached_model(model_name, weight_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_tuberculosis_model(model_name, pretrained=True)
    if weight_path and os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model, device

# --- SIDEBAR CONTROL DESK ---
with st.sidebar:
    st.markdown("<div style='text-align: center; margin-top: 15px;'><img src='https://cdn-icons-png.flaticon.com/512/2870/2870638.png' width='70'></div>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: center; color: #F8FAFC; font-weight: 700; margin-bottom: 20px;'>Control Desk</h2>", unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("<h4 style='color: #06B6D4; font-weight: 600;'>AI Engine Configuration</h4>", unsafe_allow_html=True)
    model_choice = st.selectbox("Model Architecture", ["efficientnet_b0", "mobilenet_v3"])
    
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
    
    st.markdown("<h4 style='color: #06B6D4; font-weight: 600;'>Triage Threshold</h4>", unsafe_allow_html=True)
    threshold = st.slider(
        "Triage Cutoff", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.05,
        help="Adjusting this changes the threshold for TB classification. Lower values increase screening sensitivity."
    )
    
    st.divider()
    st.info("💡 **Clinical Tip:** Set cutoff to `0.40` or `0.45` to optimize screening sensitivity and prevent false negatives.")

# Load the configured model
model, device = load_cached_model(model_choice, weight_path)

# --- 1. CLINICAL DISCLAIMER (HONEST FRAMING) ---
st.markdown("""
<div class="clinical-disclaimer">
    <div class="disclaimer-title">🏥 Clinical Decision Support Tool</div>
    <div class="disclaimer-text">
        This portal acts as a screening triage assistant for clinicians. It does not replace clinical evaluation or microbiological confirmation (e.g., GeneXpert molecular testing or sputum culture). All AI predictions and saliency mappings must be reviewed by a certified medical professional.
    </div>
</div>
""", unsafe_allow_html=True)

# --- 2. PAC MASTHEAD TITLE ---
st.markdown("""
<div class="pacs-title-block">
    <h1 class="pacs-title">PACS <span>Tuberculosis AI Workstation</span></h1>
    <p class="pacs-subtitle">System Version: 2.1-Edge // Engine Device: {}</p>
</div>
""".format("NVIDIA GPU (CUDA)" if torch.cuda.is_available() else "Cortex-A57 CPU"), unsafe_allow_html=True)

# --- 3. PACS TABS ---
diag_tab, db_tab, metric_tab, edu_tab = st.tabs([
    "🔬 Diagnostic Workstation",
    "📂 Patient Archive & Registry",
    "📊 Performance Metrics",
    "📖 Clinical Education & Resources"
])

# ==========================================
# 🔬 TAB 1: DIAGNOSTIC WORKSTATION
# ==========================================
with diag_tab:
    col_left, col_right = st.columns([1, 1])
    
    with col_left:
        # Patient Demographic Intake Card
        st.markdown("<div class='radiology-card'><h3>1. Patient Demographics & Registry</h3></div>", unsafe_allow_html=True)
        patient_id = st.text_input("Patient ID*", value="P-" + datetime.datetime.now().strftime("%y%m%d%H%M"))
        patient_name = st.text_input("Full Name*", placeholder="e.g. Jane Doe")
        
        col_age, col_gen = st.columns(2)
        with col_age:
            patient_age = st.number_input("Age*", min_value=0, max_value=120, value=30)
        with col_gen:
            patient_gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
            
        # Symptom Questionnaire Card
        st.markdown("<div class='radiology-card' style='margin-top: 15px;'><h3>2. Clinical Symptom Questionnaire</h3></div>", unsafe_allow_html=True)
        st.markdown("<p style='color: #9CA3AF; font-size: 0.9rem; margin-bottom: 15px;'>Toggle patient symptoms to calculate overall combined clinical risk profile:</p>", unsafe_allow_html=True)
        
        s_cough = st.checkbox("Persistent Cough (> 2 weeks)")
        s_sputum = st.checkbox("Hemoptysis (Blood in Sputum)")
        s_sweats = st.checkbox("Night Sweats")
        s_loss = st.checkbox("Unexplained Weight Loss")
        s_fever = st.checkbox("Fever / Chills")
        
        # Radiograph Upload Card
        st.markdown("<div class='radiology-card' style='margin-top: 15px;'><h3>3. Acquire Chest Radiograph</h3></div>", unsafe_allow_html=True)
        uploaded_file = st.file_uploader("Select Chest X-ray (PA View)...", type=["png", "jpg", "jpeg"])
        
    with col_right:
        st.markdown("<div class='radiology-card'><h3>4. Diagnostic Interpretation & PACS Viewer</h3></div>", unsafe_allow_html=True)
        
        if uploaded_file and patient_name:
            # Save uploaded file
            temp_dir = Path("data/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            original_path = temp_dir / f"orig_{patient_id}.png"
            enhanced_path = temp_dir / f"enhanced_{patient_id}.png"
            
            with open(original_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            # Perform Image Validation
            is_valid, validation_msg = validate_chest_xray(original_path)
            
            if not is_valid:
                st.error(f"⚠️ Scan Validation Error: {validation_msg}")
                st.info("Ensure you upload a standard, grayscale, frontal Chest X-ray image (PA/AP view).")
            else:
                # Apply CLAHE Enhancement
                apply_clahe_enhancement(original_path, enhanced_path)
                
                run_diag = st.button("🔬 Execute AI Screening & Visual Mapping", type="primary", use_container_width=True)
                
                if run_diag:
                    with st.spinner("Processing radiograph, applying CLAHE, and computing saliency maps..."):
                        # Preprocessing & Inference
                        img = Image.open(enhanced_path).convert('RGB')
                        preprocess = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                        ])
                        input_tensor = preprocess(img).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            logits = model(input_tensor)
                            prob = torch.sigmoid(logits).item()
                            
                        # Generate Grad-CAM image
                        gradcam_save_path = temp_dir / f"gradcam_{patient_id}.png"
                        generate_and_save_gradcam(model, model_choice, enhanced_path, gradcam_save_path, device)
                        
                        # Store in session state
                        st.session_state['diag_active'] = True
                        st.session_state['prob'] = prob
                        st.session_state['orig_img'] = str(original_path)
                        st.session_state['enhanced_img'] = str(enhanced_path)
                        st.session_state['gradcam_img'] = str(gradcam_save_path)
                        
                if st.session_state.get('diag_active'):
                    prob = st.session_state['prob']
                    orig_img_path = st.session_state['orig_img']
                    enhanced_img_path = st.session_state['enhanced_img']
                    gradcam_img_path = st.session_state['gradcam_img']
                    
                    # 1. Calculate Combined Clinical Risk
                    # Symptom weights: Hemoptysis (+4), Cough (+3), Weight loss (+2), Night sweats (+1), Fever (+1) -> Total possible: 11
                    sym_points = (4 if s_sputum else 0) + (3 if s_cough else 0) + (2 if s_loss else 0) + (1 if s_sweats else 0) + (1 if s_fever else 0)
                    prob_symptoms = sym_points / 11.0
                    
                    # Combined score: 70% X-ray model probability + 30% symptom profile
                    combined_score = (0.70 * prob) + (0.30 * prob_symptoms)
                    predicted_label = "Tuberculosis" if combined_score >= threshold else "Normal"
                    
                    # Display Diagnostic Risk Banner
                    if predicted_label == "Tuberculosis":
                        st.markdown(f"""
                        <div class="risk-banner risk-positive">
                            <h4>⚠️ Tuberculosis Suspicion: HIGH</h4>
                            <p>Combined Risk Score is <strong>{combined_score*100:.1f}%</strong> (Cutoff threshold set at {threshold*100:.0f}%)</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="risk-banner risk-negative">
                            <h4>✅ Screening Outcome: Normal / Low Risk</h4>
                            <p>Combined Risk Score is <strong>{combined_score*100:.1f}%</strong>. Active Tuberculosis characteristics not prominent.</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                    # Detailed Metrics Readouts
                    col_m1, col_m2, col_m3 = st.columns(3)
                    with col_m1:
                        st.markdown(f"AI Model Score:<br><span class='diagnostic-val'>{prob*100:.1f}%</span>", unsafe_allow_html=True)
                    with col_m2:
                        st.markdown(f"Symptom Score:<br><span class='diagnostic-val'>{prob_symptoms*100:.1f}%</span>", unsafe_allow_html=True)
                    with col_m3:
                        st.markdown(f"Combined Score:<br><span class='diagnostic-val'>{combined_score*100:.1f}%</span>", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Signature Interaction: Interactive Visual Saliency Blender Slider
                    st.markdown("##### 🎛️ Visual Saliency Blender")
                    alpha = st.slider("Alpha Blend Control (CLAHE Scan vs Grad-CAM Heatmap)", 0.0, 1.0, 0.5, step=0.05)
                    
                    # Compute Blended Image dynamically
                    img_enhanced = cv2.imread(enhanced_img_path)
                    img_gradcam = cv2.imread(gradcam_img_path)
                    blended_img = cv2.addWeighted(img_enhanced, 1 - alpha, img_gradcam, alpha, 0)
                    
                    # Convert to RGB to display in Streamlit
                    blended_rgb = cv2.cvtColor(blended_img, cv2.COLOR_BGR2RGB)
                    
                    # Display in radiology dark viewport
                    st.markdown("<div class='lightbox-viewport'>", unsafe_allow_html=True)
                    st.markdown("<p class='lightbox-tag'>Active Triage Lightbox Viewport</p>", unsafe_allow_html=True)
                    st.image(blended_rgb, use_container_width=True)
                    st.markdown("</div>", unsafe_allow_html=True)
                    
                    st.divider()
                    
                    # Log notes & syncing
                    doc_notes = st.text_area("Observations / Follow-up notes", value=f"Combined Risk Score: {combined_score*100:.1f}% ({predicted_label}). Patient presents with {sym_points}/11 symptom severity score. AI heatmap shows active consolidations focus.")
                    
                    col_sync, col_pdf = st.columns(2)
                    with col_sync:
                        if st.button("📝 Log Scan and Findings", use_container_width=True):
                            add_patient(patient_id, patient_name, patient_age, patient_gender)
                            add_scan(patient_id, enhanced_img_path, combined_score, predicted_label, str(gradcam_img_path), doc_notes)
                            st.success("Synchronized with Database.")
                            
                    with col_pdf:
                        p_info = {"patient_id": patient_id, "name": patient_name, "age": patient_age, "gender": patient_gender}
                        s_info = {
                            "scan_date": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                            "image_path": original_path,
                            "prediction_score": combined_score,
                            "prediction_label": predicted_label,
                            "gradcam_path": gradcam_img_path,
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
            st.info("Input patient profile details, verify symptoms, and upload a Chest X-ray to activate the workstation.")

# ==========================================
# 📂 TAB 2: PATIENT ARCHIVE & REGISTRY
# ==========================================
with db_tab:
    st.markdown("<div class='radiology-card'><h3>Clinical Patients Archive</h3></div>", unsafe_allow_html=True)
    scans = get_all_scans()
    
    if len(scans) == 0:
        st.info("No records found in database registry. Perform screenings to compile archives.")
    else:
        df = pd.DataFrame(scans)
        
        search_query = st.text_input("🔍 Filter Registry by Name or ID:")
        if search_query:
            df = df[df['patient_id'].str.contains(search_query, case=False) | df['name'].str.contains(search_query, case=False)]
            
        df_display = df[['scan_date', 'patient_id', 'name', 'age', 'gender', 'prediction_label', 'prediction_score']].copy()
        df_display['prediction_score'] = df_display['prediction_score'].map(lambda x: f"{x*100:.1f}%")
        st.dataframe(df_display, use_container_width=True)
        
        st.divider()
        
        st.markdown("### 🔍 Scan Registry Inspector")
        selected_scan_id = st.selectbox(
            "Select Scan to Inspect", 
            df['id'].tolist(),
            format_func=lambda x: f"Scan #{x} - {df[df['id']==x]['name'].values[0]} ({df[df['id']==x]['patient_id'].values[0]})"
        )
        
        if selected_scan_id:
            row = df[df['id'] == selected_scan_id].iloc[0]
            
            ins1, ins2 = st.columns([1, 1])
            with ins1:
                st.markdown(f"""
                <div style='background-color: #111827; padding: 20px; border-radius: 12px; border: 1px solid #1F2937;'>
                    <h5 style='margin-top: 0; color: #06B6D4;'>Demographic Details</h5>
                    <p><strong>Name:</strong> {row['name']} | <strong>ID:</strong> {row['patient_id']}</p>
                    <p><strong>Demographics:</strong> {row['age']} yrs / {row['gender']}</p>
                    <p><strong>Scan Date:</strong> {row['scan_date']}</p>
                    <p><strong>Risk Score:</strong> {row['prediction_score']*100:.1f}% ({row['prediction_label']})</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                new_notes = st.text_area("Observations Log", value=row['doctor_notes'], key=f"notes_{selected_scan_id}")
                
                col_up, col_dn = st.columns(2)
                with col_up:
                    if st.button("💾 Synchronize Logs", key=f"btn_up_{selected_scan_id}", use_container_width=True):
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
# 📊 TAB 3: PERFORMANCE METRICS
# ==========================================
with metric_tab:
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

# ==========================================
# 📖 TAB 4: CLINICAL EDUCATION & RESOURCES
# ==========================================
with edu_tab:
    st.markdown("<div class='radiology-card'><h3>Clinical Information & Diagnostic Resources</h3></div>", unsafe_allow_html=True)
    
    col_edu1, col_edu2 = st.columns(2)
    with col_edu1:
        st.markdown("""
        #### Understanding Tuberculosis (TB)
        Tuberculosis is a potentially serious infectious disease that primarily affects the lungs (Pulmonary TB). It is spread through the air when people who have active TB in their lungs cough, sneeze, or spit.
        
        ##### Core Clinical Symptoms
        *   **Persistent Cough**: Lasting 2 weeks or more.
        *   **Hemoptysis**: Coughing up blood or blood-streaked sputum.
        *   **Systemic Symptoms**: Night sweats, unexplained weight loss, low-grade fever, and fatigue.
        
        ##### Diagnostic Guidelines (WHO Standards)
        1.  **Chest Radiography (CXR)**: High-sensitivity first-line screening tool to detect opacities, consolidations, or cavities.
        2.  **Molecular Assays**: GeneXpert MTB/RIF for rapid molecular diagnosis and rifampicin-resistance detection.
        3.  **Sputum Microscopy & Culture**: Traditional gold standards for detecting active acid-fast bacilli (AFB).
        """)
        
    with col_edu2:
        st.markdown("""
        #### National & Global Resources
        
        ##### India's National TB Elimination Program (NTEP)
        The Government of India is committed to eliminating TB by 2025 through the National Tuberculosis Elimination Program.
        *   **Ni-kshay Portal**: The unified ICT system for TB patient tracking and notification in India.
        *   **DOTS (Directly Observed Treatment, Short-course)**: Standardized treatment guidelines provided free of charge through local health centers.
        
        ##### Useful Clinical Links
        *   🔗 [WHO Global Tuberculosis Report](https://www.who.int/teams/global-tuberculosis-programme/tb-reports)
        *   🔗 [Central TB Division, Ministry of Health, India](https://tbcindia.gov.in)
        *   🔗 [Ni-kshay Portal](https://www.nikshay.in)
        *   🔗 [WHO Operational Handbook on Tuberculosis (CXR Screening)](https://www.who.int/publications/i/item/9789240022614)
        """)

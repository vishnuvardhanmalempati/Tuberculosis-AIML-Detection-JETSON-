import os
import sys
from pathlib import Path
import datetime
import pandas as pd
from PIL import Image
import streamlit as st
import torch

# Add root folder to path to allow absolute imports
sys.path.append(str(Path(__file__).parent.parent))

from src.data_loader import get_dataloaders
from src.model import get_tuberculosis_model
from src.explain import generate_and_save_gradcam, overlay_heatmap, get_gradcam_target_layer, GradCAM
from src.database import init_db, add_patient, add_scan, get_all_scans, get_patient_scans, update_doctor_notes
from src.report_generator import generate_pdf_report
from src.benchmark import run_benchmarks

# Page Setup
st.set_page_config(
    page_title="Tuberculosis Medical AI Diagnostic Portal",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Styling (Rich Aesthetics)
st.markdown("""
<style>
    .reportview-container {
        background: #F7FAFC;
    }
    .main-title {
        color: #1A365D;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
        margin-bottom: 2px;
    }
    .subtitle {
        color: #4A5568;
        font-size: 1.1rem;
        margin-bottom: 25px;
    }
    .status-card {
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        border-left: 5px solid;
    }
    .status-positive {
        background-color: #FED7D7;
        color: #9B2C2C;
        border-left-color: #E53E3E;
    }
    .status-negative {
        background-color: #C6F6D5;
        color: #22543D;
        border-left-color: #38A169;
    }
</style>
""", unsafe_allow_html=True)

# Initialize database
init_db()

# Cache model loader to speed up dashboard
@st.cache_resource
def load_cached_model(model_name, weight_path=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = get_tuberculosis_model(model_name, pretrained=True)
    if weight_path and os.path.exists(weight_path):
        model.load_state_dict(torch.load(weight_path, map_location=device))
    model.eval()
    return model, device

# --- SIDEBAR CONTROLS ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2870/2870638.png", width=80)
    st.markdown("<h2 style='color: #1A365D; font-weight: 700;'>Clinical Control Panel</h2>", unsafe_allow_html=True)
    
    # Navigation
    app_mode = st.radio(
        "Navigation",
        ["Diagnostic Portal", "Patient Database", "System Analytics & Benchmarks"]
    )
    
    st.divider()
    
    # Model configuration
    st.markdown("<h4 style='color: #2B6CB0;'>AI Configuration</h4>", unsafe_allow_html=True)
    model_choice = st.selectbox("Model Architecture", ["mobilenet_v3", "efficientnet_b0"])
    
    # Find weight file in models directory
    model_weight_file = f"models/best_tb_model_{model_choice}.pth"
    weight_status = "Pretrained (ImageNet Baseline)"
    weight_path = None
    
    if os.path.exists(model_weight_file):
        weight_status = f"Trained Model Weights Loaded"
        weight_path = model_weight_file
    st.info(f"Model State: **{weight_status}**")
    
    # Threshold configuration
    threshold = st.slider(
        "Classification Decision Threshold", 
        min_value=0.1, 
        max_value=0.9, 
        value=0.5, 
        step=0.05,
        help="Adjusting this slider changes the sensitivity and specificity of the system. Lower threshold increases sensitivity (finds more cases), higher threshold increases specificity (avoids false alarms)."
    )

# Load selected model
model, device = load_cached_model(model_choice, weight_path)

# --- MAIN APP LOGIC ---

if app_mode == "Diagnostic Portal":
    st.markdown("<h1 class='main-title'>🫁 Tuberculosis AI Diagnostic Portal</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Production-grade clinical decision support system using chest radiography</p>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.subheader("1. Patient Profile")
        col_id, col_name = st.columns([1, 2])
        with col_id:
            patient_id = st.text_input("Patient ID*", value="P-" + datetime.datetime.now().strftime("%y%m%d%H%M"))
        with col_name:
            patient_name = st.text_input("Patient Full Name*", placeholder="e.g. John Doe")
            
        col_age, col_gen = st.columns(2)
        with col_age:
            patient_age = st.number_input("Age*", min_value=0, max_value=120, value=30)
        with col_gen:
            patient_gender = st.selectbox("Gender*", ["Male", "Female", "Other"])
            
        st.subheader("2. Upload Chest X-Ray")
        uploaded_file = st.file_uploader("Select Frontal Chest Radiograph (CXR)...", type=["png", "jpg", "jpeg"])
        
        if uploaded_file:
            st.image(uploaded_file, caption="Uploaded Chest X-ray", use_container_width=True)
            
    with col2:
        st.subheader("3. AI Diagnostics & Findings")
        
        if uploaded_file and patient_name:
            # Save uploaded file temporarily
            temp_dir = Path("data/temp")
            temp_dir.mkdir(parents=True, exist_ok=True)
            temp_image_path = temp_dir / f"temp_{patient_id}.png"
            
            with open(temp_image_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
                
            run_diag = st.button("🔬 Run Automated AI Diagnostic", type="primary")
            
            if run_diag:
                with st.spinner("Analyzing radiograph & generating saliency maps..."):
                    # Step 1: Preprocess and run prediction
                    from PIL import Image
                    from torchvision import transforms
                    
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
                        
                    # Classification based on user threshold
                    predicted_label = "Tuberculosis" if prob >= threshold else "Normal"
                    
                    # Step 2: Generate Grad-CAM image
                    gradcam_save_path = temp_dir / f"gradcam_{patient_id}.png"
                    generate_and_save_gradcam(model, model_choice, temp_image_path, gradcam_save_path, device)
                    
                    # Store session variables
                    st.session_state['diagnosis_run'] = True
                    st.session_state['prob'] = prob
                    st.session_state['label'] = predicted_label
                    st.session_state['temp_img'] = str(temp_image_path)
                    st.session_state['gradcam_img'] = str(gradcam_save_path)
                    
            if st.session_state.get('diagnosis_run'):
                prob = st.session_state['prob']
                predicted_label = st.session_state['label']
                temp_image_path = st.session_state['temp_img']
                gradcam_save_path = st.session_state['gradcam_img']
                
                # Visual output card
                if predicted_label == "Tuberculosis":
                    st.markdown(f"""
                    <div class="status-card status-positive">
                        <h3>⚠️ Tuberculosis Suspicion High</h3>
                        <p>AI Screening indicates findings highly suggestive of Tuberculosis. Confidence: {prob*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="status-card status-negative">
                        <h3>✅ Screening Normal</h3>
                        <p>No significant radiographic signs of active Tuberculosis detected. Confidence: {(1-prob)*100:.1f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                st.progress(prob)
                
                # Show images side-by-side
                st.subheader("Visual Explanations (Grad-CAM)")
                img_col1, img_col2 = st.columns(2)
                with img_col1:
                    st.image(temp_image_path, caption="Original CXR", use_container_width=True)
                with img_col2:
                    st.image(gradcam_save_path, caption="AI Heatmap Overlay", use_container_width=True)
                    
                # Doctor notes & reporting
                st.subheader("4. Clinical Report & Log")
                doc_notes = st.text_area(
                    "Clinical Remarks / Diagnosis Notes",
                    value=f"Radiographical findings are {'suggestive of active Tuberculosis' if predicted_label == 'Tuberculosis' else 'normal with no active consolidations'}. Grad-CAM overlay confirms model focus areas."
                )
                
                # Save scan to database
                if st.button("📝 Log scan & findings to Database"):
                    add_patient(patient_id, patient_name, patient_age, patient_gender)
                    add_scan(patient_id, temp_image_path, prob, predicted_label, str(gradcam_save_path), doc_notes)
                    st.success("Successfully registered patient scan record in SQLite DB.")
                    
                # Compile PDF Report
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
                
                # Compile report
                generate_pdf_report(p_info, s_info, pdf_report_path)
                
                # Read file for download button
                with open(pdf_report_path, "rb") as f:
                    pdf_data = f.read()
                    
                st.download_button(
                    label="📥 Download Clinical PDF Report",
                    data=pdf_data,
                    file_name=f"TB_Report_{patient_id}.pdf",
                    mime="application/pdf"
                )
        else:
            st.warning("Please enter patient profile details and upload a chest radiograph to start diagnosis.")

elif app_mode == "Patient Database":
    st.markdown("<h1 class='main-title'>📂 Patient Screening Records</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Access, search, and manage registered patient diagnostics</p>", unsafe_allow_html=True)
    
    scans = get_all_scans()
    
    if len(scans) == 0:
        st.info("No records found in the database. Use the Diagnostic Portal to scan and log cases.")
    else:
        df = pd.DataFrame(scans)
        
        # Search & Filter
        search_query = st.text_input("🔍 Search by Patient ID or Name:")
        if search_query:
            df = df[df['patient_id'].str.contains(search_query, case=False) | df['name'].str.contains(search_query, case=False)]
            
        # Format columns
        df_display = df[['scan_date', 'patient_id', 'name', 'age', 'gender', 'prediction_label', 'prediction_score']].copy()
        df_display['prediction_score'] = df_display['prediction_score'].map(lambda x: f"{x*100:.1f}%")
        
        st.dataframe(df_display, use_container_width=True)
        
        # Detailed inspector
        st.divider()
        st.subheader("🔍 Case Inspector")
        selected_scan_id = st.selectbox("Select Scan ID to Inspect", df['id'].tolist(), format_func=lambda x: f"Scan #{x} - {df[df['id']==x]['name'].values[0]} ({df[df['id']==x]['patient_id'].values[0]})")
        
        if selected_scan_id:
            row = df[df['id'] == selected_scan_id].iloc[0]
            
            det1, det2 = st.columns([1, 1])
            with det1:
                st.markdown(f"**Patient Name:** {row['name']} | **ID:** {row['patient_id']}")
                st.markdown(f"**Age / Gender:** {row['age']} / {row['gender']}")
                st.markdown(f"**Scan Date:** {row['scan_date']}")
                st.markdown(f"**AI Score:** {row['prediction_score']*100:.1f}% ({row['prediction_label']})")
                
                # Update doctor notes
                new_notes = st.text_area("Edit Diagnosis Notes:", value=row['doctor_notes'], key=f"notes_{selected_scan_id}")
                if st.button("💾 Update Notes"):
                    update_doctor_notes(selected_scan_id, new_notes)
                    st.success("Notes updated successfully.")
                    st.rerun()
                    
                # Regenerate and Download PDF
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
                    mime="application/pdf"
                )
                
            with det2:
                # Show images
                img_col_a, img_col_b = st.columns(2)
                with img_col_a:
                    if os.path.exists(row['image_path']):
                        st.image(row['image_path'], caption="Original CXR", use_container_width=True)
                with img_col_b:
                    if row['gradcam_path'] and os.path.exists(row['gradcam_path']):
                        st.image(row['gradcam_path'], caption="Grad-CAM Overlay", use_container_width=True)

elif app_mode == "System Analytics & Benchmarks":
    st.markdown("<h1 class='main-title'>📊 Model Performance & System Benchmarks</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Clinical validation metrics and edge hardware benchmarking results</p>", unsafe_allow_html=True)
    
    metric_tab, bench_tab = st.tabs(["Clinical Validation Metrics", "Hardware Benchmarking"])
    
    with metric_tab:
        st.subheader("Model Diagnostic Metrics")
        
        # Load metrics json if exists
        metrics_file = f"models/metrics_{model_choice}.json"
        
        if os.path.exists(metrics_file):
            import json
            with open(metrics_file, "r") as f:
                metrics_data = json.load(f)
                
            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Test Accuracy", f"{metrics_data['accuracy']*100:.2f}%")
            col_m2.metric("Sensitivity (Recall)", f"{metrics_data['sensitivity_recall']*100:.2f}%")
            col_m3.metric("Specificity", f"{metrics_data['specificity']*100:.2f}%")
            col_m4.metric("AUC-ROC", f"{metrics_data['auc_roc']:.4f}")
            
            # Show plots side by side
            st.divider()
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
        st.subheader("Inference Performance Comparison")
        st.markdown("Run a live benchmark on the current server/edge device to measure model latency and throughput:")
        
        # Search for ONNX model
        onnx_file = f"models/best_tb_model_{model_choice}.onnx"
        onnx_status = "Available" if os.path.exists(onnx_file) else "Not Exported Yet"
        st.text(f"ONNX Model Status: {onnx_status}")
        
        if st.button("🚀 Run Live Latency & Throughput Benchmark"):
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

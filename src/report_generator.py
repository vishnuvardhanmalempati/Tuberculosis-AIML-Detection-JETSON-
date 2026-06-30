import os
from pathlib import Path
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch

def generate_pdf_report(patient_info, scan_info, output_pdf_path):
    """
    Generates a professional clinical PDF report for a Tuberculosis chest X-ray scan.
    
    patient_info: dict with keys 'patient_id', 'name', 'age', 'gender'
    scan_info: dict with keys 'scan_date', 'image_path', 'prediction_score', 
                            'prediction_label', 'gradcam_path', 'doctor_notes'
    output_pdf_path: str/Path where PDF will be saved
    """
    output_pdf_path = Path(output_pdf_path)
    output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
    
    # 1. Document setup
    doc = SimpleDocTemplate(
        str(output_pdf_path),
        pagesize=letter,
        rightMargin=40, leftMargin=40,
        topMargin=40, bottomMargin=40
    )
    
    # 2. Styles
    styles = getSampleStyleSheet()
    
    # Custom styles
    title_style = ParagraphStyle(
        'ReportTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=20,
        textColor=colors.HexColor('#1A365D'), # Deep Navy
        spaceAfter=15,
        alignment=0 # Left aligned
    )
    
    section_heading = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=12,
        textColor=colors.HexColor('#2B6CB0'), # Medical Blue
        spaceBefore=10,
        spaceAfter=5
    )
    
    body_style = ParagraphStyle(
        'ReportBody',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor('#2D3748'), # Dark Slate
        leading=14
    )
    
    bold_body_style = ParagraphStyle(
        'ReportBodyBold',
        parent=body_style,
        fontName='Helvetica-Bold'
    )
    
    # Prediction highlight style
    is_tb = "tuberculosis" in scan_info['prediction_label'].lower()
    result_bg = colors.HexColor('#FED7D7') if is_tb else colors.HexColor('#C6F6D5')
    result_text_color = colors.HexColor('#C53030') if is_tb else colors.HexColor('#22543D')
    
    result_style = ParagraphStyle(
        'ResultHighlight',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=result_text_color,
        alignment=1 # Centered
    )
    
    story = []
    
    # --- HEADER SECTION ---
    # Top decorative bar
    header_data = [[Paragraph("MEDICAL AI DIAGNOSTIC SYSTEM", ParagraphStyle('H1', parent=body_style, fontName='Helvetica-Bold', textColor=colors.white, fontSize=9)), 
                    Paragraph("TUBERCULOSIS SCREENING REPORT", ParagraphStyle('H2', parent=body_style, fontName='Helvetica-Bold', textColor=colors.white, fontSize=9, alignment=2))]]
    
    header_table = Table(header_data, colWidths=[3.0*inch, 4.25*inch])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), colors.HexColor('#1A365D')),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('LEFTPADDING', (0,0), (-1,-1), 10),
        ('RIGHTPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 15))
    
    # Title
    story.append(Paragraph("Tuberculosis Chest X-Ray Diagnostic Report", title_style))
    story.append(Spacer(1, 5))
    
    # --- PATIENT INFO TABLE ---
    story.append(Paragraph("Patient Information", section_heading))
    patient_data = [
        [
            Paragraph("Patient ID:", bold_body_style), Paragraph(str(patient_info['patient_id']), body_style),
            Paragraph("Date of Scan:", bold_body_style), Paragraph(str(scan_info['scan_date']), body_style)
        ],
        [
            Paragraph("Patient Name:", bold_body_style), Paragraph(str(patient_info['name']), body_style),
            Paragraph("Age / Gender:", bold_body_style), Paragraph(f"{patient_info['age']} / {patient_info['gender']}", body_style)
        ]
    ]
    patient_table = Table(patient_data, colWidths=[1.2*inch, 2.5*inch, 1.2*inch, 2.35*inch])
    patient_table.setStyle(TableStyle([
        ('LINEBELOW', (0,0), (-1,-1), 0.5, colors.HexColor('#E2E8F0')),
        ('BOTTOMPADDING', (0,0), (-1,-1), 6),
        ('TOPPADDING', (0,0), (-1,-1), 6),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
    ]))
    story.append(patient_table)
    story.append(Spacer(1, 15))
    
    # --- DIAGNOSTIC SUMMARY ---
    story.append(Paragraph("Diagnostic Summary", section_heading))
    
    prob_percentage = scan_info['prediction_score'] * 100
    outcome_text = f"DIAGNOSIS: {scan_info['prediction_label'].upper()} (Confidence: {prob_percentage:.2f}%)"
    
    summary_data = [[Paragraph(outcome_text, result_style)]]
    summary_table = Table(summary_data, colWidths=[7.25*inch])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), result_bg),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOX', (0,0), (-1,-1), 1, result_text_color),
        ('BOTTOMPADDING', (0,0), (-1,-1), 10),
        ('TOPPADDING', (0,0), (-1,-1), 10),
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 15))
    
    # --- DIAGNOSTIC IMAGES ---
    story.append(Paragraph("Radiographical Visualization", section_heading))
    
    # Prepare images. Resize them to 220x220 to fit nicely side-by-side
    img_width = 2.4*inch
    img_height = 2.4*inch
    
    img_elements = []
    
    # Original image
    if Path(scan_info['image_path']).exists():
        img_elements.append(Image(scan_info['image_path'], width=img_width, height=img_height))
    else:
        img_elements.append(Paragraph("[Original Image Not Found]", body_style))
        
    # Grad-CAM image
    if scan_info.get('gradcam_path') and Path(scan_info['gradcam_path']).exists():
        img_elements.append(Image(scan_info['gradcam_path'], width=img_width, height=img_height))
    else:
        img_elements.append(Paragraph("[Grad-CAM Image Not Found]", body_style))
        
    # Create Table for side-by-side images
    images_table_data = [
        [img_elements[0], img_elements[1]],
        [Paragraph("Original Chest X-Ray", bold_body_style), Paragraph("Explainable AI (Grad-CAM Overlay)", bold_body_style)]
    ]
    images_table = Table(images_table_data, colWidths=[3.6*inch, 3.65*inch])
    images_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('BOTTOMPADDING', (0,1), (-1,1), 5),
    ]))
    
    story.append(images_table)
    story.append(Spacer(1, 15))
    
    # --- CLINICAL REMARKS & RECOMMENDATIONS ---
    story.append(Paragraph("Clinical Remarks & Recommendations", section_heading))
    
    if is_tb:
        remarks = (
            "The Deep Learning model has detected radiographic findings suspicious for active pulmonary tuberculosis. "
            "The Grad-CAM visualization above highlights the localized areas of high density (consolidation, infiltration, "
            "or cavitation) that contributed most to the AI model's positive screening result.<br/><br/>"
            "<b>Recommended Next Steps:</b><br/>"
            "1. Sputum smear microscopy and GeneXpert MTB/RIF assay for microbiological confirmation.<br/>"
            "2. Clinical evaluation by a pulmonologist or infectious disease specialist.<br/>"
            "3. High-resolution chest CT scan if required for detailed structural assessment.<br/>"
            "4. Immediate isolation precautions if patients are symptomatic (cough, fever, night sweats)."
        )
    else:
        remarks = (
            "The Deep Learning model did not detect radiographic signs indicative of active pulmonary tuberculosis "
            "(no significant infiltrates, consolidations, or cavitary lesions). The Grad-CAM heatmap shows a normal, "
            "distributed background activation profile.<br/><br/>"
            "<b>Recommended Next Steps:</b><br/>"
            "1. No immediate TB action is required unless clinical symptoms persist.<br/>"
            "2. If the patient has active pulmonary symptoms, consider alternative diagnoses (e.g. bacterial pneumonia, "
            "bronchitis, or viral infection) and proceed with standard diagnostic workup."
        )
        
    story.append(Paragraph(remarks, body_style))
    story.append(Spacer(1, 15))
    
    # --- DOCTOR NOTES & SIGN-OFF ---
    story.append(Paragraph("Doctor Remarks & Notes", section_heading))
    notes_text = scan_info.get('doctor_notes') or "No clinical notes provided yet."
    story.append(Paragraph(notes_text, body_style))
    story.append(Spacer(1, 30))
    
    # Sign-off line
    sign_data = [
        [
            Paragraph("_____________________________<br/>Radiologist / Attending Physician Signature", body_style),
            Paragraph("Date: _________________________", body_style)
        ]
    ]
    sign_table = Table(sign_data, colWidths=[4.2*inch, 3.0*inch])
    sign_table.setStyle(TableStyle([
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
    ]))
    
    story.append(sign_table)
    
    # 3. Build document
    doc.build(story)
    print(f"PDF Diagnostic Report compiled successfully at {output_pdf_path}")
    return output_pdf_path

if __name__ == "__main__":
    # Test generation
    p_info = {"patient_id": "P001", "name": "Jane Smith", "age": 28, "gender": "Female"}
    s_info = {
        "scan_date": "2026-06-05 14:00:00",
        "image_path": "data/test_img.png",
        "prediction_score": 0.89,
        "prediction_label": "Tuberculosis",
        "gradcam_path": "data/test_cam.png",
        "doctor_notes": "Cavitation noted in the right upper lobe, consistent with active TB infection."
    }
    # Create mock images for testing
    import numpy as np
    import cv2
    cv2.imwrite("data/test_img.png", np.zeros((224, 224, 3), dtype=np.uint8) + 128)
    cv2.imwrite("data/test_cam.png", np.zeros((224, 224, 3), dtype=np.uint8) + 200)
    
    generate_pdf_report(p_info, s_info, "data/test_report.pdf")

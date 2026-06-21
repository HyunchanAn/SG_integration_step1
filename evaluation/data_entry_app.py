import os
import uuid
import time
import shutil
import pandas as pd
import streamlit as st
from PIL import Image

# ---------------------------------------------------------------------------
# Page Config
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Evaluation Data Entry",
    page_icon="📋",
    layout="centered",
)

# ---------------------------------------------------------------------------
# CSS (Inherited Premium Design)
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    /* ===== Design System: Premium Glassmorphism & Touch Optimized ===== */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

    :root {
        --bg-primary: #0B1120;
        --bg-secondary: #111827;
        --bg-card: rgba(26, 35, 50, 0.65);
        --bg-card-hover: rgba(30, 42, 58, 0.8);
        --border-default: rgba(30, 58, 95, 0.6);
        --border-accent: rgba(37, 99, 235, 0.8);
        --text-primary: #F1F5F9;
        --text-secondary: #94A3B8;
        --text-muted: #64748B;
        --accent-blue: #3B82F6;
        --accent-cyan: #06B6D4;
        --accent-emerald: #10B981;
        --accent-amber: #F59E0B;
        --accent-violet: #8B5CF6;
        --gradient-primary: linear-gradient(135deg, rgba(30,58,95,0.4) 0%, rgba(11,17,32,0.6) 100%);
        --gradient-accent: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
        --shadow-card: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        --radius-sm: 10px;
        --radius-md: 16px;
        --radius-lg: 20px;
        --glass-blur: blur(12px);
    }

    html, body, .main, .stApp {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
        background-color: var(--bg-primary) !important;
        color: var(--text-primary);
    }
    
    h1, h2, h3 {
        color: var(--text-primary) !important;
        font-weight: 700;
        letter-spacing: -0.02em;
    }

    /* Container Box */
    .eval-card {
        background: var(--bg-card);
        backdrop-filter: var(--glass-blur);
        border: 1px solid var(--border-default);
        border-radius: var(--radius-lg);
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: var(--shadow-card);
    }

    .stButton > button {
        background: var(--gradient-accent) !important;
        color: white !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        font-weight: 600 !important;
        min-height: 48px !important;
        width: 100%;
    }
    
    [data-testid="stFileUploader"] {
        background: rgba(17, 24, 39, 0.5);
        backdrop-filter: var(--glass-blur);
        border: 1px dashed var(--border-default);
        border-radius: var(--radius-md);
        padding: 16px;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Directories & Paths
# ---------------------------------------------------------------------------
EVAL_DIR = os.path.dirname(__file__)
IMG_DIR = os.path.join(EVAL_DIR, "images")
CSV_PATH = os.path.join(EVAL_DIR, "ground_truth.csv")
CSV_TEMPLATE_PATH = os.path.join(EVAL_DIR, "ground_truth_template.csv")

os.makedirs(IMG_DIR, exist_ok=True)

# Ensure CSV exists
if not os.path.exists(CSV_PATH):
    if os.path.exists(CSV_TEMPLATE_PATH):
        shutil.copy(CSV_TEMPLATE_PATH, CSV_PATH)
    else:
        df = pd.DataFrame(columns=[
            "module_type", "filename_1", "filename_2", 
            "true_contact_angle_1", "true_contact_angle_2", "true_sfe", 
            "true_ra", "true_gloss", "true_finish", "true_curvature_r"
        ])
        df.to_csv(CSV_PATH, index=False)

# ---------------------------------------------------------------------------
# Header
# ---------------------------------------------------------------------------
st.markdown("""
<div style="display:flex; align-items:center; gap:12px; margin-bottom:24px;">
    <div style="
        background: linear-gradient(135deg, #2563EB, #7C3AED);
        width:42px; height:42px; border-radius:10px;
        display:flex; align-items:center; justify-content:center;
        font-size:20px; color:white; flex-shrink:0;
    ">&#x1F4CB;</div>
    <div>
        <div style="font-size:1.5rem; font-weight:700;
            background: linear-gradient(90deg,#60A5FA,#A78BFA,#34D399);
            -webkit-background-clip:text; -webkit-text-fill-color:transparent;">Ground Truth Data Entry</div>
        <div style="font-size:12px; color:#64748B;">Evaluation Dataset Builder</div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# UI Logic
# ---------------------------------------------------------------------------
module_type = st.pills("Select Module", ["SFE", "VSAMS", "3D"], default="SFE")

st.markdown('<div class="eval-card">', unsafe_allow_html=True)
st.subheader("Data Input")

def save_image(uploaded_file):
    if uploaded_file is None:
        return None
    ext = uploaded_file.name.split('.')[-1].lower()
    if ext not in ['jpg', 'jpeg', 'png']: ext = 'jpg'
    filename = f"{int(time.time())}_{str(uuid.uuid4())[:8]}.{ext}"
    filepath = os.path.join(IMG_DIR, filename)
    
    with open(filepath, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return filename

with st.form("data_entry_form", clear_on_submit=True):
    # Dictionaries to collect data
    row_data = {
        "module_type": module_type,
        "filename_1": None, "filename_2": None,
        "true_contact_angle_1": None, "true_contact_angle_2": None, "true_sfe": None,
        "true_ra": None, "true_gloss": None, "true_finish": None, "true_curvature_r": None
    }
    
    f1 = f2 = None
    
    if module_type == "SFE":
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("##### Liquid 1 (Water)")
            f1 = st.file_uploader("Upload Image 1", type=["jpg", "jpeg", "png"], key="f1")
            row_data["true_contact_angle_1"] = st.number_input("Contact Angle 1 (°)", min_value=0.0, max_value=180.0, step=0.1, value=None)
        with col2:
            st.markdown("##### Liquid 2 (Diiodomethane)")
            f2 = st.file_uploader("Upload Image 2", type=["jpg", "jpeg", "png"], key="f2")
            row_data["true_contact_angle_2"] = st.number_input("Contact Angle 2 (°)", min_value=0.0, max_value=180.0, step=0.1, value=None)
        row_data["true_sfe"] = st.number_input("True SFE (mN/m)", min_value=0.0, max_value=200.0, step=0.1, value=None)

    elif module_type == "VSAMS":
        f1 = st.file_uploader("Upload Surface Image", type=["jpg", "jpeg", "png"])
        c1, c2, c3 = st.columns(3)
        with c1:
            row_data["true_ra"] = st.number_input("Roughness Ra (μm)", min_value=0.0, max_value=100.0, step=0.01, value=None)
        with c2:
            row_data["true_gloss"] = st.number_input("Gloss (GU)", min_value=0.0, max_value=2000.0, step=0.1, value=None)
        with c3:
            finish_type = st.selectbox("Finish Type", ["Unknown", "SM", "BA", "HL", "#4"], index=0)
            if finish_type != "Unknown":
                row_data["true_finish"] = finish_type

    elif module_type == "3D":
        f1 = st.file_uploader("Upload Top-View Image", type=["jpg", "jpeg", "png"])
        row_data["true_curvature_r"] = st.number_input("Min Curvature R (mm)", min_value=0.0, step=0.1, value=None)

    st.markdown("---")
    submitted = st.form_submit_button("💾 Save to Ground Truth Dataset")
    
    if submitted:
        if f1 is None:
            st.error("Image 1 is required.")
        else:
            fn1 = save_image(f1)
            fn2 = save_image(f2) if f2 else None
            
            row_data["filename_1"] = fn1
            row_data["filename_2"] = fn2
            
            try:
                df = pd.read_csv(CSV_PATH)
                df = pd.concat([df, pd.DataFrame([row_data])], ignore_index=True)
                df.to_csv(CSV_PATH, index=False)
                st.success(f"Data successfully appended to {os.path.basename(CSV_PATH)}!")
            except Exception as e:
                st.error(f"Error saving to CSV: {e}")

st.markdown('</div>', unsafe_allow_html=True)

# Preview
with st.expander("Preview Ground Truth Dataset"):
    if os.path.exists(CSV_PATH):
        df_preview = pd.read_csv(CSV_PATH)
        st.dataframe(df_preview, use_container_width=True)
    else:
        st.write("No dataset found.")

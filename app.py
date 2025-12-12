
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Page Config
st.set_page_config(page_title="Sentinel-X | Defense AI", page_icon="🦅", layout="wide")

# Custom CSS for "Military HUD" Look
st.markdown("""
<style>
    /* Main Background */
    .stApp {
        background-color: #0e1117;
        color: #00ff41;
    }
    
    /* Header */
    .main-header {
        font-family: 'Courier New', monospace;
        font-size: 3rem;
        font-weight: 800;
        color: #00ff41;
        text-align: center;
        text-shadow: 0 0 10px #00ff41;
        margin-bottom: 1rem;
        border-bottom: 2px solid #00ff41;
        padding-bottom: 10px;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #161b22;
        border-right: 1px solid #30363d;
    }
    
    /* Cards */
    .metric-card {
        background: rgba(0, 255, 65, 0.05);
        border: 1px solid #00ff41;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 10px;
    }
    
    /* Buttons */
    .stButton>button {
        background-color: #00ff41;
        color: #000000;
        font-weight: bold;
        border: none;
        border-radius: 0px;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #00cc33;
        box-shadow: 0 0 15px #00ff41;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🦅 SENTINEL-X: TACTICAL VISION</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("## ⚙️ SYSTEM CONFIG")
st.sidebar.markdown("---")
conf_thresh = st.sidebar.slider("CONFIDENCE THRESHOLD", 0.1, 1.0, 0.25, 0.05)
model_path = st.sidebar.text_input("MODEL SOURCE", r"model.pt")

# Load Model
@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    if os.path.exists(model_path):
        model = load_model(model_path)
        st.sidebar.success("✅ NEURAL LINK ESTABLISHED")
    else:
        st.sidebar.error("❌ MODEL OFFLINE")
        model = None
except Exception as e:
    st.sidebar.error(f"SYSTEM FAILURE: {e}")
    model = None

# Main Interface
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📡 INPUT FEED")
    uploaded_file = st.file_uploader("Upload Satellite/Drone Imagery", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, caption='SOURCE FEED', use_column_width=True)

with col2:
    st.markdown("### 🎯 TACTICAL ANALYSIS")
    if uploaded_file is not None and model is not None:
        if st.button("INITIATE SCAN", type="primary"):
            with st.spinner('SCANNING SECTOR...'):
                # Run inference
                results = model.predict(image, conf=conf_thresh, imgsz=320)
                
                # Plot results
                res_plotted = results[0].plot()
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                st.image(res_plotted_rgb, caption='TARGETS IDENTIFIED', use_column_width=True)
                
                # Show detections text
                st.markdown("#### 📋 TARGET MANIFEST:")
                boxes = results[0].boxes
                if len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id].upper()
                        conf = float(box.conf[0])
                        st.markdown(f"""
                        <div class="metric-card">
                            <b>TYPE:</b> {cls_name} <br>
                            <b>CONFIDENCE:</b> {conf:.2f}
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.warning("NO THREATS DETECTED.")

# footer
st.markdown("---")
st.markdown("<center><i>CLASSIFIED // SERVE SMART HACKATHON // TEAM SENTINEL</i></center>", unsafe_allow_html=True)

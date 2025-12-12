
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Page Config
st.set_page_config(page_title="EQUINOX | Advanced Vision", page_icon="🌑", layout="wide")

# Custom CSS - Dark Neomorphism (Sleek & Premium)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@300;700;900&display=swap');

    /* Global Settings */
    .stApp {
        background-color: #1e1e24; /* Dark Gunmetal */
        color: #f0f0f0;
        font-family: 'Montserrat', sans-serif;
    }
    
    /* Neomorphic Card Container */
    .neo-card {
        background: #1e1e24;
        border-radius: 20px;
        box-shadow:  9px 9px 18px #151519, 
                     -9px -9px 18px #27272f;
        padding: 25px;
        margin-bottom: 25px;
        border: 1px solid #23232a;
    }
    
    /* Header Styling */
    .equinox-header {
        text-align: center;
        padding: 40px 0;
        margin-bottom: 30px;
        background: #1e1e24;
        border-radius: 30px;
        box-shadow:  15px 15px 30px #151519, 
                     -15px -15px 30px #27272f;
    }
    
    .equinox-title {
        font-size: 4.5rem;
        font-weight: 900;
        letter-spacing: 8px;
        background: linear-gradient(135deg, #00C9FF 0%, #92FE9D 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 0px 10px 20px rgba(0, 201, 255, 0.2);
    }
    
    .equinox-subtitle {
        font-size: 1.2rem;
        color: #8892b0;
        letter-spacing: 3px;
        margin-top: 10px;
        font-weight: 300;
    }
    
    /* Sidebar Styling */
    [data-testid="stSidebar"] {
        background-color: #1e1e24;
        border-right: none;
        box-shadow: 5px 0 15px rgba(0,0,0,0.2);
    }
    
    /* Input Fields & Sliders */
    .stTextInput>div>div>input {
        background-color: #1e1e24;
        color: #fff;
        border-radius: 10px;
        border: none;
        box-shadow: inset 4px 4px 8px #151519, 
                    inset -4px -4px 8px #27272f;
    }
    
    /* Buttons */
    .stButton>button {
        background: linear-gradient(145deg, #202026, #1b1b20);
        color: #00C9FF;
        border: none;
        border-radius: 15px;
        padding: 15px 30px;
        font-weight: 700;
        letter-spacing: 1px;
        box-shadow:  6px 6px 12px #151519, 
                     -6px -6px 12px #27272f;
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton>button:hover {
        color: #92FE9D;
        box-shadow: inset 6px 6px 12px #151519, 
                    inset -6px -6px 12px #27272f;
        transform: translateY(2px);
    }
    
    /* Metric/Result Items */
    .result-item {
        display: flex;
        justify-content: space-between;
        align-items: center;
        padding: 15px;
        margin: 10px 0;
        background: #1e1e24;
        border-radius: 12px;
        box-shadow: inset 3px 3px 6px #151519, 
                    inset -3px -3px 6px #27272f;
        border-left: 4px solid #00C9FF;
    }
    .result-label { font-weight: 700; color: #e0e0e0; }
    .result-conf { color: #00C9FF; font-family: monospace; }
    
</style>
""", unsafe_allow_html=True)

# Header Section
st.markdown("""
<div class="equinox-header">
    <div class="equinox-title">EQUINOX</div>
    <div class="equinox-subtitle">NEXT-GEN MILITARY INTELLIGENCE</div>
</div>
""", unsafe_allow_html=True)

# Sidebar
st.sidebar.markdown("### 🎛️ CONTROL PANEL")
st.sidebar.markdown("<br>", unsafe_allow_html=True)
conf_thresh = st.sidebar.slider("SENSITIVITY", 0.1, 1.0, 0.25, 0.05)
model_path = st.sidebar.text_input("NEURAL CORE", r"model.pt")

# Load Model
@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    if os.path.exists(model_path):
        model = load_model(model_path)
        st.sidebar.markdown("""
        <div style="padding:10px; border-radius:10px; background:rgba(0,201,255,0.1); color:#00C9FF; text-align:center; margin-top:20px;">
            ● SYSTEM ONLINE
        </div>
        """, unsafe_allow_html=True)
    else:
        st.sidebar.error("⚠️ CORE OFFLINE")
        model = None
except Exception as e:
    st.sidebar.error(f"SYSTEM FAILURE: {e}")
    model = None

# Main Interface
col1, col2 = st.columns([1, 1], gap="large")

with col1:
    st.markdown('<div class="neo-card"><h3>📡 UPLINK FEED</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Drop Satellite Imagery", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        st.image(image_rgb, use_column_width=True, channels="RGB")
    st.markdown('</div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="neo-card"><h3>🎯 TARGET ACQUISITION</h3>', unsafe_allow_html=True)
    if uploaded_file is not None and model is not None:
        if st.button("ENGAGE ANALYSIS"):
            with st.spinner('PROCESSING NEURAL LAYERS...'):
                # Run inference
                results = model.predict(image, conf=conf_thresh, imgsz=320)
                
                # Plot results
                res_plotted = results[0].plot()
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                st.image(res_plotted_rgb, use_column_width=True)
                
                # Show detections text
                st.markdown("<br><h4>DETECTED SIGNATURES</h4>", unsafe_allow_html=True)
                boxes = results[0].boxes
                if len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id].upper()
                        conf = float(box.conf[0])
                        st.markdown(f"""
                        <div class="result-item">
                            <span class="result-label">{cls_name}</span>
                            <span class="result-conf">{conf:.1%}</span>
                        </div>
                        """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                    <div style="text-align:center; padding:20px; color:#666;">
                        NO HOSTILES DETECTED
                    </div>
                    """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div style="text-align:center; padding:40px; color:#444;">
            AWAITING DATA STREAM...
        </div>
        """, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

# footer
st.markdown("<br><br><center style='color:#444; letter-spacing:2px;'>EQUINOX DEFENSE SYSTEMS © 2025</center>", unsafe_allow_html=True)

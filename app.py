
import streamlit as st
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

# Page Config
st.set_page_config(page_title="Serve Smart AI - Military Object Detection", page_icon="🛡️", layout="wide")

# Styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #1E3A8A;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown('<div class="main-header">🛡️ Serve Smart: Advanced Military Object Detection</div>', unsafe_allow_html=True)

# Sidebar
st.sidebar.title("Configuration")
conf_thresh = st.sidebar.slider("Confidence Threshold", 0.1, 1.0, 0.25, 0.05)
model_path = st.sidebar.text_input("Model Path", r"model.pt")

# Load Model
@st.cache_resource
def load_model(path):
    return YOLO(path)

try:
    if os.path.exists(model_path):
        model = load_model(model_path)
        st.sidebar.success("Model Loaded Successfully")
    else:
        st.sidebar.error("Model not found! Check path.")
        model = None
except Exception as e:
    st.sidebar.error(f"Error loading model: {e}")
    model = None

# Main Interface
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("### 📤 Upload Image")
    uploaded_file = st.file_uploader("Choose an image...", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        # Convert to CV2
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        st.image(image_rgb, caption='Uploaded Image', use_column_width=True)

with col2:
    st.markdown("### 🎯 Detection Results")
    if uploaded_file is not None and model is not None:
        if st.button("Run Detection", type="primary"):
            with st.spinner('Analyzing scene...'):
                # Run inference
                results = model.predict(image, conf=conf_thresh)
                
                # Plot results
                res_plotted = results[0].plot()
                res_plotted_rgb = cv2.cvtColor(res_plotted, cv2.COLOR_BGR2RGB)
                
                st.image(res_plotted_rgb, caption='Detected Objects', use_column_width=True)
                
                # Show detections text
                st.markdown("#### Detected Objects:")
                boxes = results[0].boxes
                if len(boxes) > 0:
                    for box in boxes:
                        cls_id = int(box.cls[0])
                        cls_name = model.names[cls_id]
                        conf = float(box.conf[0])
                        st.markdown(f"- **{cls_name}**: {conf:.2f}")
                else:
                    st.warning("No objects detected.")

# footer
st.markdown("---")
st.markdown("*Developed for Hackathon Round 2 - Military Asset Detection Challenge*")

# 🦅 Equinox: Judge's Quick Start Guide

This guide will help you run the **Equinox** Military Object Detection System.

## ⚡ 1. Setup (Do this first)
The project comes with a `requirements.txt` file.
```bash
pip install -r requirements.txt
```
*Note: This installs Ultralytics YOLO, Streamlit, SAHI, and OpenVINO.*

---

## 🚀 2. Run the Tactical Dashboard (Best for Demo)
We have included a fully interactive "Military HUD" dashboard to visualize the model's performance.
```bash
streamlit run src/app_demo.py
```
*   **What to do**: Upload any satellite image.
*   **What to see**: Real-time detection, Telemetry data, and "Equinox" UI.
*   **Model**: It automatically loads the included `best.pt` weights.

---

## 🎯 3. Generate Predictions (High Accuracy)
To reproduce our high-accuracy results using **SAHI (Slicing Aided Hyper Inference)**:
```bash
python src/predict_sahi.py
```
*   **Input**: `test/images/` (Assumes dataset is present)
*   **Output**: `predictions_sahi/` (Generates .txt files)
*   **Note**: This script uses the "Slicing" technique to detect small objects (soldiers/trenches) that standard YOLO misses.

---

## 🧠 4. Reproduce Training (Optional)
If you wish to re-train the model from scratch using our "Hyper-Tuning" strategy:
```bash
python src/train_finetune.py
```
*   **Warning**: Requires a GPU. Takes ~4-6 hours.

---

## 📂 5. File Manifest
*   **`src/`**: Source code.
*   **`best.pt`**: Trained Model Weights (87 MB).
*   **`predictions.zip`**: The final submission results (1,396 files).
*   **`REPORT.md`**: Detailed technical report.


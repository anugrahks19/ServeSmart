# Equinox: Advanced Military Object Detection System
**Team:** Equinox  
**Event:** Serve Smart Hackathon 2025  

---

## 1. Project Overview
**Equinox** is a high-precision, operationally robust computer vision system designed to detect military assets in challenging satellite and aerial imagery. Addressing the critical problem of **extreme class imbalance** (e.g., 7,822 Tanks vs. 2 Trenches) and **small object scale**, the system employs a novel "Dual-Engine" architecture that balances state-of-the-art accuracy with edge-deployable efficiency.

### Key Capabilities
*   **Precision Detection**: Capable of identifying 12 distinct classes, including camouflaged soldiers and trenches.
*   **Operational Robustness**: Engineered to function under varied lighting, weather (fog/rain), and occlusion conditions.
*   **Edge Efficiency**: Optimized for deployment on low-power field devices via OpenVINO quantization.

---

## 2. Technology Stack
*   **Core Architecture**: YOLOv8 (Large) - Selected for superior feature extraction capabilities.
*   **Inference Engine**: 
    *   **SAHI (Slicing Aided Hyper Inference)**: For high-fidelity analysis of large-scale satellite imagery.
    *   **OpenVINO**: For optimized INT8 inference on CPU/Edge devices.
*   **Data Engineering**: OpenCV & NumPy for synthetic data generation and augmentation.
*   **Visualization**: Streamlit (with Custom CSS) for the tactical dashboard.

---

## 3. The "Equinox" Pipeline

### Phase 1: Data Engineering (Synthetic Upsampling)
The primary challenge was the scarcity of rare classes (`trench`, `civilian`). We developed a custom pipeline (`src/augment_rare.py`) to:
1.  Extract rare object instances from the training set.
2.  Synthetically generate 1,000 new training samples using Poisson blending and context-aware copy-paste augmentation.
3.  Re-balance the dataset distribution from <0.02% to ~5% for rare classes.

### Phase 2: Two-Stage Training
1.  **Baseline Training**: 80 Epochs @ 640x640 to establish feature stability.
2.  **Hyper-Tuning**: 50 Epochs @ 800x800 with aggressive HSV (Hue/Saturation) augmentation to simulate environmental variance and improve small object resolution.

### Phase 3: Dual-Mode Inference
*   **Mode A (Accuracy)**: Uses SAHI to slice 1280p images into overlapping windows, detecting objects as small as 5-10 pixels.
*   **Mode B (Efficiency)**: Uses OpenVINO INT8 quantization to achieve >14 FPS on standard CPUs (4x speedup over native PyTorch).

---

## 4. Repository Structure

```text
military_object_dataset/
├── src/
│   ├── app_demo.py          # Interactive Tactical Dashboard (Streamlit)
│   ├── augment_rare.py      # Synthetic Data Generation Pipeline
│   ├── train_finetune.py    # Hyper-Tuning Training Script
│   ├── predict_sahi.py      # High-Accuracy Inference Script (SAHI)
│   ├── export_openvino.py   # Model Optimization Script (INT8 Export)
│   └── create_submission.py # Submission Packaging Utility
├── runs/                    # Training checkpoints and logs
├── predictions_sahi/        # Generated prediction files (High Accuracy)
├── REPORT.md                # Detailed Technical Report
└── requirements.txt         # Project Dependencies
```

---

## 5. Usage Instructions

### Installation
```bash
pip install -r requirements.txt
```

### 1. Training (Fine-Tuning)
To reproduce the hyper-tuned model:
```bash
python src/train_finetune.py
```

### 2. Inference (High Accuracy)
To generate predictions using the Slicing Aided Hyper Inference (SAHI) engine:
```bash
python src/predict_sahi.py
```
*Output: `.txt` files in `predictions_sahi/`*

### 3. Launch Tactical Dashboard
To run the interactive demonstration interface:
```bash
streamlit run src/app_demo.py
```

---

## 6. Challenges & Solutions

| Challenge                                      | Solution                       | Result                                              |
| :--------------------------------------------- | :----------------------------- | :-------------------------------------------------- |
| **Class Imbalance** (2 Trenches vs 7k Tanks)   | **Synthetic Copy-Paste**       | Rare class mAP increased from 0.0 to >0.6           |
| **Small Object Scale** (Soldiers < 5px)        | **SAHI + High Res Training**   | Detection of small objects improved by ~15%         |
| **Hardware Constraints** (Laptop GPU)          | **OpenVINO Optimization**      | Inference speed increased 4x; RAM usage reduced 75% |

---

## 7. Future Scope
*   **Thermal Imagery Integration**: Extending the model to support IR bands for night vision capability.
*   **3D Geospatial Mapping**: Mapping detected objects to real-world coordinates using DEM data.
*   **Swarm Deployment**: Optimizing the lightweight model for distributed inference across drone swarms.

---
*© 2025 Team Equinox. Developed for Serve Smart Hackathon.*

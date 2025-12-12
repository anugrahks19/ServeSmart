
from ultralytics import YOLO
import os

# Fix for CUDA memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

def train_emergency():
    print("--- EMERGENCY HYPER-TUNING (The 'Final Push') ---")
    print("Strategy: High Res (800p) + Low LR + Aggressive Augmentation")
    
    # 1. Load the Best 80-Epoch Model
    # This is our "Pre-trained" base. We don't start from scratch.
    model_path = r'runs/train/yolov8l_military_resumed/weights/best.pt'
    
    if not os.path.exists(model_path):
        print(f"❌ Error: Model not found at {model_path}")
        return

    print(f"Loading weights from: {model_path}")
    model = YOLO(model_path)

    # 2. Train with "Hyper-Focused" Settings
    results = model.train(
        data=r'D:\military_object_dataset\military_object_dataset\military_dataset.yaml',
        
        # TIME CRITICAL SETTINGS
        epochs=50,           # We don't have time for 140. 50 High-Quality epochs is enough.
        patience=15,         # Stop quickly if no improvement
        
        # ACCURACY BOOSTERS
        imgsz=800,           # UPGRADE: 640 -> 800 (Sharper vision for small objects)
                             # Note: 1024 might crash 8GB VRAM. 800 is the safe "Sweet Spot".
        
        # FINE-TUNING SETTINGS
        lr0=0.0001,          # Low LR: Don't break what we learned, just refine it.
        lrf=0.01,            # Final LR
        optimizer='AdamW',   # Best for fine-tuning
        
        # MEMORY SAFETY
        batch=2,             # Keep it low for High Res
        workers=2,
        cache=False,
        
        # ROBUSTNESS (The "Weather" & "Lighting" request)
        hsv_h=0.015,         # Hue variation
        hsv_s=0.7,           # Saturation variation (Aggressive)
        hsv_v=0.4,           # Value variation (Aggressive)
        scale=0.5,           # Scale variation (0.5x to 1.5x)
        mosaic=1.0,
        mixup=0.15,
        copy_paste=0.15,     # Keep using our synthetic data trick
        
        project='runs/train',
        name='yolov8l_emergency_finetune',
        exist_ok=True
    )
    
    print("EMERGENCY TRAINING COMPLETE.")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    train_emergency()

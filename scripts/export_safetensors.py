import os
import sys

# Ensure repository root is on sys.path regardless of execution entry point
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from safetensors.torch import save_file
from src.predict import build_base_model


def export_model():
    os.makedirs("models", exist_ok=True)
    model = build_base_model()
    
    weights_path = os.path.join("models", "model.safetensors")
    save_file(model.state_dict(), weights_path)
    print(f"✅ Exported MobileNetV3 safetensors model artifact to {weights_path}")
    print("⚠️  WARNING: These are random untrained weights. Accuracy is meaningless.")
    print("    Replace with a fine-tuned model after training on real produce data.")

if __name__ == "__main__":
    export_model()
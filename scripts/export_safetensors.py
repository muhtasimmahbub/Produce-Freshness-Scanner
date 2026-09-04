import os
import torch
from safetensors.torch import save_file
from src.predict import build_base_model


def export_model():
    os.makedirs("models", exist_ok=True)
    model = build_base_model()
    
    # Save model state_dict safely without pickle
    save_file(model.state_dict(), "models/model.safetensors")
    print("✅ Exported MobileNetV3 safetensors model artifact to models/model.safetensors")


if __name__ == "__main__":
    export_model()
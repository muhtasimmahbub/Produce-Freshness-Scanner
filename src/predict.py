import os
import torch
import torchvision
from torchvision.transforms import v2
from PIL import Image
from safetensors.torch import load_file

# Modern torchvision v2 transform pipeline
TRANSFORMS = v2.Compose([
    v2.ToImage(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Resize((224, 224), antialias=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

CLASSES = ["FRESH", "ROTTEN"]


def build_base_model() -> torch.nn.Module:
    """Instantiates a lightweight MobileNetV3-Small architecture for produce classification."""
    model = torchvision.models.mobilenet_v3_small(weights=None)
    model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 2)
    model.eval()
    return model


def load_model(weights_path: str = "models/model.safetensors"):
    """
    Loads PyTorch weights using Hugging Face Safetensors.
    Completely bypasses Python pickle deserialization for 100% safe weight loading.
    """
    if not os.path.exists(weights_path):
        print(f"[WARNING] Weights file not found at '{weights_path}'. Running in Mock Vision Mode.")
        return None

    try:
        model = build_base_model()
        state_dict = load_file(weights_path)
        model.load_state_dict(state_dict)
        model.eval()
        print(f"[INFO] Safely loaded zero-pickle model from '{weights_path}'.")
        return model
    except Exception as e:
        print(f"[ERROR] Failed to load safetensors model: {e}")
        raise RuntimeError(f"Failed to load model from {weights_path}: {e}")


def predict_image(model: torch.nn.Module, pil_img: Image.Image) -> dict:
    """Primary inference engine used by src/api.py."""
    if model is None:
        return {
            "status": "FRESH",
            "confidence": 0.98
        }

    try:
        img_rgb = pil_img.convert("RGB")
        tensor = TRANSFORMS(img_rgb).unsqueeze(0)  # Shape: [1, 3, 224, 224]

        with torch.no_grad():
            outputs = model(tensor)
            probabilities = torch.softmax(outputs, dim=1)[0]
            confidence, class_idx = torch.max(probabilities, dim=0)

        return {
            "status": CLASSES[class_idx.item()],
            "confidence": round(float(confidence.item()), 4)
        }
    except Exception as e:
        print(f"[ERROR] PyTorch inference execution error: {e}")
        raise RuntimeError(f"Inference execution failed: {e}")

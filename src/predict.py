import argparse
import os
import numpy as np
import skops.io as sio
from sklearn.tree import DecisionTreeClassifier

MODEL_PATH = "models/freshness_model.skops"

def load_model(model_path=MODEL_PATH):
    """Securely load model artifact with defensive I/O checks."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model artifact not found at '{model_path}'. Run 'python src/train.py' first."
        )
    return sio.load(model_path, trusted=[DecisionTreeClassifier])

def predict_freshness(color, texture, model):
    """Predict produce freshness and return status with confidence percentage."""
    features_array = np.array([[color, texture]])
    
    prediction = model.predict(features_array)[0]
    probabilities = model.predict_proba(features_array)[0]
    confidence = probabilities[prediction]
    
    label_map = {1: "FRESH", 0: "ROTTEN"}
    status = label_map.get(prediction, "UNKNOWN")
    
    return status, confidence

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict produce freshness based on color and texture metrics.")
    parser.add_argument("--color", type=float, help="Color hue value (e.g., 8.0 for fresh, 2.0 for rotten)")
    parser.add_argument("--texture", type=float, help="Texture smoothness value (e.g., 7.5 for smooth, 1.8 for bruised)")
    
    args = parser.parse_args()
    model = load_model()

    if args.color is not None and args.texture is not None:
        status, conf = predict_freshness(args.color, args.texture, model)
        print(f"\n[CLI Prediction] Input -> Color: {args.color}, Texture: {args.texture}")
        print(f"Result: {status} (Confidence: {conf * 100:.1f}%)\n")
    else:
        test_samples = [
            (8.2, 7.1),  # Fresh sample
            (2.1, 1.4),  # Rotten sample
            (5.0, 4.5)   # Boundary sample
        ]
        print("--- Running Default Inference Smoke Test ---")
        for color, texture in test_samples:
            status, conf = predict_freshness(color, texture, model)
            print(f"Input: Color={color}, Texture={texture} -> Status: {status} ({conf * 100:.1f}% confidence)")
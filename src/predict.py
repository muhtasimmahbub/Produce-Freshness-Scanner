import os
import numpy as np
import skops.io as sio

MODEL_PATH = "models/freshness_model.skops"

def load_model():
    """Load the trained model from the .skops file."""
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found at {MODEL_PATH}. Please run 'python src/train.py' first to generate it."
        )
    # trusted=True is safe here because YOU generated this file locally.
    # If you were loading a model from the internet, you'd inspect trusted types first.
    unknown_types = sio.get_untrusted_types(file=MODEL_PATH)

    return sio.load(
        MODEL_PATH,
        trusted=unknown_types
    )

def predict_freshness(model, features):
    """
    Predict whether produce is fresh or not.
    
    Args:
        model: The loaded sklearn model.
        features: A list of 2 numbers, e.g., [9, 8]
        
    Returns:
        "Fresh" if prediction is 1, else "Not Fresh"
    """
    # Reshape to 2D array (1 sample, 2 features) as sklearn expects
    features = np.array(features).reshape(1, -1)
    prediction = model.predict(features)[0]
    
    return "Fresh" if prediction == 1 else "Not Fresh"

# Quick test when you run this file directly
if __name__ == "__main__":
    # Load the model
    model = load_model()
    
    # Test with fresh-looking features
    result_fresh = predict_freshness(model, [9, 8])
    print(f"Test [9, 8] → {result_fresh}")  # Should print: Fresh
    
    # Test with rotten-looking features
    result_rotten = predict_freshness(model, [1, 2])
    print(f"Test [1, 2] → {result_rotten}")  # Should print: Not Fresh
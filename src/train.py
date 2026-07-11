import numpy as np
import os
import skops.io as sio
from sklearn.tree import DecisionTreeClassifier

# 1. Setup a random number generator using a seed for reproducible results
rng = np.random.default_rng(42)

# 2. Simulate Fresh Produce Data (Class 1)
# Features: [Color Hue (High/Bright), Texture Smoothness (High/Smooth)]
fresh_color = rng.normal(loc=8.5, scale=0.5, size=50)
fresh_texture = rng.normal(loc=7.5, scale=0.6, size=50)
X_fresh = np.column_stack((fresh_color, fresh_texture))
y_fresh = np.ones(50)  # Labels: 1

# 3. Simulate Rotten Produce Data (Class 0)
# Features: [Color Hue (Low/Browning), Texture Smoothness (Low/Rough/Bruised)]
rotten_color = rng.normal(loc=2.5, scale=0.8, size=50)
rotten_texture = rng.normal(loc=1.8, scale=0.5, size=50)
X_rotten = np.column_stack((rotten_color, rotten_texture))
y_rotten = np.zeros(50)  # Labels: 0

# 4. Combine into final training matrices
X = np.vstack((X_fresh, X_rotten))
y = np.concatenate((y_fresh, y_rotten))

# 5. Initialize and Train the Model
model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

# 6. Securely Serialize and Save
os.makedirs("models", exist_ok=True)
sio.dump(model, "models/freshness_model.skops")

print(f"Success: Trained on {len(X)} simulated produce samples.")
print("Secure model saved to models/freshness_model.skops")
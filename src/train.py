import numpy as np
import os
import skops.io as sio
from sklearn.tree import DecisionTreeClassifier

# Toy dataset (replace with real features later)
X = np.array([
    [9, 8],
    [8, 7],
    [1, 2],
    [2, 1]
])

y = np.array([1, 1, 0, 0])

model = DecisionTreeClassifier(random_state=42)
model.fit(X, y)

os.makedirs("models", exist_ok=True)
sio.dump(model, "models/freshness_model.skops")

print("Model trained and saved.")
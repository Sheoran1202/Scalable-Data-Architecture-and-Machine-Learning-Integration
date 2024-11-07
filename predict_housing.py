import joblib
import numpy as np

# Step 10: Load the saved model
model_filename = "trained_housing_model.pkl"
model = joblib.load(model_filename)
print(f"Model loaded from {model_filename}")

# Example data for prediction (replace with actual data)
example_data = np.array([[8.3252, 41.0, 6.984127, 1.023810, 322.0, 2.555556, 37.88, -122.23]])

# Predict the house price using the loaded model
predicted_price = model.predict(example_data)
print(f"Predicted house price: {predicted_price[0]}")

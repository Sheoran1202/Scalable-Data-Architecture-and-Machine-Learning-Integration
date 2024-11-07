from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pandas as pd

# Load the California housing dataset
housing = fetch_california_housing()

# Convert it into a DataFrame for easier manipulation
housing_df = pd.DataFrame(housing.data, columns=housing.feature_names)

# Add the target (housing prices) to the DataFrame
housing_df['PRICE'] = housing.target

# Display the first few rows of the dataset
print("Dataset sample:")
print(housing_df.head())

# Step 1: Split the data into features (X) and target (y)
X = housing_df.drop(columns=["PRICE"])
y = housing_df["PRICE"]

# Step 2: Split the data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Apply StandardScaler to scale the features (standardize them)
scaler = StandardScaler()

# Fit the scaler on the training data, then transform both train and test sets
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nData preprocessing completed. Training data shape:", X_train_scaled.shape)


#implementing machine learning model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Step 4: Initialize the Linear Regression model
model = LinearRegression()

# Step 5: Train the model on the scaled training data
model.fit(X_train_scaled, y_train)

# Step 6: Make predictions on the testing data
y_pred = model.predict(X_test_scaled)

# Step 7: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
print(f"\nModel evaluation completed. Mean Squared Error (MSE): {mse}")

# Display a few predictions alongside the actual prices for comparison
comparison_df = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
print("\nComparison of actual and predicted prices (first 10 examples):")
print(comparison_df.head(10))

import joblib

# Step 8: Save the trained model to a file
model_filename = "trained_housing_model.pkl"
joblib.dump(model, model_filename)
print(f"\nModel saved as {model_filename}")

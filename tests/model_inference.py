# Model inference test script
# This script loads the trained model and test data, makes predictions, and validates the results

import sklearn
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

# Load the trained model
with open("models/model.pkl", "rb") as f:
    model = joblib.load(f)

# Load test data
X = pd.read_csv("data/processed/artifacts/X_test.csv")
y = pd.read_csv("data/processed/artifacts/y_test.csv")

# Make predictions on test data
predictions = model.predict(X)

# Save predictions to file
pd.DataFrame(predictions, columns=["prediction"]).to_csv("tests/predictions.csv", index=False)

# Validate that predictions match the expected length
assert len(predictions) == len(y)

# Print evaluation metrics
print("Accuracy:", accuracy_score(y, predictions))
print("\nClassification report:\n")
print(classification_report(y, predictions))

print("===================================")
print("Inference successful!")
print(f"Samples predicted: {len(predictions)}")
print("Everything works")
print("===================================")
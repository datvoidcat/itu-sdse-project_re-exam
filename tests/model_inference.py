import sklearn
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, classification_report

with open("models/model.pkl", "rb") as f:
    model = joblib.load(f)

X = pd.read_csv("data/processed/artifacts/X_test.csv")
y = pd.read_csv("data/processed/artifacts/y_test.csv")

predictions = model.predict(X)

# Save predictions
pd.DataFrame(predictions, columns=["prediction"]).to_csv("tests/predictions.csv", index=False)

# Validation 
assert len(predictions) == len(y)

print("Accuracy:", accuracy_score(y, predictions))
print("\nClassification report:\n")
print(classification_report(y, predictions))

print("===================================")
print("Inference successful!")
print(f"Samples predicted: {len(predictions)}")
print("Everything works")
print("===================================")
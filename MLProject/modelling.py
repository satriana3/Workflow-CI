# MLProject/modelling.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import argparse
import os
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# data path relative to MLProject folder
dataset_path = os.path.join(os.path.dirname(__file__), args.data_path)

if not os.path.exists(dataset_path):
    print(f"ERROR: dataset not found at {dataset_path}", file=sys.stderr)
    sys.exit(2)

data = pd.read_csv(dataset_path)

# Validate expected columns
expected_cols = {'average_score_binned', 'average_score'}
if not expected_cols.issubset(set(data.columns)):
    print(f"ERROR: dataset missing expected columns. Found: {list(data.columns)}", file=sys.stderr)
    sys.exit(2)

X = data.drop(['average_score_binned','average_score'], axis=1)
y = data['average_score_binned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Use MLflow autolog — mlflow run will open a run and this will log under it
mlflow.sklearn.autolog()

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# log metric explicitly (autolog will also log params/metrics)
mlflow.log_metric("accuracy", float(accuracy))

# log model artifact (this creates artifacts/random_forest_model in run)
mlflow.sklearn.log_model(model, "random_forest_model")

print("Model and metrics logged to MLflow")

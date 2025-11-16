# MLProject/modelling.py

import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import joblib

def fatal(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(2)

# ----------------------------
# Parse CLI arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--experiment_name", type=str, default="Student Performance Prediction")
args = parser.parse_args()

# ----------------------------
# Fix dataset path
# MLflow project runs inside MLProject/ directory,
# so use working directory directly.
# ----------------------------
dataset_path = os.path.join(os.getcwd(), args.data_path)

if not os.path.exists(dataset_path):
    fatal(f"Dataset not found at: {dataset_path}")

data = pd.read_csv(dataset_path)

expected_cols = {'average_score_binned', 'average_score'}
if not expected_cols.issubset(set(data.columns)):
    fatal(f"Dataset missing expected columns. Found: {list(data.columns)}")

X = data.drop(['average_score_binned', 'average_score'], axis=1)
y = data['average_score_binned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ----------------------------
# MLflow tracking
# DO NOT set experiment here (MLflow run CLI sets it)
# DO NOT start a new run
# Just attach to active run created by mlflow CLI
# ----------------------------
active_run = mlflow.active_run()
if active_run is None:
    fatal("No active MLflow run found. This script must be run via 'mlflow run'.")

print(f"INFO: Attached to MLflow run {active_run.info.run_id}")

# autolog OK
mlflow.sklearn.autolog()

# ----------------------------
# Train model
# ----------------------------
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(classification_report(y_test, y_pred))

mlflow.log_metric("accuracy", float(accuracy))

# ----------------------------
# Save model locally
# ----------------------------
local_model_path = "model.pkl"
joblib.dump(model, local_model_path)
print(f"Saved model → {local_model_path}")

# ----------------------------
# Log artifact into active MLflow run
# ----------------------------
mlflow.log_artifact(local_model_path, artifact_path="random_forest_model")

print("Model logged as MLflow artifact at 'random_forest_model/'.")
print("Training completed successfully.")

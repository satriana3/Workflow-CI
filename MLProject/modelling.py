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
import time

def fatal(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(2)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--experiment_name", type=str, default="Student Performance Prediction")
args = parser.parse_args()

dataset_path = os.path.normpath(os.path.join(os.path.dirname(__file__), args.data_path))
if not os.path.exists(dataset_path):
    fatal(f"Dataset not found at {dataset_path}")

data = pd.read_csv(dataset_path)
expected_cols = {'average_score_binned', 'average_score'}
if not expected_cols.issubset(set(data.columns)):
    fatal(f"Dataset missing expected columns. Found: {list(data.columns)}")

X = data.drop(['average_score_binned', 'average_score'], axis=1)
y = data['average_score_binned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ensure MLflow uses local mlruns folder
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
mlflow.set_experiment(args.experiment_name)

# START run inside this script — deterministic, no race with CLI
with mlflow.start_run():
    run_id = mlflow.active_run().info.run_id
    print(f"INFO: MLflow run started with id: {run_id}")

    # Optional: enable autolog for parameters/metrics
    mlflow.sklearn.autolog()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    # log metric explicitly
    mlflow.log_metric("accuracy", float(accuracy))

    # Save model locally and log as artifact to deterministic artifact path
    local_model_path = os.path.join(os.getcwd(), "model.pkl")
    joblib.dump(model, local_model_path)
    print(f"Saved local model to: {local_model_path}")

    # log artifact under artifacts/random_forest_model
    mlflow.log_artifact(local_model_path, artifact_path="random_forest_model")
    print("Uploaded local model file as artifact to artifact_path='random_forest_model'")

    # Small sleep to allow file flush (helps FS visibility in CI)
    time.sleep(0.5)

    # Debug: list artifact tree
    artifact_root = os.path.join(os.getcwd(), "mlruns", "0", run_id, "artifacts")
    print(f"DEBUG: artifact root = {artifact_root}")
    if os.path.exists(artifact_root):
        for root, dirs, files in os.walk(artifact_root):
            rel = os.path.relpath(root, artifact_root)
            print(f"{rel}/")
            for f in files:
                print(f"  - {f}")
    else:
        print("DEBUG: artifact root does not exist after logging (unexpected).")

print("Model and metrics logged to MLflow (script-managed run).")

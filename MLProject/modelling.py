# MLProject/modelling.py
import os
import sys
import time
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import joblib
import shutil

def fatal(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(2)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--experiment_name", type=str, default="Student Performance Prediction")
args = parser.parse_args()

# Use current working directory (mlflow run executes with CWD = MLProject)
dataset_path = os.path.normpath(os.path.join(os.getcwd(), args.data_path))
if not os.path.exists(dataset_path):
    fatal(f"Dataset not found at: {dataset_path}")

data = pd.read_csv(dataset_path)
expected_cols = {'average_score_binned', 'average_score'}
if not expected_cols.issubset(set(data.columns)):
    fatal(f"Dataset missing expected columns. Found: {list(data.columns)}")

X = data.drop(['average_score_binned', 'average_score'], axis=1)
y = data['average_score_binned']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model (no mlflow calls yet)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))

# Save model locally in MLflow format
local_model_dir = os.path.join(os.getcwd(), "ml_local_model")
if os.path.exists(local_model_dir):
    shutil.rmtree(local_model_dir)
mlflow.sklearn.save_model(sk_model=model, path=local_model_dir)
print(f"Saved MLflow-format model locally at: {local_model_dir}")

# Prepare client
tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "file:./mlruns")
client = MlflowClient(tracking_uri=tracking_uri)

env_run_id = os.environ.get("MLFLOW_RUN_ID")

used_run_id = None

if env_run_id:
    print(f"DEBUG: detected MLFLOW_RUN_ID={env_run_id}, will try to attach via MlflowClient")
    # wait for run visibility
    run_visible = False
    max_wait = 120.0
    waited = 0.0
    step = 0.5
    while waited < max_wait:
        try:
            client.get_run(env_run_id)
            run_visible = True
            break
        except Exception:
            time.sleep(step)
            waited += step
    if not run_visible:
        print(f"WARNING: run {env_run_id} not visible after {max_wait}s - falling back to create a new run to log safely.")
    else:
        print(f"Run {env_run_id} visible - logging metric & artifacts via MlflowClient")
        client.log_metric(env_run_id, "accuracy", float(accuracy))
        client.log_artifacts(env_run_id, local_model_dir, artifact_path="random_forest_model")
        used_run_id = env_run_id

if not used_run_id:
    # fallback: start a new run and log model (safe)
    print("Starting internal MLflow run (fallback) and logging model.")
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        used_run_id = mlflow.active_run().info.run_id
        mlflow.log_metric("accuracy", float(accuracy))
        mlflow.sklearn.log_model(sk_model=model, artifact_path="random_forest_model")
        print(f"Logged model in internal run {used_run_id}")

# Debug: list artifact folder
artifact_root = os.path.join(os.getcwd(), "mlruns", "0", used_run_id, "artifacts", "random_forest_model")
print(f"DEBUG: artifact root expected at: {artifact_root}")
if os.path.exists(artifact_root):
    for root, dirs, files in os.walk(artifact_root):
        rel = os.path.relpath(root, artifact_root)
        print(f"{rel}/")
        for f in files:
            print(f"  - {f}")
else:
    print("DEBUG: artifact root not found yet.")

print("Finished training & logging. run_id:", used_run_id)

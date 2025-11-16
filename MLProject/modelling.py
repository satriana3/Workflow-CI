# MLProject/modelling.py  (FINAL, robust: works with `mlflow run` + GitHub Actions)
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

# Data path: mlflow run executes with working-directory = MLProject, so use cwd
dataset_path = os.path.normpath(os.path.join(os.getcwd(), args.data_path))
if not os.path.exists(dataset_path):
    fatal(f"Dataset not found at: {dataset_path}")

# Load data
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
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Save model locally in MLflow format (directory)
local_model_dir = os.path.join(os.getcwd(), "ml_local_model")
# remove if exists
if os.path.exists(local_model_dir):
    shutil.rmtree(local_model_dir)
# Save in MLflow-compatible format (contains MLmodel)
mlflow.sklearn.save_model(sk_model=model, path=local_model_dir)
print(f"Saved MLflow-format model locally at: {local_model_dir}")

# === Logging to MLflow: two paths ===
# 1) If running under `mlflow run`, the runner usually sets MLFLOW_RUN_ID in env.
#    We'll prefer to attach using MlflowClient.log_artifacts() so we avoid active_run race.
# 2) If env var not present, start a run here and use mlflow.log_model (standard).
env_run_id = os.environ.get("MLFLOW_RUN_ID")

client = MlflowClient(tracking_uri=os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))

if env_run_id:
    print(f"DEBUG: Detected MLFLOW_RUN_ID in env: {env_run_id}. Will attach via MlflowClient.")
    # Wait until run directory exists (file-store race protection)
    run_dir = os.path.join(os.getcwd(), "mlruns", "0", env_run_id)
    max_wait = 30  # seconds
    waited = 0.0
    sleep_step = 0.5
    while waited < max_wait and not os.path.exists(run_dir):
        time.sleep(sleep_step)
        waited += sleep_step
    if not os.path.exists(run_dir):
        print(f"WARNING: run directory {run_dir} not present after {max_wait}s; trying client.get_run() anyway.")
    # Also wait until MLflow server acknowledges run via client.get_run
    got_run = False
    got_wait = 0.0
    while got_wait < max_wait:
        try:
            r = client.get_run(env_run_id)
            got_run = True
            break
        except Exception as e:
            time.sleep(sleep_step)
            got_wait += sleep_step
    if not got_run:
        fatal(f"Run {env_run_id} not visible to Mlflow after waiting {max_wait}s. Aborting to avoid corruption.")
    # Now log metric and artifacts into that run
    print(f"Logging metric 'accuracy' to run {env_run_id}")
    client.log_metric(env_run_id, "accuracy", float(accuracy))
    # Log artifacts (entire directory) so MLmodel is preserved
    # MlflowClient.log_artifacts(local_dir, run_id, artifact_path=None)
    print(f"Uploading MLflow model directory '{local_model_dir}' as artifacts/random_forest_model ...")
    client.log_artifacts(env_run_id, local_model_dir, artifact_path="random_forest_model")
    print("Artifact upload via MlflowClient complete.")
    used_run_id = env_run_id
else:
    # No env run id: create run here and log model normally
    print("No MLFLOW_RUN_ID in env: starting a local run and logging model normally.")
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run():
        run_id = mlflow.active_run().info.run_id
        print(f"Started internal run: {run_id}")
        # Log metric
        mlflow.log_metric("accuracy", float(accuracy))
        # Log model in MLflow-managed artifacts as 'random_forest_model'
        mlflow.sklearn.log_model(sk_model=model, artifact_path="random_forest_model")
        used_run_id = run_id
        print("Model logged via mlflow.sklearn.log_model in internal run.")

# Debug: list resulting artifact folder
artifact_root = os.path.join(os.getcwd(), "mlruns", "0", used_run_id, "artifacts", "random_forest_model")
print(f"DEBUG: looking for artifact root: {artifact_root}")
if os.path.exists(artifact_root):
    for root, dirs, files in os.walk(artifact_root):
        rel = os.path.relpath(root, artifact_root)
        print(f"{rel}/")
        for f in files:
            print(f"  - {f}")
else:
    print("DEBUG: artifact root not found (unexpected).")

print("Finished training & logging. Run id used:", used_run_id)

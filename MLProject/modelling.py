# MLProject/modelling.py
import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import joblib
import time

def fatal(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(2)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
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

# determine run id: prefer env var, fallback to newest folder in mlruns/0
env_run_id = os.environ.get("MLFLOW_RUN_ID")
if env_run_id:
    print(f"DEBUG: MLFLOW_RUN_ID from env = {env_run_id}")
else:
    # try to find newest run folder in mlruns/0
    exp_dir = os.path.join(os.getcwd(), "mlruns", "0")
    run_id = None
    if os.path.isdir(exp_dir):
        entries = [e for e in os.listdir(exp_dir) if os.path.isdir(os.path.join(exp_dir, e))]
        if entries:
            # choose most recently modified directory
            entries.sort(key=lambda e: os.path.getmtime(os.path.join(exp_dir, e)), reverse=True)
            run_id = entries[0]
            print(f"DEBUG: selected newest run id from mlruns/0 -> {run_id}")
        else:
            print("DEBUG: no runs found in mlruns/0")
    else:
        print("DEBUG: mlruns/0 not present yet")
    env_run_id = run_id

# attach to run if run_id known, else start new run (fallback)
attached_run_id = None
if env_run_id:
    try:
        print(f"Attempting to attach to run_id {env_run_id}")
        # Attach to existing run created by mlflow run
        with mlflow.start_run(run_id=env_run_id):
            attached_run_id = mlflow.active_run().info.run_id
            print(f"Attached to run {attached_run_id}")
            # train & log inside this context
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            print(f"Accuracy: {accuracy}")
            mlflow.log_metric("accuracy", float(accuracy))
            # save local file and log artifact to deterministic artifact path
            local_model_path = os.path.join(os.getcwd(), "model.pkl")
            joblib.dump(model, local_model_path)
            print(f"Saved local model at {local_model_path}")
            mlflow.log_artifact(local_model_path, artifact_path="random_forest_model")
            print("Logged artifact to artifact_path='random_forest_model'")
    except Exception as e:
        print(f"WARNING: attaching by run_id failed: {e}")
        env_run_id = None  # force fallback

if not env_run_id:
    # fallback: create a new run
    print("Starting new MLflow run (fallback)")
    with mlflow.start_run():
        attached_run_id = mlflow.active_run().info.run_id
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"Accuracy: {accuracy}")
        mlflow.log_metric("accuracy", float(accuracy))
        local_model_path = os.path.join(os.getcwd(), "model.pkl")
        joblib.dump(model, local_model_path)
        print(f"Saved local model at {local_model_path}")
        mlflow.log_artifact(local_model_path, artifact_path="random_forest_model")
        print("Logged artifact to artifact_path='random_forest_model'")

# debug: print artifacts folder
time.sleep(0.5)  # small wait for FS flush
run_to_inspect = attached_run_id or env_run_id
if run_to_inspect:
    artifact_root = os.path.join(os.getcwd(), "mlruns", "0", run_to_inspect, "artifacts")
    print(f"DEBUG: artifact root = {artifact_root}")
    if os.path.exists(artifact_root):
        for root, dirs, files in os.walk(artifact_root):
            level = root.replace(artifact_root, "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            for f in files:
                print(f"{indent}  - {f}")
    else:
        print("DEBUG: artifact root does not exist yet.")
else:
    print("DEBUG: no run id available to inspect artifacts.")

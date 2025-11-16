# MLProject/modelling.py  (REPLACE existing file with this)
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import argparse
import os
import sys
import joblib

def fatal(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(2)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

dataset_path = os.path.join(os.path.dirname(__file__), args.data_path)
dataset_path = os.path.normpath(dataset_path)

if not os.path.exists(dataset_path):
    fatal(f"Dataset not found at {dataset_path}")

data = pd.read_csv(dataset_path)

expected_cols = {'average_score_binned', 'average_score'}
if not expected_cols.issubset(set(data.columns)):
    fatal(f"Dataset missing expected columns. Found: {list(data.columns)}")

X = data.drop(['average_score_binned', 'average_score'], axis=1)
y = data['average_score_binned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# MLflow autolog is OK, but we will explicitly log a stable artifact file too
mlflow.sklearn.autolog()

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Eval
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {accuracy}")
print("Classification Report:")
print(report)

# Log metric explicitly
mlflow.log_metric("accuracy", float(accuracy))

# === Save model locally and log as artifact to ensure artifact folder exists ===
local_model_path = os.path.join(os.path.dirname(__file__), "model.pkl")
joblib.dump(model, local_model_path)
print(f"Saved local model to: {local_model_path}")

# log the local file as artifact under 'random_forest_model' artifact folder
mlflow.log_artifact(local_model_path, artifact_path="random_forest_model")
print("Uploaded local model file as artifact to artifact_path='random_forest_model'")

# For debug: list artifacts dir (relative to mlruns) if env var RUN_ID present
try:
    run_id = mlflow.active_run().info.run_id
    print(f"DEBUG: active run id = {run_id}")
    run_artifacts_dir = os.path.join(os.getcwd(), "mlruns", "0", run_id, "artifacts")
    if os.path.exists(run_artifacts_dir):
        print("DEBUG: listing artifacts directory:")
        for root, dirs, files in os.walk(run_artifacts_dir):
            level = root.replace(run_artifacts_dir, "").count(os.sep)
            indent = " " * 2 * (level)
            print(f"{indent}{os.path.basename(root)}/")
            for f in files:
                print(f"{indent}  - {f}")
    else:
        print(f"DEBUG: artifacts dir not found at expected path: {run_artifacts_dir}")
except Exception as e:
    print(f"DEBUG: could not print run artifacts: {e}")

print("Model and metrics logged to MLflow (and artifact file saved).")

# MLProject/modelling.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import argparse
import os
import sys
import joblib

def fatal(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(2)

# ======================
# Parse argument
# ======================
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

# ======================
# Prepare data
# ======================
X = data.drop(['average_score_binned', 'average_score'], axis=1)
y = data['average_score_binned']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ======================
# Get ACTIVE RUN from MLflow
# MLflow CLI already starts run → we must attach to it
# ======================
active = mlflow.active_run()
if active is None:
    fatal("ERROR: No active MLflow run found (CLI run failed).")

run_id = active.info.run_id
print(f"Using active MLflow run id: {run_id}")

# Enable autologging (safe)
mlflow.sklearn.autolog()

# ======================
# Train model
# ======================
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Log metric manually
mlflow.log_metric("accuracy", float(accuracy))

# ======================
# FORCE ARTIFACT CREATION
# ======================
local_model = os.path.join(os.getcwd(), "model.pkl")
joblib.dump(model, local_model)
print(f"Saved model locally at {local_model}")

# THIS GUARANTEES artifacts/random_forest_model exists
mlflow.log_artifact(local_model, artifact_path="random_forest_model")

print("Artifact uploaded into: artifacts/random_forest_model")

# ======================
# DEBUG print artifacts
# ======================
artifact_root = os.path.join(
    os.getcwd(),
    "mlruns", "0", run_id, "artifacts"
)

print("DEBUG: listing artifacts root:", artifact_root)

for root, dirs, files in os.walk(artifact_root):
    level = root.replace(artifact_root, "").count(os.sep)
    indent = " " * 2 * level
    print(f"{indent}{os.path.basename(root)}/")
    for f in files:
        print(f"{indent}  - {f}")

print("FINISHED: model + artifacts logged successfully.")

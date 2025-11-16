import os
import sys
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import shutil

def fatal(msg):
    print("FATAL: " + msg, file=sys.stderr)
    sys.exit(2)

parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--experiment_name", type=str, default="Student Performance Prediction")
args = parser.parse_args()

# Load dataset
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

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)
print(classification_report(y_test, y_pred))

# Save local ML model folder
local_model_dir = "ml_local_model"
if os.path.exists(local_model_dir):
    shutil.rmtree(local_model_dir)
mlflow.sklearn.save_model(model, local_model_dir)

print("Saved ML model locally at:", local_model_dir)

# =========================
# ALWAYS CREATE NEW MLflow RUN
# =========================
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment(args.experiment_name)

with mlflow.start_run() as run:
    run_id = run.info.run_id
    print("Created MLflow run:", run_id)

    mlflow.log_metric("accuracy", float(accuracy))
    mlflow.sklearn.log_model(model, artifact_path="random_forest_model")

print("Final run id:", run_id)

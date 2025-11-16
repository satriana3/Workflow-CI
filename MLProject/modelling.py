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
# Parse arguments
# ----------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
parser.add_argument("--experiment_name", type=str, default="Student Performance Prediction")
args = parser.parse_args()

# ----------------------------
# Fix dataset path handling
# MLflow already runs in MLProject folder,
# so data_path must be relative to project root.
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
# ----------------------------
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "mlruns"))
mlflow.set_experiment(args.experiment_name)

# Note:
# DO NOT CREATE A NEW RUN HERE.
# MLflow CLI already creates the run.
# We attach to it using start_run()
# ----------------------------
with mlflow.start_run(run_name="training_run"):
    print(f"INFO: Active run ID = {mlflow.active_run().info.run_id}")

    mlflow.sklearn.autolog()

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    mlflow.log_metric("accuracy", float(accuracy))

    # ----------------------------
    # Save model locally (inside project folder)
    # ----------------------------
    local_model_path = "model.pkl"
    joblib.dump(model, local_model_path)
    print(f"Saved model → {local_model_path}")

    # ----------------------------
    # Log artifact to MLflow
    # ----------------------------
    mlflow.log_artifact(local_model_path, artifact_path="random_forest_model")
    print("Model uploaded to MLflow under artifact_path='random_forest_model'")

print("Training completed and logged to MLflow.")

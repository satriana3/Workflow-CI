import argparse
import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
import os

warnings.filterwarnings("ignore", category=FutureWarning)

def train_and_log_model(X_train, X_test, y_train, y_test):
    print("🚀 Training RandomForest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc:.4f}")

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(rf, "model")

    report_path = "classification_report.txt"
    with open(report_path, "w") as f:
        f.write(classification_report(y_test, preds))
    mlflow.log_artifact(report_path)

    print("✅ Model training complete and logged to MLflow.")

def main(data_path):
    print("📂 Loading dataset from:", data_path)
    df = pd.read_csv(data_path)

    # Example preprocessing
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop("math score", axis=1)
    y = (df["math score"] > df["math score"].mean()).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # MLflow experiment setup
    mlflow.set_experiment("Student Performance Workflow CI")
    tracking_uri = "file://" + os.path.abspath("mlruns")
    mlflow.set_tracking_uri(tracking_uri)
    print(f"📘 MLflow tracking URI: {tracking_uri}")

    # Cek apakah sudah ada run aktif
    active_run = mlflow.active_run()

    if active_run is not None:
        print(f"ℹ️ Detected existing MLflow run: {active_run.info.run_id}")
        train_and_log_model(X_train, X_test, y_train, y_test)
    else:
        # Cek apakah dijalankan di dalam MLflow CLI (mlflow run .)
        in_mlflow_cli = os.getenv("MLFLOW_PROJECT_ENVIRONMENT") is not None
        if in_mlflow_cli:
            print("⚙️ Running inside mlflow run — using existing run context")
            train_and_log_model(X_train, X_test, y_train, y_test)
        else:
            print("ℹ️ Running manually — starting new MLflow run")
            with mlflow.start_run(run_name="RandomForest_StudentPerformance"):
                train_and_log_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    main(args.data_path)

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

# Fungsi training dan logging
def train_and_log_model(X_train, X_test, y_train, y_test):
    print("🚀 Training RandomForest model...")
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)

    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)
    print(f"✅ Accuracy: {acc:.4f}")

    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(rf, "model")
    mlflow.log_artifact("StudentsPerformance_preprocessing.csv")

    print("✅ Model training complete and logged to MLflow.")

# Fungsi utama
def main(data_path):
    print(f"📂 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Contoh preprocessing sederhana
    df = pd.get_dummies(df, drop_first=True)
    X = df.drop("math score", axis=1)
    y = (df["math score"] > df["math score"].mean()).astype(int)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set experiment
    mlflow.set_experiment("Student Performance Workflow CI")

    tracking_uri = os.getenv("MLFLOW_TRACKING_URI", "file://" + os.path.abspath("mlruns"))
    mlflow.set_tracking_uri(tracking_uri)
    print(f"📘 MLflow tracking URI: {tracking_uri}")

    # Deteksi apakah sudah ada run aktif (karena GitHub Actions via mlflow run)
    active_run = mlflow.active_run()
    if active_run is not None:
        print(f"ℹ️ Detected existing MLflow run ({active_run.info.run_id}), using it.")
        train_and_log_model(X_train, X_test, y_train, y_test)
    else:
        print("ℹ️ No active MLflow run detected — starting a new one...")
        with mlflow.start_run(run_name="RandomForest_StudentPerformance"):
            train_and_log_model(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    print("⚙️ Running inside GitHub Actions (default tracking)")
    main(args.data_path)

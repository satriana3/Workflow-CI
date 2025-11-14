import argparse
import os
import shutil
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report
)

import mlflow
import mlflow.sklearn


def main(data_path):
    print(f"📂 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # Basic feature-target setup
    if "average_score" in df.columns:
        y = (df["average_score"] > df["average_score"].mean()).astype(int)
        X = df.drop(columns=["average_score"], errors="ignore")
    elif "math score" in df.columns:
        y = (df["math score"] > df["math score"].mean()).astype(int)
        X = df.drop(columns=["math score"], errors="ignore")
    else:
        raise ValueError("Dataset tidak memiliki kolom target yang dikenali.")

    # Keep numeric only
    X = X.select_dtypes(include=["number"])

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Setup MLflow Locally
    tracking_dir = os.path.abspath("mlruns")
    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    mlflow.set_experiment("Student Performance Workflow CI")

    with mlflow.start_run() as run:
        run_id = run.info.run_id
        print(f"🔥 Active MLflow Run ID: {run_id}")

        # Model parameters
        params = {
            "n_estimators": 100,
            "max_depth": None,
            "random_state": 42
        }
        mlflow.log_params(params)

        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, Y_train)
        y_pred = model.predict(X_test)

        # Metrics
        metrics = {
            "accuracy": accuracy_score(Y_test, y_pred),
            "f1_score": f1_score(Y_test, y_pred),
            "recall": recall_score(Y_test, y_pred),
            "precision": precision_score(Y_test, y_pred),
        }
        mlflow.log_metrics(metrics)

        # Confusion matrix
        os.makedirs("artifacts", exist_ok=True)
        cm = confusion_matrix(Y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d")
        plt.title("Confusion Matrix")
        cm_path = "artifacts/confusion_matrix.png"
        plt.savefig(cm_path)
        plt.close()
        mlflow.log_artifact(cm_path, artifact_path="artifacts")

        # Classification report
        report_path = "artifacts/classification_report.txt"
        with open(report_path, "w") as f:
            f.write(classification_report(Y_test, y_pred))
        mlflow.log_artifact(report_path, artifact_path="artifacts")

        # Save & log model
        model_path = "model"
        mlflow.sklearn.log_model(model, artifact_path="model")

    print("\n🎉 Training selesai.")
    print("Run ID:", run_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)

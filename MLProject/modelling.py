# ======================================================
# modelling.py 
# ======================================================

import argparse
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report
)


def main(data_path):
    # ===============================
    # 1. Load dataset
    # ===============================
    df = pd.read_csv(data_path)
    target_column = "average_score_binned"

    X = df.drop([target_column, "average_score"], axis=1, errors="ignore")
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ===============================
    # 2. Setup MLflow backend
    # ===============================
    if not os.getenv("GITHUB_ACTIONS"):
        mlflow.set_tracking_uri("sqlite:///mlflow.db")

    mlflow.set_experiment("Student Performance Workflow CI")

    # ===============================
    # 3. Jalankan training
    # ===============================
    active_run = mlflow.active_run()

    if active_run:
        print(f"ℹ️ Detected active MLflow run: {active_run.info.run_id}")
        run_id = active_run.info.run_id
    else:
        print("ℹ️ No active run found, starting new local run...")
        run = mlflow.start_run(run_name="RandomForest_StudentPerformance")
        run_id = run.info.run_id

    print(f"📘 Using run ID: {run_id}")
    train_and_log_model(X_train, X_test, y_train, y_test)

    # End run hanya jika kita yang mulai (local)
    if not os.getenv("GITHUB_ACTIONS"):
        mlflow.end_run()
        print("✅ Local run ended successfully.")


def train_and_log_model(X_train, X_test, y_train, y_test):
    # ===============================
    # Train model
    # ===============================
    params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": 42,
    }
    mlflow.log_params(params)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "precision": precision_score(y_test, y_pred, average="macro"),
    }
    mlflow.log_metrics(metrics)

    # Confusion matrix
    os.makedirs("artifacts", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # Classification report
    report = classification_report(y_test, y_pred)
    report_path = "artifacts/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # Save model
    os.makedirs("output", exist_ok=True)
    model_path = "output/random_forest_model.pkl"
    joblib.dump(model, model_path)
    mlflow.sklearn.log_model(model, artifact_path="model")

    print("\n=== Model Performance Summary ===")
    for k, v in metrics.items():
        print(f"{k:<10}: {v:.4f}")
    print("✅ Model training complete. Logged to MLflow.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)

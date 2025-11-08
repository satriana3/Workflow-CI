# ======================================================
# modelling.py (Final, CI/CD Ready)
# ======================================================

import argparse
import os
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
    # ==================================================
    # 1. Load Dataset
    # ==================================================
    df = pd.read_csv(data_path)
    target_column = "average_score_binned"

    # Memastikan kolom target ada
    if target_column not in df.columns:
        raise ValueError(f"Kolom target '{target_column}' tidak ditemukan dalam dataset!")

    X = df.drop([target_column, "average_score"], axis=1, errors="ignore")
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ==================================================
    # 2. Setup MLflow
    # ==================================================
    # Gunakan database backend modern (SQLite)
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Student Performance Workflow CI")

    # Dapatkan run aktif dari MLflow Projects
    run = mlflow.active_run()
    if run:
        print(f"Active MLflow Run ID: {run.info.run_id}")
    else:
        print("⚠️ Tidak ada active run — menjalankan langsung dalam mode lokal.")
        mlflow.start_run(run_name="StudentPerformance_Run")

    # ==================================================
    # 3. Hyperparameter dan Model
    # ==================================================
    params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": 42
    }
    for key, val in params.items():
        mlflow.log_param(key, val)

    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    # ==================================================
    # 4. Evaluasi Model
    # ==================================================
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "precision": precision_score(y_test, y_pred, average="macro")
    }
    for key, val in metrics.items():
        mlflow.log_metric(key, val)

    # ==================================================
    # 5. Visualisasi Confusion Matrix
    # ==================================================
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")

    os.makedirs("artifacts", exist_ok=True)
    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()
    mlflow.log_artifact(cm_path)

    # ==================================================
    # 6. Classification Report
    # ==================================================
    report = classification_report(y_test, y_pred)
    report_path = "artifacts/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)
    mlflow.log_artifact(report_path)

    # ==================================================
    # 7. Simpan & Log Model
    # ==================================================
    os.makedirs("output", exist_ok=True)
    model_path = "output/random_forest_model.pkl"
    joblib.dump(model, model_path)

    # log model ke MLflow
    mlflow.sklearn.log_model(model, artifact_path="model")

    # ==================================================
    # 8. Ringkasan
    # ==================================================
    print("\n=== Model Performance Summary ===")
    for k, v in metrics.items():
        print(f"{k:<10}: {v:.4f}")

    print("\n✅ Model training selesai dan berhasil dilog ke MLflow.")
    print(f"Model path (local): {model_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)

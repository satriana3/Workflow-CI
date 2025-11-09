# modelling.py
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
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


def safe_log_params(client, run_id, params):
    for k, v in params.items():
        client.log_param(run_id, k, v)


def safe_log_metrics(client, run_id, metrics):
    for k, v in metrics.items():
        client.log_metric(run_id, k, float(v))


def safe_log_artifact(client, run_id, local_path, artifact_path=None):
    # MlflowClient has no direct log_artifact; use file-based artifact upload via artifacts API:
    # fallback to mlflow.log_artifact if there's an active run — else use file copy into mlruns folder
    try:
        # If run active, mlflow.log_artifact will use it
        if mlflow.active_run() is not None:
            mlflow.log_artifact(local_path, artifact_path=artifact_path)
            return
    except Exception:
        pass

    # fallback: copy file into the run's artifacts dir (best-effort)
    run_info = client.get_run(run_id).info
    artifact_uri = run_info.artifact_uri  # e.g., file:///.../mlruns/0/<run_id>/artifacts
    if artifact_uri.startswith("file://"):
        artifact_base = artifact_uri[len("file://"):]
    else:
        artifact_base = artifact_uri
    dest_dir = artifact_base
    if artifact_path:
        dest_dir = os.path.join(artifact_base, artifact_path)
    os.makedirs(dest_dir, exist_ok=True)
    shutil.copy(local_path, os.path.join(dest_dir, os.path.basename(local_path)))


def main(data_path):
    print(f"📂 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # minimal validation (sesuaikan datasetmu)
    if "math score" not in df.columns and "average_score" not in df.columns:
        # kalau datamu berbeda, ubah validasi ini
        print("⚠️ Tidak menemukan kolom 'math score' atau 'average_score' — lanjutkan dengan asumsi kolom numerik.")
    # gunakan kolom numerik sebagai fitur
    X = df.select_dtypes(include=["number"]).copy()
    # pilih target jika ada, fallback: gunakan 'math score' > mean
    if "math score" in df.columns:
        y = (df["math score"] > df["math score"].mean()).astype(int)
        X = X.drop(columns=["math score"], errors="ignore")
    elif "average_score" in df.columns:
        # contoh: binned sudah ada?
        if "average_score_binned" in df.columns:
            y = df["average_score_binned"]
            X = X.drop(columns=["average_score", "average_score_binned"], errors="ignore")
        else:
            y = (df["average_score"] > df["average_score"].mean()).astype(int)
            X = X.drop(columns=["average_score"], errors="ignore")
    else:
        raise ValueError("Tidak ada kolom target yang dikenali. Periksa dataset.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # setup mlflow
    tracking_dir = os.path.abspath("mlruns")
    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    mlflow.set_experiment("Student Performance Workflow CI")
    print(f"📘 MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # prepare common objects
    params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": 42,
    }

    env_run_id = os.environ.get("MLFLOW_RUN_ID") or os.environ.get("MLFLOW_RUN_NAME") or None
    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri().replace("file://", "file://"))

    used_run_id = None
    used_client_logging = False

    # If mlflow run environment created a run, prefer to log into that run via client
    if "MLFLOW_RUN_ID" in os.environ:
        env_run_id = os.environ["MLFLOW_RUN_ID"]
        print(f"ℹ️ Detected MLFLOW_RUN_ID from environment: {env_run_id}")
        # We will not call mlflow.start_run() to avoid run-id conflict; instead log via MlflowClient
        used_run_id = env_run_id
        used_client_logging = True
    else:
        # no env run id — safe to create a run locally with start_run()
        print("ℹ️ No MLFLOW_RUN_ID detected — starting local mlflow run.")
        try:
            run = mlflow.start_run(run_name="RandomForest_StudentPerformance")
            used_run_id = run.info.run_id
            print(f"✅ Created local run: {used_run_id}")
            used_client_logging = False
        except MlflowException as e:
            # fallback to client if start_run unexpectedly fails
            print("⚠️ start_run() failed — falling back to MlflowClient logging. Error:", e)
            # try to create a run via client
            run = client.create_run(experiment_id=mlflow.get_experiment_by_name("Student Performance Workflow CI").experiment_id)
            used_run_id = run.info.run_id
            print(f"✅ Created run via client: {used_run_id}")
            used_client_logging = True

    # Train
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_score": float(f1_score(y_test, y_pred, average="macro")),
        "recall": float(recall_score(y_test, y_pred, average="macro")),
        "precision": float(precision_score(y_test, y_pred, average="macro"))
    }

    # Logging (safe)
    if used_client_logging:
        print("ℹ️ Logging params/metrics/artifacts using MlflowClient to run_id =", used_run_id)
        safe_log_params(client, used_run_id, params)
        safe_log_metrics(client, used_run_id, metrics)
    else:
        # active run exists — use convenient mlflow.log_*
        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

    # Confusion matrix artifact
    os.makedirs("artifacts", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    cm_path = "artifacts/confusion_matrix.png"
    plt.savefig(cm_path)
    plt.close()

    # classification report
    report = classification_report(y_test, y_pred)
    report_path = "artifacts/classification_report.txt"
    with open(report_path, "w") as f:
        f.write(report)

    # Log artifacts
    if used_client_logging:
        # upload artifacts into run's artifact directory (file-based)
        safe_log_artifact(client, used_run_id, cm_path, artifact_path="artifacts")
        safe_log_artifact(client, used_run_id, report_path, artifact_path="artifacts")
    else:
        mlflow.log_artifact(cm_path, artifact_path="artifacts")
        mlflow.log_artifact(report_path, artifact_path="artifacts")

    # Save model file locally too
    os.makedirs("output", exist_ok=True)
    model_file = "output/random_forest_model.pkl"
    joblib.dump(model, model_file)

    # Log model via mlflow.sklearn.log_model (this ensures proper MLmodel metadata)
    if used_client_logging:
        # mlflow.sklearn.log_model expects an active run normally; as a workaround, use client model logging:
        # We'll create a temporary run context by setting MLFLOW_RUN_ID env and using mlflow.sklearn.log_model inside it.
        # But simpler: use mlflow.sklearn.save_model + copy files into artifacts dir.
        tmp_dir = "tmp_model_artifact"
        if os.path.exists(tmp_dir):
            shutil.rmtree(tmp_dir)
        mlflow.sklearn.save_model(sk_model=model, path=tmp_dir)
        # copy saved model into run artifact folder
        safe_log_artifact(client, used_run_id, os.path.join(tmp_dir, "MLmodel"), artifact_path="model")
        # copy whole tree
        dest_artifact_base = client.get_run(used_run_id).info.artifact_uri.replace("file://", "")
        dst_model_dir = os.path.join(dest_artifact_base, "model")
        shutil.copytree(tmp_dir, dst_model_dir, dirs_exist_ok=True)
        shutil.rmtree(tmp_dir)
        print(f"✅ Model saved and copied into artifacts for run {used_run_id}")
    else:
        mlflow.sklearn.log_model(sk_model=model, artifact_path="model")
        print("✅ Model logged with mlflow.sklearn.log_model")

    # Print summary
    print("\n=== Model Performance Summary ===")
    for k, v in metrics.items():
        print(f"{k:<10}: {v:.4f}")

    # IMPORTANT: copy run folder to model/mlruns/... so pipeline can find it reliably
    try:
        src_run_dir = os.path.join("mlruns", "0", used_run_id)
        dst_base = os.path.join("model", "mlruns", "0", used_run_id)
        if os.path.exists(src_run_dir):
            os.makedirs(os.path.dirname(dst_base), exist_ok=True)
            shutil.copytree(src_run_dir, dst_base, dirs_exist_ok=True)
            print(f"📁 Copied run folder to {dst_base}")
        else:
            print("⚠️ Source mlruns run directory not found — cannot copy to model/mlruns")
    except Exception as e:
        print("⚠️ Gagal menyalin run folder:", e)

    # If we created a run with mlflow.start_run earlier and we are the owner, end it
    if (not used_client_logging) and mlflow.active_run() is not None:
        mlflow.end_run()
        print("ℹ️ Ended local mlflow run.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    main(args.data_path)

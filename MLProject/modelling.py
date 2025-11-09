# modelling.py 
import os
import argparse
import shutil
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, recall_score, precision_score,
    confusion_matrix, classification_report
)

import mlflow
from mlflow.tracking import MlflowClient


def train_and_save_model(X_train, X_test, y_train, y_test, out_dir="output"):
    params = {
        "n_estimators": 100,
        "max_depth": None,
        "min_samples_split": 2,
        "random_state": 42,
    }
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred, average="macro"),
        "recall": recall_score(y_test, y_pred, average="macro"),
        "precision": precision_score(y_test, y_pred, average="macro"),
    }

    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, "random_forest_model.pkl")
    joblib.dump(model, model_path)

    # save confusion matrix and report locally
    os.makedirs("artifacts", exist_ok=True)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d")
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    cm_path = os.path.join("artifacts", "confusion_matrix.png")
    plt.savefig(cm_path)
    plt.close()

    report = classification_report(y_test, y_pred)
    report_path = os.path.join("artifacts", "classification_report.txt")
    with open(report_path, "w") as f:
        f.write(report)

    return params, metrics, model_path, cm_path, report_path


def main(data_path):
    print(f"📂 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # --- Adjust this check to the column names in your dataset ---
    # If your real preprocessing file has 'average_score_binned' use that instead.
    if "math score" in df.columns:
        label_col = "math score"
        # create a simple binary target as example
        y = (df[label_col] > df[label_col].mean()).astype(int)
        X = df.select_dtypes(include=["number"]).drop(columns=[label_col], errors="ignore")
    elif "average_score_binned" in df.columns:
        label_col = "average_score_binned"
        X = df.drop([label_col, "average_score"], axis=1, errors="ignore")
        y = df[label_col]
    else:
        raise ValueError("Dataset tidak mengandung kolom 'math score' atau 'average_score_binned'. Sesuaikan modelling.py.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # ---------- MLflow setup ----------
    # Use local file backend so mlruns folder appears in repo/workdir (makes CI easier)
    tracking_dir = os.path.abspath("mlruns")
    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    exp_name = "Student Performance Workflow CI"

    client = MlflowClient(tracking_uri=mlflow.get_tracking_uri())

    # Create experiment if not exists
    exp = client.get_experiment_by_name(exp_name)
    if exp is None:
        exp_id = client.create_experiment(exp_name)
    else:
        exp_id = exp.experiment_id

    print("📘 MLflow tracking URI:", mlflow.get_tracking_uri())
    env_run_id = os.environ.get("MLFLOW_RUN_ID")  # set by `mlflow run` / Projects
    run_id = None

    if env_run_id:
        # If mlflow run already created the run, try to use it
        try:
            client.get_run(env_run_id)  # validate exists
            run_id = env_run_id
            print("ℹ️ Running inside mlflow run. Using existing run_id:", run_id)
        except Exception:
            # If not found (rare), fall back to create a new run
            print("⚠️ MLFLOW_RUN_ID present but run not found in store. Creating new run.")
    if run_id is None:
        # create a fresh run under our experiment (safe for local/manual execution)
        created = client.create_run(exp_id, run_name="RandomForest_StudentPerformance")
        run_id = created.info.run_id
        print("ℹ️ Created new run_id:", run_id)

    # --- Train & save artifacts locally ---
    params, metrics, model_path, cm_path, report_path = train_and_save_model(
        X_train, X_test, y_train, y_test, out_dir="output"
    )

    # --- Log params, metrics, artifacts, and model using MlflowClient (explicit run_id) ---
    for k, v in params.items():
        client.log_param(run_id, k, v)
    for k, v in metrics.items():
        client.log_metric(run_id, k, float(v))

    # log artifacts
    client.log_artifact(run_id, cm_path, artifact_path="artifacts")
    client.log_artifact(run_id, report_path, artifact_path="artifacts")

    # log saved model file as artifact under artifact_path "model" so mlflow models build-docker can find it
    # We'll copy local output into a folder that becomes artifacts/model/
    # Mlflow expects a directory under artifacts/model that contains model files (for mlflow.sklearn: conda.yaml + data + MLmodel, but for simplicity we'll store our pickle as model.pkl)
    model_artifact_dir = "temp_model_for_artifact"
    if os.path.exists(model_artifact_dir):
        shutil.rmtree(model_artifact_dir)
    os.makedirs(model_artifact_dir, exist_ok=True)
    shutil.copy(model_path, os.path.join(model_artifact_dir, "model.pkl"))

    # Use client.log_artifact to upload directory contents
    # But client.log_artifact only uploads single files. To upload folder, walk and log each file into artifact_path "model"
    for root, _, files in os.walk(model_artifact_dir):
        for f in files:
            local_file = os.path.join(root, f)
            rel_dir = os.path.relpath(root, model_artifact_dir)
            if rel_dir == ".":
                artifact_subpath = "model"
            else:
                artifact_subpath = os.path.join("model", rel_dir)
            client.log_artifact(run_id, local_file, artifact_path=artifact_subpath)

    # Also create a convenience copy under model/mlruns so GitHub Actions build step can find it as we arranged
    latest_run = run_id
    target_model_path = os.path.join("model", "mlruns", "0", latest_run, "artifacts", "model")
    os.makedirs(target_model_path, exist_ok=True)
    shutil.copy(os.path.join(model_artifact_dir, "model.pkl"), os.path.join(target_model_path, "model.pkl"))

    print("\n=== Model Performance Summary ===")
    for k, v in metrics.items():
        print(f"{k:<10}: {v:.4f}")

    print("\n✅ Training complete. Artifacts logged under run_id:", run_id)
    print("Model artifacts (convenience copy) at:", target_model_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()
    print("⚙️ Running modelling.py ...")
    main(args.data_path)

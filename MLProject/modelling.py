import os
import shutil
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def main(data_path):
    print(f"📂 Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)

    # --- Simple preprocessing ---
    if "math score" not in df.columns:
        raise ValueError("Kolom 'math score' tidak ditemukan di dataset.")

    X = df.select_dtypes(include=["number"]).dropna(axis=1)
    y = (df["math score"] > df["math score"].mean()).astype(int)
    X = X.drop(columns=["math score"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- MLflow setup ---
    tracking_dir = os.path.abspath("mlruns")
    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    mlflow.set_experiment("Student Performance Workflow CI")

    print(f"📘 MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # --- Start MLflow run ---
    with mlflow.start_run(run_name="RandomForest_StudentPerformance"):
        mlflow.autolog()

        print("🚀 Training model RandomForestClassifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        print(f"✅ Model trained successfully — Accuracy: {acc:.4f}")

        mlflow.log_metric("accuracy_manual", acc)
        mlflow.sklearn.log_model(model, "model")

    print("📦 Model logged successfully to MLflow artifacts.")

    latest_run = sorted(os.listdir("mlruns/0"))[-1]
    src_model = f"mlruns/0/{latest_run}/artifacts/model"
    if os.path.exists(src_model):
        os.makedirs("model/mlruns/0", exist_ok=True)
        shutil.copytree(src_model, f"model/mlruns/0/{latest_run}/artifacts/model", dirs_exist_ok=True)
        print(f"✅ Model disalin ke model/mlruns/0/{latest_run}/artifacts/model untuk CI.")
    else:
        print("⚠️ Model artifacts tidak ditemukan — pastikan log_model berjalan.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    print("⚙️ Running modelling.py ...")
    main(args.data_path)

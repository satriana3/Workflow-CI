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

    # --- Simple preprocessing (contoh aman) ---
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

    # --- Gunakan autolog agar tidak perlu start_run() manual ---
    mlflow.autolog()

    print("🚀 Training model RandomForestClassifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    acc = accuracy_score(y_test, preds)
    print(f"✅ Model trained successfully — Accuracy: {acc:.4f}")

    # --- Log manual metric tambahan (tanpa start_run) ---
    mlflow.log_metric("accuracy_manual", acc)
    mlflow.sklearn.log_model(model, "model")

    print("📦 Model logged successfully to MLflow artifacts.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    print("⚙️ Running modelling.py ...")
    main(args.data_path)

# Pastikan folder output model ada untuk CI
if os.path.exists("mlruns"):
    os.makedirs("model", exist_ok=True)
    shutil.copytree("mlruns", "model/mlruns", dirs_exist_ok=True)
    print("✅ Folder mlruns disalin ke model/mlruns agar CI dapat menemukannya.")
else:
    print("⚠️ Folder mlruns tidak ditemukan, pastikan MLflow berjalan dengan benar.")

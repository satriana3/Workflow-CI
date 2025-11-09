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

    # --- Validasi kolom ---
    if "math score" not in df.columns:
        raise ValueError("Kolom 'math score' tidak ditemukan di dataset.")

    # --- Preprocessing sederhana ---
    X = df.select_dtypes(include=["number"]).dropna(axis=1)
    y = (df["math score"] > df["math score"].mean()).astype(int)
    X = X.drop(columns=["math score"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # --- Setup MLflow ---
    tracking_dir = os.path.abspath("mlruns")
    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    mlflow.set_experiment("Student Performance Workflow CI")

    print(f"📘 MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # Jalankan run MLflow
    with mlflow.start_run(run_name="RandomForest_StudentPerformance") as run:
        run_id = run.info.run_id
        print(f"🚀 Run ID aktif: {run_id}")

        # Autolog semua parameter & metric
        mlflow.autolog()

        # --- Model Training ---
        print("🚀 Training model RandomForestClassifier...")
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        acc = accuracy_score(y_test, preds)
        print(f"✅ Model trained successfully — Accuracy: {acc:.4f}")

        # Log metrik tambahan manual
        mlflow.log_metric("accuracy_manual", acc)

        # --- Log model ke MLflow (format resmi) ---
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            registered_model_name="student_performance_model"
        )
        print("📦 Model berhasil disimpan ke MLflow artifacts.")

    # --- Copy hasil training ke folder 'model/' agar CI bisa mendeteksi ---
    print("📁 Menyiapkan folder model/mlruns untuk CI...")
    target_base = os.path.join("model", "mlruns", "0", run_id)
    src_dir = os.path.join("mlruns", "0", run_id)
    if os.path.exists(src_dir):
        shutil.copytree(src_dir, target_base, dirs_exist_ok=True)
        print(f"✅ Folder MLflow run disalin ke: {target_base}")
    else:
        print("⚠️ Folder run source tidak ditemukan!")

    print("🎉 Training dan logging selesai tanpa error!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    print("⚙️ Running modelling.py ...")
    main(args.data_path)

import os
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
    X = df.select_dtypes(include=["number"]).dropna(axis=1)
    y = (df["math score"] > df["math score"].mean()).astype(int)
    X = X.drop(columns=["math score"], errors="ignore")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # --- MLflow setup ---
    tracking_dir = os.path.abspath("mlruns")
    mlflow.set_tracking_uri(f"file://{tracking_dir}")
    mlflow.set_experiment("Student Performance Workflow CI")

    print(f"📘 MLflow tracking URI: {mlflow.get_tracking_uri()}")

    # --- Cek apakah sudah ada run aktif (dari `mlflow run`) ---
    active_run = mlflow.active_run()

    if active_run:
        print(f"ℹ️ Detected active run: {active_run.info.run_id}")
    else:
        print("ℹ️ No active run detected — starting new one")
        mlflow.start_run(run_name="RandomForest_StudentPerformance")

    # --- Model training ---
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    acc = accuracy_score(y_test, preds)

    print(f"✅ Model trained successfully — Accuracy: {acc:.4f}")

    # --- Log ke MLflow ---
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(rf, "model")

    # --- Tutup run hanya jika kita yang buka ---
    if not active_run:
        mlflow.end_run()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    print("⚙️ Running modelling.py ...")
    main(args.data_path)

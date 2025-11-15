# MLProject/modelling.py
import argparse
import os
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


def main(data_path: str, tracking_uri: str = None, experiment_name: str = "student_performance_experiment"):
    # Jika diberikan tracking_uri oleh workflow, gunakan itu (agar mlruns tersimpan di WORKSPACE)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Pastikan experiment ter-set (jika belum ada, akan dibuat)
    mlflow.set_experiment(experiment_name)

    # Gunakan nested run untuk menghindari konflik ketika mlflow CLI sudah membuat run
    # (mlflow run ... memulai run env ; kita pakai nested=True agar tidak error)
    with mlflow.start_run(nested=True):
        # Enable autolog (sklearn)
        mlflow.sklearn.autolog()

        # Load dataset
        print(f"📂 Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)

        # Periksa kolom yang relevan; adaptif untuk dataset Anda
        if "average_score_binned" in df.columns and "average_score" in df.columns:
            X = df.drop(["average_score_binned", "average_score"], axis=1)
            y = df["average_score_binned"]
        elif "math score" in df.columns:
            # contoh fallback (jika dataset berbeda)
            X = df.drop(["math score"], axis=1)
            y = df["math score"]
        else:
            raise ValueError("Dataset tidak mengandung kolom target yang dikenali.")

        # Hanya ambil numeric features (jika perlu)
        X = X.select_dtypes(include=["number"])

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Predict & eval
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        report = classification_report(y_test, preds, zero_division=0)

        print(f"\nAccuracy: {acc}")
        print("\nClassification report:")
        print(report)

        # Log metric manual (opsional); autolog juga sudah merekam
        mlflow.log_metric("accuracy_manual", float(acc))

        # === PENTING: simpan model dengan artifact_path 'model' agar mlflow models build-docker bisa menemukan ===
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("✅ Model logged to MLflow (artifact_path='model')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to csv dataset")
    parser.add_argument("--tracking_uri", type=str, required=False, help="MLflow tracking URI (file://...)")
    parser.add_argument("--experiment_name", type=str, required=False, default="student_performance_experiment")
    args = parser.parse_args()

    main(args.data_path, tracking_uri=args.tracking_uri, experiment_name=args.experiment_name)

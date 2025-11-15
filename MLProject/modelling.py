# modelling.py

import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn


def load_data(data_path: str):
    """Load dataset dari path argument MLflow."""
    return pd.read_csv(data_path)


def main(data_path: str):

    # Buat experiment khusus (opsional tapi direkomendasikan)
    mlflow.set_experiment("student_performance_experiment")

    # Aktifkan autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run():

        # Load data
        data = load_data(data_path)

        # Pastikan kolom tersedia
        required_columns = ["average_score_binned", "average_score"]
        for col in required_columns:
            if col not in data.columns:
                raise ValueError(f"Kolom '{col}' tidak ditemukan di dataset!")

        # Pisahkan fitur dan target
        X = data.drop(required_columns, axis=1)
        y = data["average_score_binned"]

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)

        # Train
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Evaluasi manual
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        print(f"\nAccuracy: {accuracy}")
        print("\nClassification Report:")
        print(report)

        # Log metric manual
        mlflow.log_metric("accuracy_manual", accuracy)

        # Simpan model — gunakan nama "model" agar CI tidak error
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("\nModel & metrics logged successfully to MLflow")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path ke dataset preprocessing"
    )
    args = parser.parse_args()

    main(args.data_path)

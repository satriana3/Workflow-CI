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
    data = pd.read_csv(data_path)
    return data


def main(data_path: str):
    # MLflow autolog
    mlflow.sklearn.autolog()

    # Mulai eksperimen MLflow
    with mlflow.start_run():
        # Load dataset
        data = load_data(data_path)

        # Pisahkan fitur dan target
        X = data.drop(['average_score_binned', 'average_score'], axis=1)
        y = data['average_score_binned']

        # Bagi data train / test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Inisialisasi model
        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        # Train model
        model.fit(X_train, y_train)

        # Prediksi
        y_pred = model.predict(X_test)

        # Evaluasi
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)

        # Print ke terminal (untuk CI logs)
        print(f"\nAccuracy: {accuracy}")
        print("\nClassification Report:")
        print(report)

        # Log metrik manual (opsional)
        mlflow.log_metric("accuracy_manual", accuracy)

        # Simpan model
        mlflow.sklearn.log_model(model, "random_forest_model")

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

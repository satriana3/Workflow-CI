# modelling.py

import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


def main(data_path):

    # Tidak memulai run baru jika MLflow sudah memulai run dari CLI
    # Gunakan start_run(nested=True) untuk menghindari conflict
    with mlflow.start_run(nested=True):

        mlflow.autolog()

        # Load dataset
        df = pd.read_csv(data_path)

        X = df.drop("math score", axis=1)
        y = df["math score"]

        # Train test split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Model
        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        # Prediction
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        # Log metrics
        mlflow.log_metric("accuracy", acc)

        print(f"Model Accuracy: {acc}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    args = parser.parse_args()

    main(args.data_path)

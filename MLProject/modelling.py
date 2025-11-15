# MLProject/modelling.py
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

def main(data_path: str, tracking_uri: str = None, experiment_name: str = "student_performance_experiment"):
    # Resolve data path (absolute if relative)
    data_path = os.path.abspath(data_path)

    # If provided, set tracking uri (file://... to store mlruns in workspace)
    if tracking_uri:
        mlflow.set_tracking_uri(tracking_uri)

    # Ensure experiment is set (create if missing)
    mlflow.set_experiment(experiment_name)

    # Use nested run to avoid conflicts when invoked from mlflow CLI
    with mlflow.start_run(nested=True):
        # Enable autolog
        mlflow.sklearn.autolog()

        print(f"📂 Loading dataset from: {data_path}")
        df = pd.read_csv(data_path)

        # Try to find target column; fallback to 'math score' if needed
        if "average_score_binned" in df.columns and "average_score" in df.columns:
            X = df.drop(["average_score_binned", "average_score"], axis=1)
            y = df["average_score_binned"]
        elif "math score" in df.columns:
            X = df.drop(["math score"], axis=1)
            y = df["math score"]
        else:
            raise ValueError("Dataset tidak mengandung kolom target yang dikenali.")

        # select numeric features only to avoid categorical encoding issues
        X = X.select_dtypes(include=["number"])

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestClassifier(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = float(accuracy_score(y_test, preds))
        report = classification_report(y_test, preds, zero_division=0)

        print(f"\nAccuracy: {acc}")
        print("\nClassification report:")
        print(report)

        # log manual metric in addition to autolog
        mlflow.log_metric("accuracy_manual", acc)

        # IMPORTANT: save model using artifact_path "model" so build-docker can find it
        mlflow.sklearn.log_model(model, artifact_path="model")

        print("✅ Model logged to MLflow (artifact_path='model')")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="Path to csv dataset")
    parser.add_argument("--tracking_uri", type=str, required=False, default="file://./mlruns", help="MLflow tracking URI (file://...)")
    parser.add_argument("--experiment_name", type=str, required=False, default="student_performance_experiment")
    args = parser.parse_args()

    main(args.data_path, tracking_uri=args.tracking_uri, experiment_name=args.experiment_name)

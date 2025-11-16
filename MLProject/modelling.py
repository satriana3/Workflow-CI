import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path")
    args = parser.parse_args()

    # Tracking local file store
    mlflow.set_tracking_uri("file:./mlruns")
    mlflow.set_experiment("Student Performance Prediction")

    # Start SINGLE RUN — no nested!
    with mlflow.start_run(run_name="training_run"):
        df = pd.read_csv(args.data_path)

        X = df.drop("performance_level", axis=1)
        y = df["performance_level"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        model = RandomForestClassifier(random_state=42)
        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)

        print("Accuracy:", acc)
        print(classification_report(y_test, preds))

        # Log metrics and model
        mlflow.log_metric("accuracy", acc)
        mlflow.sklearn.log_model(model, "random_forest_model")

        # Log artifact
        mlflow.log_artifact(args.data_path)

        # Params
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("model_type", "RandomForestClassifier")

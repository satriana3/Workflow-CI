# MLProject/modelling.py
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn

# -------------------------------
# Argument parsing untuk MLflow CLI
# -------------------------------
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# -------------------------------
# Load dataset
# -------------------------------
df = pd.read_csv(args.data_path)

# Pastikan kolom sesuai dataset
X = df.drop(["average_score_binned", "average_score"], axis=1)
y = df["average_score_binned"]

# -------------------------------
# Split data
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# MLflow setup
# -------------------------------
mlflow.set_experiment("Student Performance Prediction")

# Tentukan folder artifact tetap
artifact_dir = os.path.join("artifacts", "random_forest_model")
os.makedirs(artifact_dir, exist_ok=True)

with mlflow.start_run():
    mlflow.sklearn.autolog()

    # ---------------------------
    # Train model
    # ---------------------------
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Print results
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # Manual log metric
    mlflow.log_metric("accuracy", accuracy)

    # Save model ke folder artifact tetap
    mlflow.sklearn.save_model(model, path=artifact_dir)
    print(f"Model saved to {artifact_dir}")

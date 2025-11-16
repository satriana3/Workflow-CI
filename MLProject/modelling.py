import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
import argparse
import os

# Arguments
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()

# Set experiment
mlflow.set_experiment("Student Performance Prediction")

# Start run normally (do NOT attach to MLflow run ID)
with mlflow.start_run(run_name="training_run"):
    
    # Load data
    df = pd.read_csv(args.data_path)

    # Encode target
    le = LabelEncoder()
    df['performance'] = le.fit_transform(df['performance'])

    # Train test split
    X = df.drop("performance", axis=1)
    y = df["performance"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predict
    y_pred = model.predict(X_test)

    # Metrics
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print("Accuracy:", acc)
    print("Classification Report:\n", report)

    # Log metrics
    mlflow.log_metric("accuracy", acc)
    
    # Log model
    mlflow.sklearn.log_model(model, artifact_path="model")

    # Save local
    output_path = "ml_local_model"
    mlflow.sklearn.save_model(model, output_path)
    print("Saved MLflow-format model locally at:", os.path.abspath(output_path))

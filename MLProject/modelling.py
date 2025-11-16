import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import mlflow
import mlflow.sklearn
import argparse

# parsing argumen
parser = argparse.ArgumentParser()
parser.add_argument("--data_path", type=str, required=True)
args = parser.parse_args()
dataset_path = args.data_path

# load dataset
data = pd.read_csv(dataset_path)

# pisahkan fitur dan target
X = data.drop(['average_score_binned','average_score'], axis=1)
y = data['average_score_binned']

# bagi data menjadi training dan testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# konfigurasikan MLflow
mlflow.set_experiment("Student Performance Prediction")

with mlflow.start_run():
    mlflow.sklearn.autolog()
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(report)

    mlflow.log_metric("accuracy", accuracy)
    mlflow.sklearn.log_model(model, "random_forest_model")

    print("Model and metrics logged to MLflow")

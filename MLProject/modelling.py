# MLProject/modelling.py
import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    log_loss,
    matthews_corrcoef,
    roc_auc_score,
    cohen_kappa_score
)
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
    y_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None

    # ---------------------------
    # Hitung metric
    # ---------------------------
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    
    # Additional recommended metrics
    f1_macro = f1_score(y_test, y_pred, average="macro", zero_division=0)
    recall_macro = recall_score(y_test, y_pred, average="macro", zero_division=0)
    
    balanced_acc = balanced_accuracy_score(y_test, y_pred)
    cohen_kappa = cohen_kappa_score(y_test, y_pred)
    matthews = matthews_corrcoef(y_test, y_pred)
    logloss = log_loss(y_test, y_proba) if y_proba is not None else 0.0
    
    # ROC AUC (only when 2 classes)
    try:
        roc_auc = roc_auc_score(y_test, y_proba[:, 1]) if y_proba.shape[1] == 2 else 0.0
    except:
        roc_auc = 0.0


    # ---------------------------
    # Print results
    # ---------------------------
    print(f"Accuracy: {accuracy}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))

    # ---------------------------
    # Log semua metric ke MLflow
    # ---------------------------
    mlflow.log_metrics({
    "accuracy": accuracy,
    "precision_weighted": precision,
    "recall_weighted": recall,
    "f1_weighted": f1,
    "f1_macro": f1_macro,
    "recall_macro": recall_macro,
    "balanced_accuracy": balanced_acc,
    "cohen_kappa": cohen_kappa,
    "matthews_corrcoef": matthews,
    "log_loss": logloss
    })



    # Save model ke folder artifact tetap
    mlflow.sklearn.save_model(model, path=artifact_dir)
    print(f"Model saved to {artifact_dir}")

import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import mlflow
import matplotlib.pyplot as plt
import seaborn as sns
import dagshub
import os

# 1. Hubungkan ke DagsHub
dagshub.init(repo_owner='vivalafreak1', repo_name='Telco-Churn-MLOps', mlflow=True)

# 2. Load Data Hasil Preprocessing
df = pd.read_csv("../telco_preprocessing/telco_processed.csv")
X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Setup Eksperimen MLflow
mlflow.set_experiment("Telco_Churn_XGBoost_Advance")

# 4. Hyperparameter Tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2]
}

# Gunakan XGBoost
xgb_model = xgb.XGBClassifier(eval_metric='logloss')
random_search = RandomizedSearchCV(xgb_model, param_distributions=param_grid, n_iter=5, cv=3, random_state=42)

# 5. MLflow Tracking (Manual Logging)
with mlflow.start_run(run_name="XGBoost_Tuning"):
    print("Melatih model dan mencari hyperparameter terbaik...")
    random_search.fit(X_train, y_train)

    best_model = random_search.best_estimator_
    y_pred = best_model.predict(X_test)

    # Hitung Metrics
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Akurasi Model Terbaik: {acc:.4f}")

    # Manual Logging: Parameter & Metrics
    mlflow.log_params(random_search.best_params_)
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("precision", prec)
    mlflow.log_metric("recall", rec)
    mlflow.log_metric("f1_score", f1)

    # Artefak 1: Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    cm_path = "confusion_matrix.png"
    plt.savefig(cm_path)
    mlflow.log_artifact(cm_path)
    plt.close()

    # Artefak 2: Feature Importance
    plt.figure(figsize=(10,6))
    xgb.plot_importance(best_model, max_num_features=10)
    plt.title("Top 10 Feature Importance")
    fi_path = "feature_importance.png"
    plt.savefig(fi_path, bbox_inches='tight')
    mlflow.log_artifact(fi_path)
    plt.close()

    # Log Model & Input Example (sangat penting untuk Kriteria 4 nanti)
    input_example = X_train.iloc[:5]
    mlflow.xgboost.log_model(best_model, "model", input_example=input_example)

    print("Training selesai! Artefak dan model telah dikirim ke DagsHub.")
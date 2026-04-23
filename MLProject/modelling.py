import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
import mlflow
import sys

if __name__ == "__main__":
    # Mengambil argumen path dataset dari MLproject
    file_path = sys.argv[1] if len(sys.argv) > 1 else "telco_preprocessing/telco_processed.csv"
    
    print(f"Memuat dataset dari: {file_path}")
    df = pd.read_csv(file_path)
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Set nama eksperimen
    # mlflow.set_experiment("Telco_Churn_CI")

    with mlflow.start_run():
        mlflow.autolog() # Autolog 
        
        # Train model sederhana
        model = xgb.XGBClassifier(eval_metric='logloss')
        model.fit(X_train, y_train)
        
        # Log model dengan input example
        input_example = X_train.iloc[:5]
        mlflow.xgboost.log_model(xgb_model=model, artifact_path="model", input_example=input_example)
        print("Model berhasil dilatih dan disimpan di mlruns!")
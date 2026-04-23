import pandas as pd
import os

def load_data(file_path):
    print(f"Memuat data dari {file_path}...")
    return pd.read_csv(file_path)

def preprocess_data(df):
    print("Melakukan preprocessing data...")
    # 1. Menangani TotalCharges
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'] = df['TotalCharges'].fillna(0)
    
    # 2. Menghapus customerID
    if 'customerID' in df.columns:
        df.drop('customerID', axis=1, inplace=True)
    
    # 3. Encoding Kategorikal
    X = df.drop('Churn', axis=1)
    y = df['Churn'].map({'Yes': 1, 'No': 0})
    
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # 4. Gabungkan kembali
    df_processed = pd.concat([X_encoded, y], axis=1)
    return df_processed

def save_data(df, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Data berhasil disimpan di {output_path}")

if __name__ == "__main__":
    input_file = "telco_raw/Telco-Customer-Churn.csv"
    output_file = "telco_preprocessing/telco_processed.csv"
    
    # Eksekusi pipeline
    raw_data = load_data(input_file)
    processed_data = preprocess_data(raw_data)
    save_data(processed_data, output_file)
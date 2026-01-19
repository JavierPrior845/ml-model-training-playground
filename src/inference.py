import joblib
import pandas as pd
import numpy as np
from pathlib import Path

# 1. Definimos las rutas relativas al archivo actual
# 'Path(__file__).resolve()' obtiene la ruta absoluta de inference.py
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model_v1.pkl"
DATA_PATH = BASE_DIR / "data" / "test" / "test_sample.csv"


def predict_churn(csv_path, model_path):
    # Validamos que los archivos existan antes de empezar
    if not model_path.exists():
        raise FileNotFoundError(f"No se encuentra el modelo en: {model_path}")
    if not csv_path.exists():
        raise FileNotFoundError(f"No se encuentra el CSV en: {csv_path}")

    # 1. Load Assets
    assets = joblib.load(model_path)
    ohe = assets['ohe']
    te = assets['te']
    scaler = assets['scaler']
    pca = assets['pca']
    model = assets['model']

    # 2. Load New Data
    df = pd.read_csv(csv_path)

    y_true = df['Churn'].map({'Yes': 1, 'No': 0}) if 'Churn' in df.columns else None
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # 3. Basic Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # 4. Transform
    ohe_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
    te_cols = ['Contract', 'PaymentMethod']
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    X_ohe = ohe.transform(df[ohe_cols])
    X_te = te.transform(df[te_cols])
    X_combined = np.hstack([df[num_cols].values, X_ohe, X_te])

    X_scaled = scaler.transform(X_combined)
    X_final = pca.transform(X_scaled)

    # 5. Prediction
    predictions = model.predict(X_final)

    return predictions, y_true


if __name__ == "__main__":
    print(f"Ejecutando inferencia desde: {BASE_DIR}")
    preds, real = predict_churn(DATA_PATH, MODEL_PATH)

    if real is not None:
        from sklearn.metrics import accuracy_score

        print(f"Inference Accuracy: {accuracy_score(real, preds):.4f}")

    print("\n--- Primeras 5 Predicciones (0=No, 1=Yes) ---")
    print(preds[:5])
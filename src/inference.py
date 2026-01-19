import joblib
import pandas as pd
import numpy as np


def predict_churn(csv_path, model_path):
    # 1. Load Assets
    assets = joblib.load(model_path)
    ohe = assets['ohe']
    te = assets['te']
    scaler = assets['scaler']
    pca = assets['pca']
    model = assets['model']

    # 2. Load New Data
    df = pd.read_csv(csv_path)

    # Guardamos el valor real si existe para comparar al final
    y_true = df['Churn'].map({'Yes': 1, 'No': 0}) if 'Churn' in df.columns else None
    if 'Churn' in df.columns:
        df = df.drop('Churn', axis=1)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)

    # 3. Basic Cleaning
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

    # 4. Transform (The same columns as in training)
    ohe_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
    te_cols = ['Contract', 'PaymentMethod']
    num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']

    # Aplicamos transformaciones
    X_ohe = ohe.transform(df[ohe_cols])
    X_te = te.transform(df[te_cols])
    X_combined = np.hstack([df[num_cols].values, X_ohe, X_te])

    # Escalado y PCA
    X_scaled = scaler.transform(X_combined)
    X_final = pca.transform(X_scaled)

    # 5. Prediction
    predictions = model.predict(X_final)

    return predictions, y_true


if __name__ == "__main__":
    preds, real = predict_churn('../data/test/test_sample.csv', '../models/churn_model_v1.pkl')

    # Comparar resultados
    if real is not None:
        from sklearn.metrics import accuracy_score

        print(f"Inference Accuracy: {accuracy_score(real, preds):.4f}")
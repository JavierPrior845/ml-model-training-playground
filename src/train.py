import os
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, TargetEncoder, StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


def load_and_preprocess(path):
    # 1. Data loading
    df = pd.read_csv(path)
    print(f"Data loaded: {df.shape}")

    # 2. Drop CustomerID (Remove noise)
    if 'customerID' in df.columns:
        df = df.drop('customerID', axis=1)
        print("Column 'customerID' removed successfully.")

    # 3. TotalCharges cleaning (Conversion and handling whitespace/nulls)
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)
    print("TotalCharges converted to numeric and nulls filled with 0.")
    return df

def split_data(data):
    # 6. Train/Test Split
    x = data.drop('Churn', axis=1)
    y = data['Churn']

    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.4, random_state=42) # 60x100 train
    x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=0.5, random_state=324) # 20x100 validation | 20x100 test
    return x_train, x_val, x_test, y_train, y_val, y_test


def encode_features(x_train, x_val, x_test, y_train):
    ohe_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']
    te_cols = ['Contract', 'PaymentMethod']

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    te = TargetEncoder(smooth="auto")

    x_train_ohe = ohe.fit_transform(x_train[ohe_cols])
    x_train_te = te.fit_transform(x_train[te_cols], y_train)

    # Transformamos val y test usando los objetos ya "ajustados"
    x_val_ohe = ohe.transform(x_val[ohe_cols])
    x_val_te = te.transform(x_val[te_cols])
    x_test_ohe = ohe.transform(x_test[ohe_cols])
    x_test_te = te.transform(x_test[te_cols])

    # Función interna para combinar (como ya tenías)
    def combine(orig_df, ohe_arr, te_arr):
        num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
        return np.hstack([orig_df[num_cols].values, ohe_arr, te_arr])

    # RETORNO: Los datos procesados Y los objetos encoders
    return (combine(x_train, x_train_ohe, x_train_te),
            combine(x_val, x_val_ohe, x_val_te),
            combine(x_test, x_test_ohe, x_test_te),
            ohe, te)


def scale_features(x_train, x_val, x_test):
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)
    return x_train_scaled, x_val_scaled, x_test_scaled, scaler  # Retornamos el objeto scaler


def apply_pca(x_train, x_val, x_test, n_components=0.95):
    pca = PCA(n_components=n_components)
    x_train_pca = pca.fit_transform(x_train)
    x_val_pca = pca.transform(x_val)
    x_test_pca = pca.transform(x_test)
    return x_train_pca, x_val_pca, x_test_pca, pca  # Retornamos el objeto pca


def train_and_evaluate(x_train, y_train, x_val, y_val):
    # 1. Initialize the model
    # We use random_state for reproducibility
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42, class_weight='balanced')

    # 2. Train (Fit) the model
    print("Training the Random Forest model...")
    model.fit(x_train, y_train)

    # 3. Predict on Validation set
    y_pred_val = model.predict(x_val)

    # 4. Evaluation Metrics
    print("\n--- Validation Performance ---")
    print(f"Accuracy Score: {accuracy_score(y_val, y_pred_val):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_val, y_pred_val))

    return model


def export_model_assets(model, ohe, te, scaler, pca):
    # Create a directory for the model if it doesn't exist
    os.makedirs('../models', exist_ok=True)

    # We save a dictionary containing the model AND the preprocessing steps
    artifacts = {
        'model': model,
        'ohe': ohe,
        'te': te,
        'scaler': scaler,
        'pca': pca
    }
    joblib.dump(artifacts, '../models/churn_model_v1.pkl')
    print("Assets exported successfully to /models/churn_model_v1.pkl")

def save_test_data(x_test, y_test):
    test_data = x_test.copy()
    test_data['Churn'] = y_test

    os.makedirs('../data/test', exist_ok=True)
    test_data.to_csv('../data/test/test_sample.csv', index=False)
    print("Test sample saved to ../data/test/test_sample.csv")

if __name__ == "__main__":
    DATA_PATH = '../data/WA_Fn-UseC_-Telco-Customer-Churn.csv'
    data = load_and_preprocess(DATA_PATH)
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(data)

    x_train_e, x_val_e, x_test_e, ohe, te = encode_features(x_train, x_val, x_test, y_train)

    x_train_s, x_val_s, x_test_s, scaler = scale_features(x_train_e, x_val_e, x_test_e)

    x_train_final, x_val_final, x_test_final, pca = apply_pca(x_train_s, x_val_s, x_test_s)

    # Targets
    y_train_num = y_train.map({'Yes': 1, 'No': 0}) if y_train.dtype == 'O' else y_train
    y_val_num = y_val.map({'Yes': 1, 'No': 0}) if y_val.dtype == 'O' else y_val

    # Training
    churn_model = train_and_evaluate(x_train_final, y_train_num, x_val_final, y_val_num)

    save_test_data(x_test, y_test)
    # Final export
    export_model_assets(churn_model, ohe, te, scaler, pca)

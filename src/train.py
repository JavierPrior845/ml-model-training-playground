import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, TargetEncoder

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

    '''
    # Point 4 and 5 I'm not sure if they must be here or after data split
    # 4. Label Encoding (Binary variables: Yes/No -> 1/0)
    binary_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col == 'gender':
            df[col] = df[col].map({'Female': 1, 'Male': 0})
        else:
            df[col] = df[col].map({'Yes': 1, 'No': 0})
    print("Label Encoding completed for binary variables.")

    # 5. One-Hot Encoding (Multiclass variables using Dummies)
    multi_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                  'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                  'Contract', 'PaymentMethod']

    # drop_first=True to avoid the dummy variable trap and reduce multicollinearity
    df_final = pd.get_dummies(df, columns=multi_cols, drop_first=True)
    print(f"One-Hot Encoding completed. Total columns: {len(df_final.columns)}")
    '''
    return df

def split_data(data):
    # 6. Train/Test Split
    x = data.drop('Churn', axis=1)
    y = data['Churn']

    x_train, x_rest, y_train, y_rest = train_test_split(x, y, test_size=0.4, random_state=42) # 60x100 train
    x_val, x_test, y_val, y_test = train_test_split(x_rest, y_rest, test_size=0.5, random_state=324) # 20x100 validation | 20x100 test
    return x_train, x_val, x_test, y_train, y_val, y_test

def encding_data():
    ohe_cols = ['gender', 'Partner', 'Dependents', 'PhoneService', 'MultipleLines',
                'InternetService', 'OnlineSecurity', 'OnlineBackup', 'DeviceProtection',
                'TechSupport', 'StreamingTV', 'StreamingMovies', 'PaperlessBilling']

    # Using TargetEncoder for columns where it might be more efficient (e.g., PaymentMethod or Contract)
    # TargetEncoder is great for high-cardinality or highly predictive categories
    te_cols = ['Contract', 'PaymentMethod']

    # A) Apply OneHotEncoder
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore', drop='first')
    # We fit only on train to avoid leakage
    x_train_ohe = ohe.fit_transform(x_train[ohe_cols])
    x_val_ohe = ohe.transform(x_val[ohe_cols])
    x_test_ohe = ohe.transform(x_test[ohe_cols])

    # B) Apply TargetEncoder (Requires y_train!)
    te = TargetEncoder(smooth="auto")
    # It learns the relation between categories and the Churn probability
    x_train_te = te.fit_transform(x_train[te_cols], y_train)
    x_val_te = te.transform(x_val[te_cols])
    x_test_te = te.transform(x_test[te_cols])

    print("\nEncoding finalized using Sklearn Encoders.")
    print(f"Training set shapes: OHE: {x_train_ohe.shape}, TE: {x_train_te.shape}")
    print("Data preparation finalized.")

if __name__ == "__main__":
    # Data path - Adjust this to your local directory
    DATA_PATH = 'data/WA_Fn-UseC_-Telco-Customer-Churn.csv'

    # Run preprocessing pipeline
    data = load_and_preprocess(DATA_PATH)

    # Split data
    x_train, x_val, x_test, y_train, y_val, y_test = split_data(data)

    print("\nData preparation finalized.")
    print(f"Training set: {x_train.shape[0]} samples | Test set: {x_test.shape[0]} samples")
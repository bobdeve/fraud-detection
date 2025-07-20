# utils/data_utils.py

import pandas as pd
import numpy as np
import ipaddress
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

def load_data():
    """Load datasets from raw CSV files."""
    fraud_df = pd.read_csv('../data/raw/Fraud_Data.csv')
    credit_df = pd.read_csv('../data/raw/creditcard.csv')
    ip_df = pd.read_csv('../data/raw/IpAddress_to_Country.csv')
    return fraud_df, credit_df, ip_df

def handle_missing_values(fraud_df, credit_df, ip_df):
    """Impute or drop missing values in datasets."""
    fraud_df['age'] = pd.to_numeric(fraud_df['age'], errors='coerce')
    fraud_df['age'] = fraud_df['age'].fillna(fraud_df['age'].median())
    fraud_df['browser'] = fraud_df['browser'].fillna(fraud_df['browser'].mode()[0])
    fraud_df = fraud_df.dropna()
    
    credit_df = credit_df.dropna()
    ip_df = ip_df.dropna()
    
    return fraud_df, credit_df, ip_df

def clean_data(fraud_df, credit_df, ip_df):
    """Remove duplicates and correct data types."""
    fraud_df = fraud_df.drop_duplicates()
    credit_df = credit_df.drop_duplicates()
    ip_df = ip_df.drop_duplicates()
    
    fraud_df['signup_time'] = pd.to_datetime(fraud_df['signup_time'])
    fraud_df['purchase_time'] = pd.to_datetime(fraud_df['purchase_time'])
    
    ip_df['lower_bound_ip_address'] = pd.to_numeric(ip_df['lower_bound_ip_address'], errors='coerce')
    ip_df['upper_bound_ip_address'] = pd.to_numeric(ip_df['upper_bound_ip_address'], errors='coerce')
    
    return fraud_df, credit_df, ip_df

def add_time_features(fraud_df):
    """Add hour_of_day, day_of_week, and time_since_signup features."""
    fraud_df['hour_of_day'] = fraud_df['purchase_time'].dt.hour
    fraud_df['day_of_week'] = fraud_df['purchase_time'].dt.dayofweek
    fraud_df['time_since_signup'] = (fraud_df['purchase_time'] - fraud_df['signup_time']).dt.total_seconds() / 3600
    return fraud_df

def add_transaction_frequency(fraud_df):
    """Add transaction count per user as a feature."""
    user_freq = fraud_df.groupby('user_id').size().rename('transaction_count')
    fraud_df = fraud_df.merge(user_freq, on='user_id')
    return fraud_df

def ip_to_int(ip_str):
    """Convert IP string to integer. Return None if invalid."""
    try:
        return int(ipaddress.IPv4Address(ip_str))
    except Exception:
        return None  # Invalid IP format

def map_ip_to_country(ip, ip_df):
    if ip == -1:
        return 'Unknown'
    match = ip_df[(ip_df['lower_bound_ip_address'] <= ip) & (ip_df['upper_bound_ip_address'] >= ip)]
    return match['country'].values[0] if not match.empty else 'Unknown'





def merge_ip_country(fraud_df, ip_df):
    fraud_df['ip_int'] = fraud_df['ip_address'].apply(ip_to_int)

    # Fill invalid IPs with -1 so they still exist
    fraud_df['ip_int'] = fraud_df['ip_int'].fillna(-1)

    fraud_df['country'] = fraud_df['ip_int'].apply(lambda x: map_ip_to_country(x, ip_df))
    return fraud_df




def encode_categoricals(fraud_df, categorical_cols):
    """One-hot encode categorical columns and append to dataframe."""

    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')

    ohe_df = pd.DataFrame(ohe.fit_transform(fraud_df[categorical_cols]), columns=ohe.get_feature_names_out())
    fraud_encoded = pd.concat([fraud_df.reset_index(drop=True), ohe_df], axis=1)
    return fraud_encoded




def transform_and_balance(X, y):
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from imblearn.over_sampling import SMOTE
    import numpy as np

    # Step 1: Select numeric features
    X_numeric = X.select_dtypes(include=[np.number])

    # Step 2: Split
    X_train, X_test, y_train, y_test = train_test_split(
        X_numeric, y, test_size=0.3, random_state=42, stratify=y
    )

    # Step 3: Drop columns that are entirely NaN (like 'time_since_last_transaction')
    cols_all_nan = X_train.columns[X_train.isnull().all()].tolist()
    if cols_all_nan:
        print("🚨 Dropping columns with all NaNs:", cols_all_nan)
        X_train = X_train.drop(columns=cols_all_nan)
        X_test = X_test.drop(columns=cols_all_nan)

    # Step 4: Replace inf with NaN
    X_train = X_train.replace([np.inf, -np.inf], np.nan)
    X_test = X_test.replace([np.inf, -np.inf], np.nan)

    # Step 5: Fill remaining NaNs with median
    X_train = X_train.fillna(X_train.median())
    X_test = X_test.fillna(X_test.median())

    # Debug check
    print("✅ Final NaNs in X_train:\n", X_train.isnull().sum()[X_train.isnull().sum() > 0])
    print("✅ Shape before SMOTE:", X_train.shape)

    # Step 6: SMOTE
    smote = SMOTE(random_state=42)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

    # Step 7: Scale
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_smote)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train_smote, y_test






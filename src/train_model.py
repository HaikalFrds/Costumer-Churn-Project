# Library
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib

# Main path
BASE_DIR = os.getcwd()
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "churn_cleaned.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")

# Load data
def load_data(path):

    print("Loading cleaned data...")
    df = pd.read_csv(path)

    return df

# Feature Engineering
def feature_engineering(df):

    print("Running feature engineering...")

    df = df.copy()

    # Active Ratio
    df['Active_ratio'] = (df['Active_subscribers'] / df['Total_SUBs'])

    # Inactive Ratio
    df['Inactive_ratio'] = (df['Not_Active_subscribers'] / df['Total_SUBs'])

    # Revenue per Subscriber
    df['Revenue_per_sub'] = (df['TotalRevenue'] / df['Total_SUBs'])

    # Revenue Mix (Mobile Ratio)
    df['Mobile_rev_ratio'] = (df['AvgMobileRevenue'] / df['TotalRevenue'])

    # Rev Active
    df['Rev_active'] = (df['ARPU'] * df['Active_ratio'])

    # Rev Inactive
    df['Rev_inactive'] = (df['ARPU'] * df['Inactive_ratio'])

    # Rev Inactive
    df['Mobile_active'] = (df['AvgMobileRevenue'] * df['Active_ratio'])

    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    print("Feature engineering completed")

    return df

def standardize_categories(df):

    print("Standardizing categories...")

    df = df.copy()

    # Standarisasi typo value
    df['CRM_PID_Value_Segment'] = df['CRM_PID_Value_Segment'].replace({
        'Sliver': 'Silver'
    })

    # Elaborasi value di luar konteks
    df['CRM_PID_Value_Segment'] = df['CRM_PID_Value_Segment'].replace({
        'SME': 'Iron',
        'SE': 'Iron',
        'Lead': 'Iron'
    })

    print("Category standardized")

    return df

def encode_features(df):

    print("Encoding features...")

    df = df.copy()

    tier_map = {
        'Iron': 0,
        'Bronze': 1,
        'Silver': 2,
        'Gold': 3,
        'Platinum': 4
    }

    df['CRM_PID_Value_Segment'] = df['CRM_PID_Value_Segment'].map(tier_map).fillna(0)

    df = pd.get_dummies(
        df,
        columns=['EffectiveSegment'],
        drop_first=True
    )

    df['CHURN'] = df['CHURN'].map({
        'No': 0,
        'Yes': 1
    })

    print("Encoding completed")

    return df

def drop_unused_columns(df):

    print("Dropping unused columns...")

    df = df.copy()

    df = df.drop(columns=[
        'PID',
        'Billing_ZIP',
        'KA_name',
        'ARPU_segment',
        'Active_subscribers',
        'Not_Active_subscribers',
        'Total_SUBs',
        'TotalRevenue',
        'AvgFIXRevenue'
    ], errors='ignore')

    print("Unused columns dropped")

    return df

def split_feature_target(df):

    print("Splitting feature and target...")

    df = df.copy()

    X = df.drop(columns=['CHURN'])
    y = df['CHURN']

    print("Split completed")

    return X, y

def split_train_test(X, y):

    print("Creating train test split...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print("Train test split completed")

    return X_train, X_test, y_train, y_test


def scale_features(X_train, X_test):

    print("Scaling features...")

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print("Scaling completed")

    return X_train_scaled, X_test_scaled, scaler

def train_logistic_model(X_train_scaled, y_train):

    print("Training Logistic Regression...")

    lr = LogisticRegression(class_weight='balanced', C=1, solver='saga',max_iter=3000, n_jobs=-1)

    # Train model
    lr.fit(X_train_scaled, y_train)

    print("Training shape:", X_train_scaled.shape)
    
    print("Training completed")

    return lr

def evaluate_model(model, X_test_scaled, y_test, threshold = 0.475):

    print("Evaluating Logistic Regression...")

    # probability
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    # inisiate threshold
    y_pred = (y_prob >= threshold).astype(int)

    # metrics
    acc = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    print("Accuracy:", acc)
    print()
    print("Recall churn:", report['1']['recall'])
    print()
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print()
    print("Confusion Matrix:")
    print(cm)

    print("Evaluation completed")

    return y_pred, y_prob

def save_model(model, scaler, feature_columns, threshold):

    print("Saving model...")

    model_dir = os.path.join(BASE_DIR, "models")
    os.makedirs(model_dir, exist_ok=True)

    model_path = os.path.join(model_dir, "churn_model.pkl")

    joblib.dump(model, model_path)
    joblib.dump(scaler, os.path.join(model_dir, "scaler.pkl"))
    joblib.dump(list(feature_columns), os.path.join(model_dir, "feature_columns.pkl"))
    joblib.dump(threshold, os.path.join(model_dir, "threshold.pkl"))

    print("Model saved at:", model_path)

# Execute
def main():

    THRESHOLD = 0.475

    df_churn = load_data(PROCESSED_PATH)

    df_churn = feature_engineering(df_churn)
    df_churn = standardize_categories(df_churn)
    df_churn = encode_features(df_churn)
    df_churn = drop_unused_columns(df_churn)

    X, y = split_feature_target(df_churn)

    X_train, X_test, y_train, y_test = split_train_test(X, y)

    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

    model = train_logistic_model(X_train_scaled, y_train)

    evaluate_model(model, X_test_scaled, y_test, threshold = THRESHOLD)

    save_model(model, scaler, X_train.columns, THRESHOLD)


if __name__ == "__main__":
    main()
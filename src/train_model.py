# Library
import os
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
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
    df = df['Active_ratio'] = (df['Active_subscribers'] / df['Total_SUBs'])

    # Inactive Ratio
    df = df['Inactive_ratio'] = (df['Not_Active_subscribers'] / df['Total_SUBs'])

    # Revenue per Subscriber
    df = df['Revenue_per_sub'] = (df['TotalRevenue'] / df['Total_SUBs'])

    # Revenue Mix (Mobile Ratio)
    df = df['Mobile_rev_ratio'] = (df['AvgMobileRevenue'] / df['TotalRevenue'])

    # Rev Active
    df = df['Rev_active'] = (df['ARPU'] * df['Active_ratio'])

    # Rev Inactive
    df = df['Rev_inactive'] = (df['ARPU'] * df['Inactive_ratio'])

    # Rev Inactive
    df = df['Mobile_active'] = (df['AvgMobileRevenue'] * df['Active_ratio'])

    print("Feature engineering completed")

    return df

def standardize_categories(df):

    print("Standardizing categories...")

    df = df.copy()

    df['CRM_PID_Value_Segment'] = df['CRM_PID_Value_Segment'].replace({
        'Sliver': 'Silver'
    })

    df['CRM_PID_Value_Segment'] = df['CRM_PID_Value_Segment'].replace({
        'SME': 'Iron',
        'SE': 'Iron',
        'Lead': 'Iron'
    })

    print("Category standardized")

    return df

def main():

    df_churn = load_data(PROCESSED_PATH)

    print("Data shape:", df_churn.shape)


if __name__ == "__main__":
    main()
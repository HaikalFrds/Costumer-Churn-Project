# Library
import os
import pandas as pd

# Main path declaration
BASE_DIR = os.getcwd()
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "Baza customer Telecom v2.csv")
PROCESSED_PATH = os.path.join(BASE_DIR, "data", "processed", "churn_cleaned.csv")

# Load dataset
def load_data(path):
    print("Loading raw data...")
    return pd.read_csv(path)

# Cleaning data
## Clean column name
def clean_columns(df):
    df.columns = df.columns.str.strip()
    return df

## Checking missing value
def handle_missing_values(df):

    print("Checking missing values...")

    # drop column extreme missing
    df = df.drop(columns=['Suspended_subscribers'], errors='ignore')

    # fill numeric
    df['Not_Active_subscribers'] = df['Not_Active_subscribers'].fillna(0)

    # fill categorical
    df['CRM_PID_Value_Segment'] = df['CRM_PID_Value_Segment'].fillna(
        df['CRM_PID_Value_Segment'].mode()[0]
    )

    df['Billing_ZIP'] = df['Billing_ZIP'].fillna(
        df['Billing_ZIP'].mode()[0]
    )

    # prevent division by zero
    df['Total_SUBs'] = df['Total_SUBs'].replace(0, 1)

    # fill ARPU
    df['ARPU'] = df['ARPU'].fillna(
        df['TotalRevenue'] / df['Total_SUBs']
    )

    print("Missing values handled")

    return df

## Checking duplicate data
def handle_duplicates(df):

    df = df.drop_duplicates()
    print("Duplicate handled")

    return df

## Data Ccnsistency
def handle_consistency(df):

    print("Checking data consistency...")

    consistency_check = (
        df['Active_subscribers'] +
        df['Not_Active_subscribers']
    ) - df['Total_SUBs']

    mask = consistency_check != 0

    df.loc[mask, 'Total_SUBs'] = (
        df.loc[mask, 'Active_subscribers'] +
        df.loc[mask, 'Not_Active_subscribers']
    )

    print("Consistency fixed")

    return df

# Save cleaned data
def save_data(df, path):

    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)

    print("Cleaned data saved")

# Cleaning pipeline
def main():

    df_churn_raw = load_data(RAW_PATH)

    df_churn = df_churn_raw.copy()
    df_churn = clean_columns(df_churn)

    df_churn = handle_missing_values(df_churn)
    df_churn = handle_duplicates(df_churn)
    df_churn = handle_consistency(df_churn)

    save_data(df_churn, PROCESSED_PATH)

# Execute
if __name__ == "__main__":
    main()
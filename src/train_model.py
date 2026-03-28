# Library
import os
import numpy as np
import pandas as pd

# Main path declaration
BASE_DIR = os.getcwd()
data_path = os.path.join(BASE_DIR, "data", "raw", "Baza customer Telecom v2.csv")

# Load dataset
df_churn_raw = pd.read_csv(data_path)

## Make Copy
df_churn = df_churn_raw.copy()

# Cleaning data
## Clean column name
df_churn.columns = df_churn.columns.str.strip()

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

# Cleaning pipeline
df_churn = handle_missing_values(df_churn)
df_churn = handle_duplicates(df_churn)
df_churn = handle_consistency(df_churn)

# Create folder processed
os.makedirs("data/processed", exist_ok=True)

# Save cleaned data
df_churn.to_csv("data/processed/churn_cleaned.csv", index=False)

print("Cleaned data saved")

# feature engineering

# split data

# training logistic regresion

# save model
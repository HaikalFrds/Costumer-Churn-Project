# Library
import os
import pandas as pd
import numpy as np
import joblib

# Main path
BASE_DIR = os.getcwd()

MODEL_PATH = os.path.join(BASE_DIR, "models", "churn_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
FEATURE_PATH = os.path.join(BASE_DIR, "models", "feature_columns.pkl")
THRESHOLD_PATH = os.path.join(BASE_DIR, "models", "threshold.pkl")

NEW_DATA_PATH = os.path.join(BASE_DIR, "data", "new_data", "new_customers.csv")
OUTPUT_PATH = os.path.join(BASE_DIR, "data", "predictions", "churn_predictions.csv")

# Load path
def load_model():

    print("Loading model...")

    model = joblib.load(MODEL_PATH)

    return model

# Load scaler
def load_scaler():

    print("Loading scaler...")

    scaler = joblib.load(SCALER_PATH)

    return scaler

# Load feature columns
def load_feature_columns():

    print("Loading feature columns...")

    feature_columns = joblib.load(FEATURE_PATH)

    return feature_columns

# Load Threshold
def load_threshold():

    print("Loading threshold...")

    threshold = joblib.load(THRESHOLD_PATH)

    return threshold

# Load new data
def load_new_data(path):

    print("Loading new data...")

    df = pd.read_csv(path)

    return df
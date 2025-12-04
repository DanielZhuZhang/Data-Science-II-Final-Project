import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
import os

def load_and_preprocess(filepath):
    """
    Loads the dataset and performs necessary preprocessing:
    - Drops missing values
    - Strips whitespace from column names (Critical fix)
    - Encodes categorical variables
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Could not find {filepath}. Please download it first.")

    df = pd.read_csv(filepath)
    
    # 1. CLEANING: Strip whitespace from column headers to match our list
    df.columns = df.columns.str.strip()
    
    # Drop rows with missing values
    df.dropna(inplace=True)

    # Create a copy for encoding
    df_encoded = df.copy()
    encoders = {}

    # The columns we expect to be text that need converting to numbers
    categorical_cols = [
        'Workout_Type', 
        'Gender', 
        'meal_type', 
        'cooking_method', 
        'diet_type', 
        'Experience_Level',
        'meal_name'
    ]
    
    print("--- Preprocessing Log ---")
    for col in categorical_cols:
        if col in df.columns:
            # Create a NEW encoder for each column
            le = LabelEncoder()
            # Convert to string to handle any mixed types safe-guard
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
            print(f"✓ Encoded column: '{col}'")
        else:
            print(f"⚠ WARNING: Column '{col}' not found in CSV. It will be skipped.")

    return df, df_encoded, encoders

def save_plot(fig, filename):
    """Helper to save plots to a directory"""
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    path = os.path.join('plots', filename)
    fig.savefig(path)
    print(f"Saved plot to {path}")
    plt.close(fig)
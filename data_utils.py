import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess(filepath):
    """
    Loads dataset, handles missing values, and encodes categorical data.
    """
    if not os.path.exists(filepath):
        # Fallback checks
        possible_files = ['gym_members_exercise_tracking.csv', 'data.csv']
        found = False
        for f in possible_files:
            if os.path.exists(f):
                filepath = f
                found = True
                break
        if not found:
            raise FileNotFoundError(f"Could not find {filepath}. Please ensure the CSV is in the folder.")

    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Clean headers
    df.columns = df.columns.str.strip()
    
    # Drop missing
    df.dropna(inplace=True)
    
    # Create encoded version
    df_encoded = df.copy()
    encoders = {}
    
    # Categorical columns to encode
    categorical_cols = [
        'Workout_Type', 'Gender', 'meal_type', 'cooking_method', 
        'diet_type', 'Experience_Level', 'meal_name'
    ]
    
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df_encoded[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    
    return df, df_encoded, encoders

def perform_feature_engineering(df):
    """
    Adds advanced features to improve model prediction power.
    """
    df_eng = df.copy()
    
    # 1. Intensity Index (Calories / Hour)
    if 'Calories_Burned' in df_eng.columns and 'Session_Duration (hours)' in df_eng.columns:
        df_eng['Intensity_Index'] = df_eng['Calories_Burned'] / df_eng['Session_Duration (hours)']
    
    # 2. Heart Rate Efficiency (BPM per Calorie)
    if 'Calories_Burned' in df_eng.columns and 'Avg_BPM' in df_eng.columns:
        df_eng['HR_Efficiency'] = df_eng['Avg_BPM'] / df_eng['Calories_Burned']
    
    # 3. Max Heart Rate Est & Utilization
    if 'Age' in df_eng.columns and 'Avg_BPM' in df_eng.columns:
        df_eng['Max_HR_Est'] = 220 - df_eng['Age']
        df_eng['HR_Utilization'] = df_eng['Avg_BPM'] / df_eng['Max_HR_Est']

    # Handle infinite values
    df_eng.replace([np.inf, -np.inf], 0, inplace=True)
    df_eng.fillna(0, inplace=True)
    
    return df_eng

def save_plot(fig, filename):
    """Saves plot to a 'plots' directory."""
    if not os.path.exists('plots'):
        os.makedirs('plots')
    
    path = os.path.join('plots', filename)
    try:
        fig.tight_layout()
    except:
        pass 
    fig.savefig(path, dpi=300)
    print(f"Saved plot: {path}")
    plt.close(fig)
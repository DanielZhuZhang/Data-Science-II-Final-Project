import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Sklearn Imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler
from mpl_toolkits.mplot3d import Axes3D 

# Local Import
from data_utils import load_and_preprocess, save_plot, perform_feature_engineering

# --- CONFIG ---
DATA_FILE = 'gym_members_exercise_tracking.csv' 

# ==========================================
# PART A: ORIGINAL CONJECTURES (1-5)
# ==========================================

def test_conjecture_1_heart_rate(df):
    """
    Conjecture 1: We can predict Heart Rate using Calories, Duration, and Age.
    """
    print("\n--- CONJECTURE 1: Heart Rate Prediction (Random Forest) ---")
    features = ['Calories_Burned', 'Session_Duration (hours)', 'Age']
    target = 'Avg_BPM'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    preds = rf.predict(X_test)
    score = r2_score(y_test, preds)

    print(f"R^2 Score: {score:.4f}")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=preds, alpha=0.5, ax=ax)
    
    # Perfect prediction line
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    ax.set_xlabel('Actual BPM')
    ax.set_ylabel('Predicted BPM')
    ax.set_title(f'Conjecture 1: Predicting Heart Rate (R^2 = {score:.2f})')
    save_plot(fig, 'conjecture_1_heart_rate.png')

def test_conjecture_2_calories_multivariate(df):
    """
    Conjecture 2: Calories burned can be predicted by Duration, split by Workout Type.
    """
    print("\n--- CONJECTURE 2: Calorie Prediction (Multivariate Viz) ---")
    
    g = sns.lmplot(
        data=df, 
        x='Session_Duration (hours)', 
        y='Calories_Burned', 
        hue='Workout_Type', 
        height=6, 
        aspect=1.5,
        scatter_kws={'alpha': 0.4},
        line_kws={'linewidth': 2}
    )
    plt.title("Conjecture 2: Linear trends of Calorie Burn by Workout Type")
    save_plot(g.fig, 'conjecture_2_calories_lines.png')
    
    # Statistical Check
    X = pd.get_dummies(df[['Session_Duration (hours)', 'Workout_Type']], drop_first=True)
    y = df['Calories_Burned']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LinearRegression()
    model.fit(X_train, y_train)
    print(f"Multivariate Linear Regression R^2: {model.score(X_test, y_test):.4f}")

def test_conjecture_3_classification(df):
    """
    Conjecture 3: Predicting Workout Type based on Duration, Calories, and BPM.
    """
    print("\n--- CONJECTURE 3: Classifying Workout Type ---")
    features = ['Session_Duration (hours)', 'Calories_Burned', 'Avg_BPM']
    target = 'Workout_Type'

    X = df[features]
    y = df[target]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    acc = accuracy_score(y_test, rf.predict(X_test))
    
    print(f"Accuracy (Basic Features): {acc:.4f}")
    
    importances = rf.feature_importances_ * 100 
    
    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=importances, y=features, ax=ax)
    ax.set_xlabel('Relative Importance (%)') 
    ax.set_title('Conjecture 3: Feature Importance (Basic)')
    plt.tight_layout()
    save_plot(fig, 'conjecture_3_feature_importance.png')

def test_conjecture_4_nutrition(df_encoded):
    """
    Conjecture 4: Does meal type or cooking method predict Calories Burned?
    (This is expected to be a 'Negative Result' which is scientifically valid)
    """
    print("\n--- CONJECTURE 4: Nutrition & Meal Analysis ---")
    features = ['Session_Duration (hours)', 'Age', 'meal_type', 'cooking_method', 'rating']
    target = 'Calories_Burned'

    # Filter for columns that actually exist
    valid_features = [f for f in features if f in df_encoded.columns]
    X = df_encoded[valid_features]
    y = df_encoded[target]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_ * 100
    
    print("Conjecture 4 Feature Importances:")
    for name, imp in zip(valid_features, importances):
        print(f"  {name}: {imp:.1f}%")

    fig, ax = plt.subplots(figsize=(12, 7))
    sns.barplot(x=importances, y=valid_features, ax=ax)
    ax.set_xlabel('Relative Importance (%)')
    ax.set_title('Conjecture 4: Impact of Nutrition on Workout Intensity')
    plt.tight_layout()
    save_plot(fig, 'conjecture_4_nutrition_analysis.png')

def test_conjecture_5_3d_clusters(df, encoders, predict_user_stats=None):
    """
    Conjecture 5: 3D Visualization of the data structure.
    """
    print("\n--- CONJECTURE 5: 3D Cluster Visualization ---")
    
    features = ['Session_Duration (hours)', 'Calories_Burned', 'Avg_BPM']
    target = 'Workout_Type'
    
    X = df[features]
    if df[target].dtype == 'object':
         y_codes = pd.factorize(df[target])[0]
    else:
         y_codes = df[target]

    # KNN for fun prediction
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X, y_codes)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Subset for clearer plotting
    subset_idx = np.random.choice(len(df), size=min(2000, len(df)), replace=False)
    X_sub = X.iloc[subset_idx]
    y_sub = y_codes[subset_idx]
    
    scatter = ax.scatter(
        X_sub['Session_Duration (hours)'],
        X_sub['Calories_Burned'],
        X_sub['Avg_BPM'],
        c=y_sub,
        cmap='viridis',
        s=40,
        alpha=0.6,
        edgecolor='w'
    )
    
    # Optional: Plot a specific user
    if predict_user_stats:
        user_arr = np.array([predict_user_stats])
        pred_code = knn.predict(user_arr)[0]
        try:
             pred_name = encoders['Workout_Type'].inverse_transform([pred_code])[0]
        except:
             pred_name = f"ID {pred_code}"
             
        print(f"  User Prediction {predict_user_stats} -> {pred_name}")
        ax.scatter(user_arr[0,0], user_arr[0,1], user_arr[0,2], c='red', marker='X', s=300, label='USER', zorder=10)

    ax.set_xlabel('Duration')
    ax.set_ylabel('Calories')
    ax.set_zlabel('BPM')
    ax.set_title('Conjecture 5: 3D Workout Clusters')
    save_plot(fig, 'conjecture_5_3d_clusters.png')

def run_kmeans_discovery(df):
    """
    Unsupervised Learning: Finding 'Archetypes'
    """
    print("\n--- K-MEANS: Unsupervised Archetype Discovery ---")
    features = ['Session_Duration (hours)', 'Calories_Burned']
    X = df[features].copy()
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(X_scaled)
    
    df_plot = df.copy()
    df_plot['Cluster_ID'] = clusters
    
    fig, ax = plt.subplots(figsize=(10, 7))
    sns.scatterplot(data=df_plot, x='Session_Duration (hours)', y='Calories_Burned', 
                    hue='Cluster_ID', palette='viridis', s=60, alpha=0.7, ax=ax)
    
    centroids = scaler.inverse_transform(kmeans.cluster_centers_)
    ax.scatter(centroids[:, 0], centroids[:, 1], c='red', s=200, marker='X', label='Centroids')
    
    ax.set_title('K-Means: Discovered Gym Archetypes')
    save_plot(fig, 'kmeans_archetypes.png')

# ==========================================
# PART B: PREDICTION ALGORITHM
# ==========================================

def run_workout_prediction_demo(df_encoded, encoders):
    """
    Trains a model and allows for 'Live' prediction of workout types.
    Calculates engineered features (Intensity, HR Efficiency) on the fly for inputs.
    """
    print("\n================================================")
    print("   FINAL DELIVERABLE: WORKOUT PREDICTION ALG")
    print("================================================")
    
    # 1. Train the best model (Random Forest with Feature Engineering)
    df_eng = perform_feature_engineering(df_encoded)
    
    features = ['Session_Duration (hours)', 'Calories_Burned', 'Avg_BPM', 
                'Intensity_Index', 'HR_Utilization', 'Age']
    target = 'Workout_Type'
    
    X = df_eng[features]
    y = df_eng[target]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Using a deeper tree for better classification accuracy
    clf = RandomForestClassifier(n_estimators=100, max_depth=12, random_state=42)
    clf.fit(X_train, y_train)
    
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"Algorithm Trained. Model Accuracy: {acc:.2%}")
    
    # 2. Define the Helper function to predict for a single person
    def predict_new_workout(duration, calories, bpm, age=25):
        # We must manually calculate the engineered features for the input
        # (Same math as in data_utils.py)
        intensity = calories / duration if duration > 0 else 0
        max_hr = 220 - age
        hr_util = bpm / max_hr
        
        # Create a mini dataframe for the model
        input_data = pd.DataFrame([{
            'Session_Duration (hours)': duration,
            'Calories_Burned': calories,
            'Avg_BPM': bpm,
            'Intensity_Index': intensity,
            'HR_Utilization': hr_util,
            'Age': age
        }])
        
        # Predict
        pred_code = clf.predict(input_data)[0]
        # Convert code back to string (e.g., 0 -> 'Yoga')
        pred_name = encoders['Workout_Type'].inverse_transform([pred_code])[0]
        return pred_name

    # 3. Run Demo Scenarios
    print("\n--- Live Prediction Scenarios ---")
    
    # Scenario A: Low intensity, low heart rate
    pred_a = predict_new_workout(duration=1.0, calories=150, bpm=90)
    print(f"1. Input: 1.0 hr, 150 cal, 90 BPM  --> Algorithm Predicts: {pred_a}")
    
    # Scenario B: High intensity, short duration
    pred_b = predict_new_workout(duration=0.5, calories=400, bpm=165)
    print(f"2. Input: 0.5 hr, 400 cal, 165 BPM --> Algorithm Predicts: {pred_b}")

    # Scenario C: Long duration, medium intensity
    pred_c = predict_new_workout(duration=1.5, calories=800, bpm=145)
    print(f"3. Input: 1.5 hr, 800 cal, 145 BPM --> Algorithm Predicts: {pred_c}")

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("--- Starting Full Project Pipeline ---")
    
    # 1. Load Data
    try:
        df, df_encoded, encoders = load_and_preprocess(DATA_FILE)
        print(f"Data Loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"Error: {e}")
        return

    # 2. Run Original Conjectures (Required)
    test_conjecture_1_heart_rate(df)
    test_conjecture_2_calories_multivariate(df)
    test_conjecture_3_classification(df)
    test_conjecture_4_nutrition(df_encoded)
    
    # 3. Visuals
    test_conjecture_5_3d_clusters(df, encoders, predict_user_stats=[1.5, 800, 140])
    run_kmeans_discovery(df)
    
    # 4. Final Prediction Algorithm
    run_workout_prediction_demo(df_encoded, encoders)
    
    print("\n--- Analysis Complete. Check 'plots' folder. ---")

if __name__ == "__main__":
    main()
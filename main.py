import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import r2_score, accuracy_score, classification_report
from data_utils import load_and_preprocess, save_plot

# --- CONFIGURATION ---
DATA_FILE = 'data.csv'

# ==========================================
# CONJECTURE 1: Heart Rate Prediction
# ==========================================

def test_conjecture_1_v1_linear(df):
    """
    Conjecture 1 (Version 1): Linear Regression.
    Expected Result: Poor performance (horizontal line) due to non-linearity.
    """
    print("\n--- CONJECTURE 1 (V1): Linear Regression ---")
    features = ['Calories_Burned', 'Session_Duration (hours)', 'Age']
    target = 'Avg_BPM'

    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    print(f"R^2 Score: {score:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.regplot(x=y_test, y=preds, scatter_kws={'alpha':0.5}, line_kws={'color':'red'}, ax=ax)
    ax.set_title(f'C1 (V1): Linear Regression (R^2 = {score:.2f})')
    save_plot(fig, 'c1_v1_linear.png')

def test_conjecture_1_v2_rf(df):
    """
    Conjecture 1 (Version 2): Random Forest Regressor.
    Expected Result: Better performance by capturing non-linear relationships.
    """
    print("\n--- CONJECTURE 1 (V2): Random Forest (Non-Linear) ---")
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

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=y_test, y=preds, alpha=0.5, ax=ax)
    # Perfect prediction line
    min_val = min(y_test.min(), preds.min())
    max_val = max(y_test.max(), preds.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual BPM')
    ax.set_ylabel('Predicted BPM')
    ax.set_title(f'C1 (V2): Random Forest (R^2 = {score:.2f})')
    save_plot(fig, 'c1_v2_random_forest.png')

# ==========================================
# CONJECTURE 2: Calorie Prediction
# ==========================================

def test_conjecture_2_v1_simple(df):
    """
    Conjecture 2 (Version 1): Simple Linear Regression (Duration only).
    Expected Result: "Banding" issues in the residual plot.
    """
    print("\n--- CONJECTURE 2 (V1): Simple Linear Regression ---")
    feature = 'Session_Duration (hours)'
    target = 'Calories_Burned'

    X = df[[feature]]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    print(f"R^2 Score: {score:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(x=X_test[feature], y=y_test, label='Actual', alpha=0.6, ax=ax)
    sort_idx = X_test[feature].argsort()
    sns.lineplot(x=X_test[feature].iloc[sort_idx], y=preds[sort_idx], color='red', label='Predicted', ax=ax)
    ax.set_title(f'C2 (V1): Simple Regression (R^2 = {score:.2f})')
    save_plot(fig, 'c2_v1_simple.png')

def test_conjecture_2_v2_multivariate(df):
    """
    Conjecture 2 (Version 2): Multivariate with Workout Type.
    Expected Result: Explains the 'bands' by adding Workout_Type as a feature.
    """
    print("\n--- CONJECTURE 2 (V2): Multivariate Regression (w/ Workout Type) ---")
    
    # 1. VISUALIZATION: Show the bands colored by type
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(data=df, x='Session_Duration (hours)', y='Calories_Burned', 
                    hue='Workout_Type', alpha=0.6, ax=ax)
    ax.set_title('C2 (V2): Duration vs Calories (Colored by Workout Type)')
    save_plot(fig, 'c2_v2_colored_bands.png')

    # 2. MODELING: One-Hot Encoding to include Workout_Type
    X = pd.get_dummies(df[['Session_Duration (hours)', 'Workout_Type']], drop_first=True)
    y = df['Calories_Burned']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    score = r2_score(y_test, preds)

    print(f"R^2 Score: {score:.4f} (Improved significantly?)")

# ==========================================
# CONJECTURE 3: Classification (Workout Type)
# ==========================================

def test_conjecture_3(df):
    """
    Conjecture 3: Random Forest Classifier.
    Output: Feature importance in %.
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
    
    print(f"Accuracy: {acc:.4f}")
    
    # Feature Importance in Percent
    importances = rf.feature_importances_ * 100 
    
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.barplot(x=importances, y=features, ax=ax)
    ax.set_xlabel('Relative Importance (%)') 
    ax.set_title('C3: Feature Importance (Workout Prediction)')
    save_plot(fig, 'c3_feature_importance.png')

# ==========================================
# CONJECTURE 4: Nutrition Analysis
# ==========================================

def test_conjecture_4(df_encoded):
    """
    Conjecture 4: Meal relevance.
    Output: Feature importance in %.
    """
    print("\n--- CONJECTURE 4: Meal Data Relevance ---")
    features = ['Session_Duration (hours)', 'Age', 'meal_type', 'cooking_method', 'rating']
    target = 'Calories_Burned'

    # Check columns exist
    valid_features = [f for f in features if f in df_encoded.columns]
    X = df_encoded[valid_features]
    y = df_encoded[target]

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X, y)

    importances = rf.feature_importances_ * 100
    
    print("Feature Importances (%):")
    for name, imp in zip(valid_features, importances):
        print(f"{name}: {imp:.1f}%")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=importances, y=valid_features, ax=ax)
    ax.set_xlabel('Relative Importance (%)')
    ax.set_title('C4: Impact on Calories Burned (Nutrition vs Fitness)')
    save_plot(fig, 'c4_meal_analysis.png')

# ==========================================
# MAIN EXECUTION
# ==========================================

def main():
    print("Loading data...")
    try:
        df, df_encoded, _ = load_and_preprocess(DATA_FILE)
        print(f"Data Loaded. Shape: {df.shape}")
    except Exception as e:
        print(f"Error: {e}")
        return

    # Run all versions
    test_conjecture_1_v1_linear(df)
    test_conjecture_1_v2_rf(df)
    
    test_conjecture_2_v1_simple(df)
    test_conjecture_2_v2_multivariate(df)
    
    test_conjecture_3(df)
    test_conjecture_4(df_encoded)

    print("\nAnalysis Complete. Check the 'plots' folder.")

if __name__ == "__main__":
    main()
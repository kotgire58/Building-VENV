import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import pickle
import json

def generate_synthetic_data(n_samples=5000):
    """Generate synthetic training data for the DDA model"""
    np.random.seed(42)
    
    data = []
    
    for _ in range(n_samples):
        # Player skill level (hidden variable, 0-1)
        skill_level = np.random.random()
        
        # Game difficulty parameters
        target_size = np.random.randint(20, 80)
        target_speed = np.random.uniform(1, 10)
        difficulty_level = np.random.uniform(0.1, 1.0)
        
        # Performance metrics influenced by skill and difficulty
        base_accuracy = skill_level * 0.7 + 0.3
        difficulty_penalty = (target_speed / 10) * (50 / target_size) * difficulty_level
        
        hit_accuracy = max(0.1, min(1.0, base_accuracy - difficulty_penalty * 0.5 + np.random.normal(0, 0.1)))
        
        # Reaction time influenced by difficulty and skill
        base_reaction = 0.3 + (1 - skill_level) * 0.7
        reaction_time = base_reaction + difficulty_penalty * 0.3 + np.random.normal(0, 0.05)
        reaction_time = max(0.15, min(2.0, reaction_time))
        
        # Streak and combo
        streak = int(hit_accuracy * 20 * np.random.uniform(0.5, 1.5))
        max_streak = streak + np.random.randint(0, 10)
        combo_multiplier = 1 + (streak / 10)
        
        # Performance score
        score = int((hit_accuracy * 1000 + (1/reaction_time) * 500) * combo_multiplier)
        
        # Recent performance
        recent_hits = int(hit_accuracy * 10 + np.random.randint(-2, 3))
        recent_misses = 10 - recent_hits
        
        data.append({
            'hit_accuracy': hit_accuracy,
            'avg_reaction_time': reaction_time,
            'current_streak': streak,
            'max_streak': max_streak,
            'score': score,
            'target_size': target_size,
            'target_speed': target_speed,
            'difficulty_level': difficulty_level,
            'recent_hits': recent_hits,
            'recent_misses': recent_misses,
            'combo_multiplier': combo_multiplier,
            'time_played': np.random.randint(10, 300),
            'total_clicks': np.random.randint(50, 1000),
            'next_reaction_time': reaction_time + np.random.normal(0, 0.02)  # Target variable
        })
    
    return pd.DataFrame(data)

def create_features(df):
    """Create additional engineered features"""
    df = df.copy()
    
    # Performance ratios with safe division
    df['hit_miss_ratio'] = df['recent_hits'] / (df['recent_misses'] + 1)
    df['score_per_click'] = df['score'] / (df['total_clicks'] + 1)
    df['streak_ratio'] = df['current_streak'] / (df['max_streak'] + 1)
    
    # Difficulty interaction features
    df['difficulty_size_interaction'] = df['difficulty_level'] * (50 / df['target_size'])
    df['difficulty_speed_interaction'] = df['difficulty_level'] * df['target_speed']
    
    # Performance consistency
    df['accuracy_reaction_product'] = df['hit_accuracy'] * df['avg_reaction_time']
    
    # Handle any potential infinities or NaN values
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Clip extreme values
    for col in df.columns:
        if df[col].dtype in ['float64', 'int64']:
            df[col] = np.clip(df[col], -1e10, 1e10)
    
    return df

def train_dda_model():
    """Train the Dynamic Difficulty Adjustment model"""
    print("Generating synthetic training data...")
    df = generate_synthetic_data(5000)
    
    print("Creating features...")
    df = create_features(df)
    
    # Define features and target
    feature_columns = [
        'hit_accuracy', 'avg_reaction_time', 'current_streak', 'max_streak',
        'score', 'target_size', 'target_speed', 'difficulty_level',
        'recent_hits', 'recent_misses', 'combo_multiplier', 'time_played',
        'hit_miss_ratio', 'score_per_click', 'streak_ratio',
        'difficulty_size_interaction', 'difficulty_speed_interaction',
        'accuracy_reaction_product'
    ]
    
    X = df[feature_columns]
    y = df['next_reaction_time']
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Scale features
    scaler = StandardScaler()
    
    # Check for any remaining issues before scaling
    print("Checking data integrity...")
    print(f"X_train shape: {X_train.shape}")
    print(f"Any NaN values: {X_train.isna().any().any()}")
    print(f"Any infinite values: {np.isinf(X_train.values).any()}")
    
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train model
    print("Training Random Forest model...")
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    
    model.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_score = model.score(X_train_scaled, y_train)
    test_score = model.score(X_test_scaled, y_test)
    
    print(f"\nModel Performance:")
    print(f"Training R² Score: {train_score:.4f}")
    print(f"Testing R² Score: {test_score:.4f}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 10 Most Important Features:")
    print(feature_importance.head(10))
    
    # Save model and scaler
    print("\nSaving model and scaler...")
    
    model_data = {
        'model': model,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'feature_importance': feature_importance.to_dict()
    }
    
    with open('dda_model.pkl', 'wb') as f:
        pickle.dump(model_data, f)
    
    # Save model metadata
    metadata = {
        'model_type': 'RandomForestRegressor',
        'n_features': len(feature_columns),
        'features': feature_columns,
        'train_score': float(train_score),
        'test_score': float(test_score),
        'n_samples_train': len(X_train),
        'n_samples_test': len(X_test)
    }
    
    with open('model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("Model saved successfully!")
    
    # Generate sample predictions
    print("\nSample Predictions:")
    sample_indices = np.random.choice(X_test.index, 5)
    for idx in sample_indices:
        sample = X_test.loc[idx:idx]
        actual = y_test.loc[idx]
        pred = model.predict(scaler.transform(sample))[0]
        print(f"  Actual: {actual:.3f}s, Predicted: {pred:.3f}s, "
              f"Accuracy: {sample['hit_accuracy'].values[0]:.2f}, "
              f"Difficulty: {sample['difficulty_level'].values[0]:.2f}")

if __name__ == "__main__":
    train_dda_model()
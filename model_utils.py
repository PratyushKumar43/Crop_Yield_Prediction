"""
Utility functions for training and saving crop yield prediction models.
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

def load_and_preprocess_data(file_path="crop yield data sheet.xlsx"):
    """
    Load and preprocess the crop yield dataset.
    
    Args:
        file_path (str): Path to the Excel file
        
    Returns:
        pd.DataFrame: Preprocessed dataset
    """
    try:
        # Load data
        df = pd.read_excel(file_path)
        
        # Remove invalid temperature values
        df = df[df['Temperatue'] != ':']
        
        # Convert temperature to float
        df['Temperatue'] = df['Temperatue'].astype(float)
        
        # Fill missing values with median
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
        
        print(f"Data loaded successfully. Shape: {df.shape}")
        return df
        
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

def prepare_features_target(df):
    """
    Prepare features and target variables.
    
    Args:
        df (pd.DataFrame): Preprocessed dataset
        
    Returns:
        tuple: X (features), y (target)
    """
    X = df.drop('Yeild (Q/acre)', axis=1)
    y = df['Yeild (Q/acre)']
    return X, y

def train_decision_tree(X_train, y_train, hyperparameter_tuning=True):
    """
    Train Decision Tree Regressor with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
        
    Returns:
        DecisionTreeRegressor: Trained model
    """
    if hyperparameter_tuning:
        # Hyperparameter tuning
        param_grid = {
            "max_depth": [2, 4, 6, 8],
            "min_samples_split": [2, 4, 6, 8],
            "min_samples_leaf": [2, 4, 6, 8],
            "random_state": [0, 42]
        }
        
        dt = DecisionTreeRegressor()
        grid_search = GridSearchCV(dt, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for Decision Tree: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        # Use best parameters from analysis
        dt = DecisionTreeRegressor(
            max_depth=4,
            min_samples_leaf=2,
            min_samples_split=8,
            random_state=0
        )
        dt.fit(X_train, y_train)
        return dt

def train_random_forest(X_train, y_train, hyperparameter_tuning=True):
    """
    Train Random Forest Regressor with optional hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training target
        hyperparameter_tuning (bool): Whether to perform hyperparameter tuning
        
    Returns:
        RandomForestRegressor: Trained model
    """
    if hyperparameter_tuning:
        # Hyperparameter tuning
        param_grid = {
            "n_estimators": [100, 200, 300],
            "max_depth": [2, 4, 6, 8],
            "min_samples_split": [2, 4, 6, 8],
            "min_samples_leaf": [2, 4, 6, 8],
            "random_state": [0, 42]
        }
        
        rf = RandomForestRegressor()
        grid_search = GridSearchCV(rf, param_grid, cv=5, n_jobs=-1, verbose=1)
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters for Random Forest: {grid_search.best_params_}")
        return grid_search.best_estimator_
    else:
        # Use best parameters from analysis
        rf = RandomForestRegressor(
            max_depth=4,
            min_samples_leaf=2,
            min_samples_split=6,
            n_estimators=100,
            random_state=42
        )
        rf.fit(X_train, y_train)
        return rf

def evaluate_model(model, X_test, y_test, model_name="Model"):
    """
    Evaluate model performance.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test target
        model_name (str): Name of the model for display
        
    Returns:
        dict: Dictionary containing predictions and metrics
    """
    predictions = model.predict(X_test)
    
    metrics = {
        'mse': mean_squared_error(y_test, predictions),
        'mae': mean_absolute_error(y_test, predictions),
        'r2': r2_score(y_test, predictions)
    }
    
    print(f"\n{model_name} Performance:")
    print(f"Mean Squared Error: {metrics['mse']:.4f}")
    print(f"Mean Absolute Error: {metrics['mae']:.4f}")
    print(f"R² Score: {metrics['r2']:.4f}")
    
    return {
        'predictions': predictions,
        'metrics': metrics
    }

def save_model(model, filename):
    """
    Save trained model to disk.
    
    Args:
        model: Trained model
        filename (str): Filename to save the model
    """
    try:
        # Create models directory if it doesn't exist
        os.makedirs('models', exist_ok=True)
        
        # Save model
        filepath = os.path.join('models', filename)
        joblib.dump(model, filepath)
        print(f"Model saved successfully to {filepath}")
        
    except Exception as e:
        print(f"Error saving model: {e}")

def load_model(filename):
    """
    Load trained model from disk.
    
    Args:
        filename (str): Filename of the saved model
        
    Returns:
        Trained model or None if loading fails
    """
    try:
        filepath = os.path.join('models', filename)
        model = joblib.load(filepath)
        print(f"Model loaded successfully from {filepath}")
        return model
        
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

def get_feature_importance(model, feature_names):
    """
    Get feature importance from trained model.
    
    Args:
        model: Trained model with feature_importances_ attribute
        feature_names: List of feature names
        
    Returns:
        pd.DataFrame: DataFrame with features and their importance scores
    """
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        return importance_df
    else:
        print("Model does not have feature importance information")
        return None

def train_and_save_models(data_file="crop yield data sheet.xlsx", save_models=True):
    """
    Complete pipeline to train and save both models.
    
    Args:
        data_file (str): Path to the data file
        save_models (bool): Whether to save the trained models
        
    Returns:
        dict: Dictionary containing trained models and results
    """
    print("Starting model training pipeline...")
    
    # Load and preprocess data
    df = load_and_preprocess_data(data_file)
    if df is None:
        return None
    
    # Prepare features and target
    X, y = prepare_features_target(df)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Training set size: {X_train.shape}")
    print(f"Test set size: {X_test.shape}")
    
    # Train Decision Tree
    print("\nTraining Decision Tree Regressor...")
    dt_model = train_decision_tree(X_train, y_train, hyperparameter_tuning=False)
    dt_results = evaluate_model(dt_model, X_test, y_test, "Decision Tree")
    
    # Train Random Forest
    print("\nTraining Random Forest Regressor...")
    rf_model = train_random_forest(X_train, y_train, hyperparameter_tuning=False)
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    
    # Save models if requested
    if save_models:
        save_model(dt_model, 'decision_tree_model.pkl')
        save_model(rf_model, 'random_forest_model.pkl')
    
    # Feature importance
    print("\nFeature Importance (Random Forest):")
    feature_importance = get_feature_importance(rf_model, X.columns)
    if feature_importance is not None:
        print(feature_importance)
    
    results = {
        'dt_model': dt_model,
        'rf_model': rf_model,
        'dt_results': dt_results,
        'rf_results': rf_results,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'feature_importance': feature_importance
    }
    
    print("\nModel training pipeline completed!")
    return results

if __name__ == "__main__":
    # Example usage
    results = train_and_save_models()
    if results:
        print("\nTraining completed successfully!")
        print(f"Decision Tree R² Score: {results['dt_results']['metrics']['r2']:.4f}")
        print(f"Random Forest R² Score: {results['rf_results']['metrics']['r2']:.4f}")

#!/usr/bin/env python3
"""
Demo script for Crop Yield Prediction models.
This script demonstrates how to use the trained models for predictions.

Usage:
    python demo.py
"""

import pandas as pd
import numpy as np
from model_utils import load_and_preprocess_data, train_models, load_model
import os

def demo_prediction():
    """Demonstrate model prediction with example data."""
    print("🌾 Crop Yield Prediction Demo")
    print("=" * 40)
    
    # Load data
    print("📂 Loading dataset...")
    df = load_and_preprocess_data()
    if df is None:
        print("❌ Failed to load data!")
        return
    
    print(f"✅ Dataset loaded: {df.shape[0]} records, {df.shape[1]} features")
    
    # Display dataset info
    print("\n📊 Dataset Overview:")
    print(f"   • Average Yield: {df['Yeild (Q/acre)'].mean():.2f} Q/acre")
    print(f"   • Max Yield: {df['Yeild (Q/acre)'].max():.2f} Q/acre")
    print(f"   • Min Yield: {df['Yeild (Q/acre)'].min():.2f} Q/acre")
    
    # Train models (quick training without hyperparameter tuning)
    print("\n🤖 Training models...")
    X = df.drop('Yeild (Q/acre)', axis=1)
    y = df['Yeild (Q/acre)']
    
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.tree import DecisionTreeRegressor
    from sklearn.metrics import r2_score, mean_absolute_error
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest (best model)
    rf_model = RandomForestRegressor(
        max_depth=4, min_samples_leaf=2, min_samples_split=6, 
        n_estimators=100, random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    
    print(f"✅ Random Forest trained - R² Score: {rf_r2:.4f}, MAE: {rf_mae:.4f}")
    
    # Example predictions
    print("\n🔮 Example Predictions:")
    print("-" * 40)
    
    # Example 1: High rainfall scenario
    example1 = np.array([[1200, 28, 75, 40, 35, 25]])  # High rainfall
    pred1 = rf_model.predict(example1)[0]
    print(f"📍 High Rainfall Scenario:")
    print(f"   Rain: 1200mm, Temp: 28°C, Fertilizer: 75kg")
    print(f"   N: 40, P: 35, K: 25")
    print(f"   🌾 Predicted Yield: {pred1:.2f} Q/acre")
    
    # Example 2: Low rainfall scenario
    example2 = np.array([[450, 35, 40, 20, 15, 10]])  # Low rainfall
    pred2 = rf_model.predict(example2)[0]
    print(f"\n📍 Low Rainfall Scenario:")
    print(f"   Rain: 450mm, Temp: 35°C, Fertilizer: 40kg")
    print(f"   N: 20, P: 15, K: 10")
    print(f"   🌾 Predicted Yield: {pred2:.2f} Q/acre")
    
    # Example 3: Optimal scenario
    example3 = np.array([[800, 30, 60, 35, 30, 20]])  # Balanced
    pred3 = rf_model.predict(example3)[0]
    print(f"\n📍 Balanced Scenario:")
    print(f"   Rain: 800mm, Temp: 30°C, Fertilizer: 60kg")
    print(f"   N: 35, P: 30, K: 20")
    print(f"   🌾 Predicted Yield: {pred3:.2f} Q/acre")
    
    # Feature importance
    print(f"\n🎯 Feature Importance (Top 3):")
    feature_names = X.columns
    importance_scores = rf_model.feature_importances_
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importance_scores
    }).sort_values('Importance', ascending=False)
    
    for i, (_, row) in enumerate(importance_df.head(3).iterrows()):
        print(f"   {i+1}. {row['Feature']}: {row['Importance']:.3f}")
    
    # Performance comparison with average
    avg_yield = df['Yeild (Q/acre)'].mean()
    print(f"\n📊 Yield Comparison (vs average {avg_yield:.2f} Q/acre):")
    
    scenarios = [
        ("High Rainfall", pred1),
        ("Low Rainfall", pred2),
        ("Balanced", pred3)
    ]
    
    for name, pred in scenarios:
        diff_pct = ((pred / avg_yield) - 1) * 100
        status = "🟢" if diff_pct > 0 else "🔴"
        print(f"   {status} {name}: {diff_pct:+.1f}%")
    
    print(f"\n💡 Tips for better yield:")
    print(f"   • Maintain optimal temperature (28-32°C)")
    print(f"   • Ensure adequate rainfall (600-1000mm)")
    print(f"   • Balance fertilizer usage (50-70kg)")
    print(f"   • Monitor soil nutrients (N>P>K)")
    
    print(f"\n🚀 For interactive predictions, run: streamlit run streamlit_app.py")

def interactive_prediction():
    """Interactive prediction mode."""
    print("\n🎮 Interactive Prediction Mode")
    print("=" * 40)
    
    try:
        # Load or train model
        df = load_and_preprocess_data()
        if df is None:
            return
        
        X = df.drop('Yeild (Q/acre)', axis=1)
        y = df['Yeild (Q/acre)']
        
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestRegressor
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        rf_model = RandomForestRegressor(
            max_depth=4, min_samples_leaf=2, min_samples_split=6, 
            n_estimators=100, random_state=42
        )
        rf_model.fit(X_train, y_train)
        
        print("🤖 Model ready for predictions!")
        print("Enter crop parameters (or 'quit' to exit):\n")
        
        while True:
            try:
                rainfall = input("🌧️  Rainfall (mm): ")
                if rainfall.lower() == 'quit':
                    break
                    
                temperature = input("🌡️  Temperature (°C): ")
                fertilizer = input("🧪 Fertilizer (kg): ")
                nitrogen = input("🟢 Nitrogen (N): ")
                phosphorus = input("🔵 Phosphorus (P): ")
                potassium = input("🟡 Potassium (K): ")
                
                # Convert to numbers
                values = [float(rainfall), float(temperature), float(fertilizer),
                         float(nitrogen), float(phosphorus), float(potassium)]
                
                # Make prediction
                prediction = rf_model.predict([values])[0]
                
                print(f"\n🌾 Predicted Yield: {prediction:.2f} Q/acre")
                
                # Context
                avg_yield = df['Yeild (Q/acre)'].mean()
                if prediction > avg_yield:
                    print(f"🎉 This is {((prediction/avg_yield-1)*100):.1f}% above average!")
                else:
                    print(f"📊 This is {((1-prediction/avg_yield)*100):.1f}% below average.")
                
                print("-" * 40)
                
            except ValueError:
                print("❌ Please enter valid numbers.")
            except KeyboardInterrupt:
                print("\n👋 Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                
    except Exception as e:
        print(f"❌ Error in interactive mode: {e}")

def main():
    """Main demo function."""
    print("🌾 CROP YIELD PREDICTION DEMO")
    print("=" * 50)
    
    choice = input("""
Choose demo mode:
1. 📊 Quick Demo (example predictions)
2. 🎮 Interactive Mode (enter your own values)

Your choice (1/2): """).strip()
    
    if choice == "1":
        demo_prediction()
    elif choice == "2":
        interactive_prediction()
    else:
        print("❌ Invalid choice. Running quick demo...")
        demo_prediction()
    
    print("\n🎉 Demo completed!")
    print("💡 For a full interactive experience, run: python run_app.py")

if __name__ == "__main__":
    main()

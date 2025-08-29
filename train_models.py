#!/usr/bin/env python3
"""
Training script for crop yield prediction models.
Run this script to train and save the machine learning models.

Usage:
    python train_models.py
"""

import sys
import os
from model_utils import train_and_save_models

def main():
    """Main function to train and save models."""
    print("=" * 60)
    print("🌾 CROP YIELD PREDICTION MODEL TRAINING")
    print("=" * 60)
    
    try:
        # Check if data file exists
        data_file = "crop yield data sheet.xlsx"
        if not os.path.exists(data_file):
            print(f"❌ Error: Data file '{data_file}' not found!")
            print("Please ensure the Excel file is in the current directory.")
            return 1
        
        print(f"📂 Found data file: {data_file}")
        print("🚀 Starting model training pipeline...")
        print()
        
        # Train and save models
        results = train_and_save_models(data_file, save_models=True)
        
        if results is None:
            print("❌ Training failed! Please check the error messages above.")
            return 1
        
        # Display results summary
        print("\n" + "=" * 60)
        print("📊 TRAINING RESULTS SUMMARY")
        print("=" * 60)
        
        print(f"✅ Decision Tree R² Score: {results['dt_results']['metrics']['r2']:.4f}")
        print(f"✅ Random Forest R² Score: {results['rf_results']['metrics']['r2']:.4f}")
        
        print(f"\n📊 Decision Tree Metrics:")
        print(f"   • MSE: {results['dt_results']['metrics']['mse']:.4f}")
        print(f"   • MAE: {results['dt_results']['metrics']['mae']:.4f}")
        
        print(f"\n📊 Random Forest Metrics:")
        print(f"   • MSE: {results['rf_results']['metrics']['mse']:.4f}")
        print(f"   • MAE: {results['rf_results']['metrics']['mae']:.4f}")
        
        # Determine best model
        if results['rf_results']['metrics']['r2'] > results['dt_results']['metrics']['r2']:
            print(f"\n🏆 Best Model: Random Forest (R² = {results['rf_results']['metrics']['r2']:.4f})")
        else:
            print(f"\n🏆 Best Model: Decision Tree (R² = {results['dt_results']['metrics']['r2']:.4f})")
        
        print("\n💾 Models saved to 'models/' directory:")
        print("   • decision_tree_model.pkl")
        print("   • random_forest_model.pkl")
        
        print("\n🎉 Training completed successfully!")
        print("📱 You can now run the Streamlit app: streamlit run streamlit_app.py")
        
        return 0
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Training interrupted by user.")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)

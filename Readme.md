# 🌾 Crop Yield Prediction

<div align="center">
  <img src="https://img.in-part.com/resize?stripmeta=true&noprofile=true&quality=95&url=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fassets.in-part.com%2Ftechnologies%2Fheader-images%2F2aVv2twTYW9qZGGhPrxw_AdobeStock_241906053.jpeg&width=1200&height=820" width="700" height="400" alt="Crop Yield Prediction"/>
</div>

## 📋 Table of Contents
- [Project Overview](#-project-overview)
- [Dataset Information](#-dataset-information)
- [Features](#-features)
- [Installation](#-installation)
- [Usage](#-usage)
- [Data Analysis Pipeline](#-data-analysis-pipeline)
- [Model Performance](#-model-performance)
- [Key Insights](#-key-insights)
- [Project Structure](#-project-structure)
- [Technologies Used](#-technologies-used)
- [Contributing](#-contributing)
- [License](#-license)

## 🎯 Project Overview

This data science project aims to predict crop yield using machine learning techniques. The project analyzes various environmental and agricultural factors to understand their impact on crop productivity and builds predictive models to estimate yield in Quintals per acre.

The analysis reveals interesting patterns suggesting the dataset contains data from two distinct crops with different environmental requirements, making this a comprehensive study of agricultural productivity factors.

## 📊 Dataset Information

### Data Source
Dataset obtained from [Kaggle - Crop Yield Prediction](https://www.kaggle.com/datasets/yaminh/crop-yield-prediction)

### Data Dictionary

| Column Name | Description | Unit |
|-------------|-------------|------|
| Rain Fall (mm) | Annual rainfall received | Millimeters |
| Temperature (C) | Average temperature | Celsius |
| Fertilizer (kg) | Amount of fertilizer used | Kilograms |
| Nitrogen (N) | Nitrogen macro nutrient level | Percentage |
| Phosphorous (P) | Phosphorous macro nutrient level | Percentage |
| Potassium (K) | Potassium macro nutrient level | Percentage |
| Yield (Q/acres) | **Target Variable** - Crop yield | Quintals per acre |

### Dataset Characteristics
- **Total Records**: Variable (cleaned dataset)
- **Features**: 6 independent variables
- **Target**: 1 dependent variable (Yield)
- **Data Types**: Numerical (continuous)

## ✨ Features

- **Data Preprocessing**: Comprehensive data cleaning and preparation
- **Exploratory Data Analysis**: In-depth statistical and visual analysis
- **Feature Engineering**: Data transformation and preparation for modeling
- **Machine Learning Models**: Multiple regression algorithms implementation
- **Model Evaluation**: Performance metrics and comparison
- **Visualization**: Interactive plots and charts using Matplotlib and Seaborn

## 🚀 Installation

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Setup Instructions

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd "Crop Yield Prediction"
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv crop_yield_env
   source crop_yield_env/bin/activate  # On Windows: crop_yield_env\Scripts\activate
   ```

3. **Install required packages**
   ```bash
   pip install -r requirements.txt
   ```

### Required Dependencies
```
pandas>=1.3.0          # Data manipulation and analysis
numpy>=1.21.0           # Numerical computations
matplotlib>=3.5.0       # Plotting and visualization
seaborn>=0.11.0         # Statistical data visualization
scikit-learn>=1.0.0     # Machine learning algorithms
openpyxl>=3.0.0         # Excel file reading support
jupyter>=1.0.0          # Jupyter notebook support
ipykernel>=6.0.0        # IPython kernel for Jupyter
```

## 📖 Usage

### 🚀 Quick Start - Streamlit Web App (Recommended)

The easiest way to explore this project is through our interactive Streamlit web application:

1. **Launch the app**
   ```bash
   python run_app.py
   ```
   Or manually:
   ```bash
   streamlit run streamlit_app.py
   ```

2. **Open your browser** to `http://localhost:8501`

3. **Explore the features**:
   - 🏠 **Overview**: Project summary and key insights
   - 📊 **Data Analysis**: Interactive visualizations and EDA
   - 🔮 **Prediction**: Real-time crop yield predictions
   - 📈 **Model Performance**: Compare model accuracy
   - 📋 **Data Explorer**: Browse and filter the dataset

### 📓 Jupyter Notebook Analysis

For detailed analysis and model development:

1. **Start Jupyter Notebook**
   ```bash
   jupyter notebook
   ```

2. **Open the main notebook**
   ```
   crop yield prediction.ipynb
   ```

3. **Execute cells sequentially** or run all cells to see the complete analysis

### 🤖 Train Models

To retrain the machine learning models:

```bash
python train_models.py
```

### Quick Start Code
```python
# Load and preview the dataset
import pandas as pd
df = pd.read_excel("crop yield data sheet.xlsx")
print(df.head())
print(f"Dataset shape: {df.shape}")
```

## 🔄 Data Analysis Pipeline

### 1. Data Preprocessing
- **Data Cleaning**: Removal of invalid entries (e.g., ":" in temperature column)
- **Type Conversion**: Converting object types to appropriate numerical types
- **Missing Value Treatment**: Imputation using median values
- **Data Validation**: Ensuring data consistency and quality

### 2. Exploratory Data Analysis (EDA)
- **Univariate Analysis**: Distribution plots for each feature
- **Bivariate Analysis**: Correlation analysis between features and target
- **Multivariate Analysis**: Relationship exploration using scatter plots
- **Statistical Summary**: Descriptive statistics and outlier detection

### 3. Feature Engineering
- **Feature Selection**: Identifying most important predictors
- **Data Splitting**: Train-test split (80-20 ratio)
- **Feature Scaling**: Normalization if required

### 4. Model Development
- **Algorithm Selection**: Decision Tree and Random Forest Regressors
- **Hyperparameter Tuning**: Grid Search CV for optimal parameters
- **Model Training**: Fitting models on training data
- **Prediction**: Generating predictions on test data

### 5. Model Evaluation
- **Performance Metrics**: MSE, MAE, R² Score
- **Model Comparison**: Comparative analysis of different algorithms
- **Feature Importance**: Understanding key predictive factors

## 📈 Model Performance

### Results Summary

| Model | R² Score | MSE | MAE | Best Parameters |
|-------|----------|-----|-----|-----------------|
| **Random Forest Regressor** | **0.802** | Low | Low | max_depth=4, min_samples_leaf=2, min_samples_split=6, n_estimators=100 |
| Decision Tree Regressor | 0.770 | Higher | Higher | max_depth=4, min_samples_leaf=2, min_samples_split=8 |

### Feature Importance Ranking
1. **Temperature** - Most influential factor
2. **Rainfall** - Second most important
3. **Fertilizer** - Moderate importance
4. **Macronutrients (N, P, K)** - Lower but significant importance

## 🔍 Key Insights

### Agricultural Insights
- **Dual Crop Pattern**: Dataset likely contains two distinct crops with different environmental requirements
- **Temperature Dependency**: Temperature emerges as the most critical factor for yield prediction
- **Rainfall Clusters**: Two distinct rainfall patterns (400-500mm and >1100mm) suggest different crop types
- **Fertilizer Impact**: Non-linear relationship between fertilizer usage and yield

### Data Science Insights
- **Model Selection**: Random Forest outperforms Decision Tree due to ensemble learning
- **Feature Engineering**: Temperature and rainfall show strong predictive power
- **Data Quality**: Preprocessing significantly improved model performance
- **Overfitting Prevention**: Hyperparameter tuning helped achieve optimal performance

## 📁 Project Structure

```
Crop Yield Prediction/
├── 📊 crop yield data sheet.xlsx      # Raw dataset
├── 📓 crop yield prediction.ipynb     # Main analysis notebook
├── 📄 crop yield prediction.pdf       # Project report
├── 📝 description.md                  # Project description
├── 📖 Readme.md                       # This file
├── 📋 requirements.txt                # Dependencies list
├── 🌐 streamlit_app.py                # Streamlit web application
├── 🤖 model_utils.py                  # Model training utilities
├── 🚂 train_models.py                 # Model training script
├── 🚀 run_app.py                      # Quick start script
├── 📚 STREAMLIT_README.md             # Streamlit app documentation
└── 📁 models/                         # Saved ML models (created after training)
    ├── decision_tree_model.pkl
    └── random_forest_model.pkl
```

## 🛠️ Technologies Used

### Programming Language
- **Python 3.7+** - Core programming language

### Web Application
- **Streamlit** - Interactive web application framework
- **Plotly** - Interactive data visualization

### Data Science Libraries
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Matplotlib** - Static plotting and visualization
- **Seaborn** - Statistical data visualization

### Machine Learning
- **Scikit-learn** - Machine learning algorithms and tools
  - Model Selection (train_test_split, GridSearchCV)
  - Regression Models (DecisionTreeRegressor, RandomForestRegressor)
  - Metrics (MSE, MAE, R² Score)
- **Joblib** - Model persistence and serialization

### Development Environment
- **Jupyter Notebook** - Interactive development environment
- **openpyxl** - Excel file processing

## 🤝 Contributing

We welcome contributions to improve this project! Here's how you can help:

### Ways to Contribute
1. **Bug Reports**: Report any issues you find
2. **Feature Requests**: Suggest new features or improvements
3. **Code Contributions**: Submit pull requests with enhancements
4. **Documentation**: Improve documentation and examples

### Contribution Guidelines
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions, suggestions, or collaborations, please feel free to reach out through GitHub issues or discussions.

## 🙏 Acknowledgments

- **Dataset Source**: Kaggle Community
- **Inspiration**: Agricultural technology and data science community
- **Tools**: Open-source Python ecosystem

---

<div align="center">
  <p>Made with ❤️ for the agricultural and data science community</p>
  <p>⭐ Star this repository if you found it helpful!</p>
</div>
# 🌾 Crop Yield Prediction - Streamlit App

<div align="center">
  <img src="https://img.in-part.com/resize?stripmeta=true&noprofile=true&quality=95&url=https%3A%2F%2Fs3-eu-west-1.amazonaws.com%2Fassets.in-part.com%2Ftechnologies%2Fheader-images%2F2aVv2twTYW9qZGGhPrxw_AdobeStock_241906053.jpeg&width=1200&height=820" width="600" height="300" alt="Crop Yield Prediction"/>
</div>

## 🎯 Overview

This interactive Streamlit web application provides a comprehensive dashboard for crop yield prediction using machine learning. The app allows users to explore data, visualize insights, and make real-time predictions based on environmental and agricultural factors.

## ✨ Features

### 🏠 Overview Page
- **Project Summary**: Comprehensive overview of the crop yield prediction project
- **Key Statistics**: Dataset insights and performance metrics
- **Data Dictionary**: Detailed explanation of all features
- **Key Insights**: Major findings from the analysis

### 📊 Data Analysis Page
- **Interactive Visualizations**: Feature distributions, correlation heatmaps, scatter plots
- **Correlation Analysis**: Understanding relationships between variables
- **Feature Relationships**: Explore how different factors affect crop yield
- **Plotly Integration**: Interactive charts with zoom, pan, and hover capabilities

### 🔮 Prediction Page
- **Interactive Input Form**: Sliders and controls for all input parameters
- **Real-time Predictions**: Instant yield predictions using trained ML models
- **Model Selection**: Choose between Decision Tree and Random Forest models
- **Feature Importance**: Understand which factors matter most
- **Performance Context**: Compare predictions with dataset averages

### 📈 Model Performance Page
- **Model Comparison**: Side-by-side comparison of different algorithms
- **Performance Metrics**: R² Score, MSE, MAE for model evaluation
- **Prediction vs Actual**: Scatter plots showing model accuracy
- **Residual Analysis**: Distribution plots for error analysis

### 📋 Data Explorer Page
- **Interactive Data Filtering**: Filter dataset by yield and temperature ranges
- **Raw Data Viewer**: Browse and explore the complete dataset
- **Summary Statistics**: Descriptive statistics for filtered data
- **Data Export**: Download filtered data as CSV

## 🚀 Installation & Setup

### Prerequisites
- Python 3.7 or higher
- pip package manager

### Step-by-Step Installation

1. **Navigate to the project directory**
   ```bash
   cd "Crop_Yield_Prediction"
   ```

2. **Create a virtual environment** (recommended)
   ```bash
   python -m venv crop_yield_env
   
   # On Windows
   crop_yield_env\Scripts\activate
   
   # On macOS/Linux
   source crop_yield_env/bin/activate
   ```

3. **Install required dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation**
   ```bash
   streamlit --version
   ```

## 🎮 Running the App

### Launch the Streamlit App
```bash
streamlit run streamlit_app.py
```

### Alternative: Run with custom configuration
```bash
streamlit run streamlit_app.py --server.port 8501 --server.address localhost
```

### Access the App
- Open your web browser
- Navigate to: `http://localhost:8501`
- The app will automatically open in your default browser

## 📱 Usage Guide

### 1. Navigation
- Use the **sidebar navigation menu** to switch between different sections
- Each section provides unique insights and functionality

### 2. Making Predictions
1. Go to the **🔮 Prediction** page
2. Adjust the sliders for each parameter:
   - 🌧️ Rainfall (mm)
   - 🌡️ Temperature (°C)
   - 🧪 Fertilizer (kg)
   - 🟢 Nitrogen (N)
   - 🔵 Phosphorus (P)
   - 🟡 Potassium (K)
3. Select your preferred model (Random Forest recommended)
4. Click **🎯 Predict Crop Yield**
5. View the prediction result and performance context

### 3. Exploring Data
1. Visit the **📊 Data Analysis** page for visualizations
2. Use the **📋 Data Explorer** for raw data investigation
3. Apply filters to focus on specific data ranges
4. Download filtered datasets for further analysis

### 4. Understanding Performance
1. Check the **📈 Model Performance** page
2. Compare different models' accuracy
3. Analyze prediction vs actual plots
4. Review residual distributions

## 🔧 Configuration

### Environment Variables
Create a `.env` file (optional) for custom configurations:
```env
STREAMLIT_SERVER_PORT=8501
STREAMLIT_SERVER_ADDRESS=localhost
DATA_PATH=crop yield data sheet.xlsx
```

### Customizing the App
- **Colors & Styling**: Modify the CSS in `streamlit_app.py`
- **Data Source**: Update the data loading path in `load_data()` function
- **Model Parameters**: Adjust hyperparameters in `model_utils.py`

## 📊 Data Requirements

### Dataset Format
The app expects an Excel file (`crop yield data sheet.xlsx`) with the following columns:
- `Rain Fall (mm)`: Rainfall in millimeters
- `Temperatue`: Temperature in Celsius (note: original typo preserved)
- `Fertilizer`: Fertilizer amount in kilograms
- `Nitrogen (N)`: Nitrogen nutrient level
- `Phosphorus (P)`: Phosphorus nutrient level
- `Potassium (K)`: Potassium nutrient level
- `Yeild (Q/acre)`: Crop yield in Quintals per acre (target variable)

### Data Preprocessing
The app automatically handles:
- ✅ Invalid temperature values (removes ":" entries)
- ✅ Data type conversions
- ✅ Missing value imputation using median
- ✅ Data validation and cleaning

## 🤖 Machine Learning Models

### Implemented Algorithms
1. **Decision Tree Regressor**
   - Max depth: 4
   - Min samples split: 8
   - Min samples leaf: 2
   - Random state: 0

2. **Random Forest Regressor** (Recommended)
   - N estimators: 100
   - Max depth: 4
   - Min samples split: 6
   - Min samples leaf: 2
   - Random state: 42

### Model Performance
- **Random Forest**: R² Score ≈ 0.802
- **Decision Tree**: R² Score ≈ 0.770

### Feature Importance Ranking
1. **Temperature** - Most influential factor
2. **Rainfall** - Second most important
3. **Fertilizer** - Moderate importance
4. **Macronutrients** - Lower but significant importance

## 🛠️ Troubleshooting

### Common Issues

#### 1. Data Loading Error
```
Error: No such file or directory: 'crop yield data sheet.xlsx'
```
**Solution**: Ensure the Excel file is in the correct directory and path is accurate.

#### 2. Module Import Error
```
ModuleNotFoundError: No module named 'streamlit'
```
**Solution**: Install requirements using `pip install -r requirements.txt`

#### 3. Port Already in Use
```
Error: Port 8501 is already in use
```
**Solution**: Use a different port: `streamlit run streamlit_app.py --server.port 8502`

#### 4. Memory Issues with Large Datasets
**Solution**: 
- Implement data sampling for very large datasets
- Use `@st.cache_data` for optimization
- Consider data chunking for processing

### Performance Optimization
- **Caching**: Models and data are cached using `@st.cache_data`
- **Lazy Loading**: Data is loaded only when needed
- **Efficient Plotting**: Using Plotly for interactive visualizations

## 📚 Additional Resources

### Documentation
- [Streamlit Documentation](https://docs.streamlit.io/)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Plotly Python Documentation](https://plotly.com/python/)

### Extension Ideas
- 🌐 **Multi-crop Support**: Extend to predict yields for different crop types
- 📅 **Time Series**: Add temporal analysis for seasonal patterns
- 🗺️ **Geographic Analysis**: Include location-based factors
- 🤝 **User Authentication**: Add user accounts and prediction history
- 📧 **Automated Reports**: Generate and email prediction reports
- 🔄 **Model Retraining**: Interface for updating models with new data

## 📄 License

This project is licensed under the MIT License - see the main project README for details.

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues, feature requests, or pull requests.

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test the Streamlit app thoroughly
5. Submit a pull request

## 📞 Support

For issues or questions regarding the Streamlit app:
1. Check this README for common solutions
2. Review the troubleshooting section
3. Open an issue in the project repository
4. Include error messages and system information

---

<div align="center">
  <p>🌾 Happy Predicting! 🚀</p>
  <p>Made with ❤️ using Streamlit and Python</p>
</div>

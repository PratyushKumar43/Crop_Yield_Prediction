import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import GridSearchCV
import pickle
import os

# Set page configuration
st.set_page_config(
    page_title="🌾 Crop Yield Prediction",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #2E8B57;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #228B22;
        margin: 1rem 0;
    }
    .metric-card {
        background-color: #f0f8f0;
        padding: 1rem;
        border-radius: 10px;
        border-left: 5px solid #2E8B57;
        margin: 0.5rem 0;
    }
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f0f8f0 0%, #e8f5e8 100%);
    }
    .prediction-result {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        font-size: 1.5rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Load and cache data
@st.cache_data
def load_data():
    """Load and preprocess the crop yield dataset"""
    try:
        df = pd.read_excel("crop yield data sheet.xlsx")
        
        # Data preprocessing
        # Remove invalid temperature values
        df = df[df['Temperatue'] != ':']
        
        # Convert temperature to float
        df['Temperatue'] = df['Temperatue'].astype(float)
        
        # Fill missing values with median
        for col in df.columns:
            if df[col].dtype in ['float64', 'int64']:
                df[col] = df[col].fillna(df[col].median())
        
        return df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None

@st.cache_data
def train_models(df):
    """Train and return both Decision Tree and Random Forest models"""
    # Prepare features and target
    X = df.drop('Yeild (Q/acre)', axis=1)
    y = df['Yeild (Q/acre)']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Decision Tree with best parameters from notebook
    dt_model = DecisionTreeRegressor(
        max_depth=4, 
        min_samples_leaf=2, 
        min_samples_split=8, 
        random_state=0
    )
    dt_model.fit(X_train, y_train)
    dt_pred = dt_model.predict(X_test)
    
    # Random Forest with best parameters from notebook
    rf_model = RandomForestRegressor(
        max_depth=4, 
        min_samples_leaf=2, 
        min_samples_split=6, 
        n_estimators=100, 
        random_state=42
    )
    rf_model.fit(X_train, y_train)
    rf_pred = rf_model.predict(X_test)
    
    # Calculate metrics
    dt_metrics = {
        'mse': mean_squared_error(y_test, dt_pred),
        'mae': mean_absolute_error(y_test, dt_pred),
        'r2': r2_score(y_test, dt_pred)
    }
    
    rf_metrics = {
        'mse': mean_squared_error(y_test, rf_pred),
        'mae': mean_absolute_error(y_test, rf_pred),
        'r2': r2_score(y_test, rf_pred)
    }
    
    return {
        'dt_model': dt_model,
        'rf_model': rf_model,
        'X_train': X_train,
        'X_test': X_test,
        'y_train': y_train,
        'y_test': y_test,
        'dt_pred': dt_pred,
        'rf_pred': rf_pred,
        'dt_metrics': dt_metrics,
        'rf_metrics': rf_metrics
    }

def main():
    # Header
    st.markdown('<h1 class="main-header">🌾 Crop Yield Prediction Dashboard</h1>', unsafe_allow_html=True)
    
    # Load data
    df = load_data()
    if df is None:
        st.error("Failed to load data. Please ensure the Excel file is in the correct location.")
        return
    
    # Sidebar navigation
    st.sidebar.title("🎯 Navigation")
    page = st.sidebar.selectbox(
        "Choose a section:",
        ["🏠 Overview", "📊 Data Analysis", "🔮 Prediction", "📈 Model Performance", "📋 Data Explorer"]
    )
    
    if page == "🏠 Overview":
        show_overview(df)
    elif page == "📊 Data Analysis":
        show_data_analysis(df)
    elif page == "🔮 Prediction":
        show_prediction(df)
    elif page == "📈 Model Performance":
        show_model_performance(df)
    elif page == "📋 Data Explorer":
        show_data_explorer(df)

def show_overview(df):
    """Display project overview and key insights"""
    st.markdown('<h2 class="sub-header">Project Overview</h2>', unsafe_allow_html=True)
    
    # Project description
    st.markdown("""
    This interactive dashboard provides insights into crop yield prediction using machine learning. 
    The analysis is based on environmental and agricultural factors such as rainfall, temperature, 
    fertilizer usage, and macronutrient levels.
    """)
    
    # Key statistics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(
            f'<div class="metric-card"><h3>{len(df)}</h3><p>Total Records</p></div>',
            unsafe_allow_html=True
        )
    
    with col2:
        st.markdown(
            f'<div class="metric-card"><h3>{len(df.columns)-1}</h3><p>Features</p></div>',
            unsafe_allow_html=True
        )
    
    with col3:
        avg_yield = df['Yeild (Q/acre)'].mean()
        st.markdown(
            f'<div class="metric-card"><h3>{avg_yield:.1f}</h3><p>Avg Yield (Q/acre)</p></div>',
            unsafe_allow_html=True
        )
    
    with col4:
        max_yield = df['Yeild (Q/acre)'].max()
        st.markdown(
            f'<div class="metric-card"><h3>{max_yield:.1f}</h3><p>Max Yield (Q/acre)</p></div>',
            unsafe_allow_html=True
        )
    
    # Data Dictionary
    st.markdown('<h3 class="sub-header">📖 Data Dictionary</h3>', unsafe_allow_html=True)
    
    data_dict = {
        'Feature': ['Rain Fall (mm)', 'Temperature (°C)', 'Fertilizer (kg)', 'Nitrogen (N)', 'Phosphorous (P)', 'Potassium (K)', 'Yield (Q/acre)'],
        'Description': [
            'Annual rainfall in millimeters',
            'Average temperature in Celsius', 
            'Amount of fertilizer used in kilograms',
            'Nitrogen macro nutrient level',
            'Phosphorous macro nutrient level',
            'Potassium macro nutrient level',
            'Crop yield in Quintals per acre (Target)'
        ]
    }
    
    st.dataframe(pd.DataFrame(data_dict), use_container_width=True)
    
    # Key Insights
    st.markdown('<h3 class="sub-header">🔍 Key Insights</h3>', unsafe_allow_html=True)
    
    insights = [
        "📊 Dataset contains patterns suggesting two distinct crop types",
        "🌡️ Temperature is the most influential factor for yield prediction",
        "🌧️ Rainfall shows two distinct clusters (400-500mm and >1100mm)",
        "🧪 Fertilizer usage has a non-linear relationship with yield",
        "🏆 Random Forest model achieves 80.2% accuracy (R² score)"
    ]
    
    for insight in insights:
        st.markdown(f"- {insight}")

def show_data_analysis(df):
    """Display comprehensive data analysis and visualizations"""
    st.markdown('<h2 class="sub-header">📊 Exploratory Data Analysis</h2>', unsafe_allow_html=True)
    
    # Feature distributions
    st.markdown('<h3 class="sub-header">📈 Feature Distributions</h3>', unsafe_allow_html=True)
    
    # Select features for distribution plots
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_features = st.multiselect(
        "Select features to visualize:",
        numeric_cols,
        default=numeric_cols[:3]
    )
    
    if selected_features:
        cols = st.columns(min(len(selected_features), 3))
        for i, feature in enumerate(selected_features):
            with cols[i % 3]:
                fig, ax = plt.subplots(figsize=(8, 6))
                sns.histplot(data=df, x=feature, kde=True, ax=ax)
                ax.set_title(f'{feature} Distribution')
                st.pyplot(fig)
                plt.close()
    
    # Correlation Analysis
    st.markdown('<h3 class="sub-header">🔗 Correlation Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Correlation heatmap
        fig, ax = plt.subplots(figsize=(10, 8))
        correlation_matrix = df.corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='RdYlGn', center=0, ax=ax)
        ax.set_title('Feature Correlation Matrix')
        st.pyplot(fig)
        plt.close()
    
    with col2:
        # Top correlations with yield
        yield_corr = df.corr()['Yeild (Q/acre)'].drop('Yeild (Q/acre)').abs().sort_values(ascending=False)
        st.markdown("**Top Correlations with Yield:**")
        for feature, corr in yield_corr.items():
            st.markdown(f"• {feature}: {corr:.3f}")
    
    # Scatter plots with yield
    st.markdown('<h3 class="sub-header">🎯 Relationship with Yield</h3>', unsafe_allow_html=True)
    
    feature_for_scatter = st.selectbox(
        "Select feature for scatter plot with yield:",
        [col for col in df.columns if col != 'Yeild (Q/acre)']
    )
    
    fig = px.scatter(
        df, 
        x=feature_for_scatter, 
        y='Yeild (Q/acre)',
        title=f'{feature_for_scatter} vs Crop Yield',
        trendline="ols"
    )
    fig.update_layout(height=500)
    st.plotly_chart(fig, use_container_width=True)

def show_prediction(df):
    """Display prediction interface"""
    st.markdown('<h2 class="sub-header">🔮 Crop Yield Prediction</h2>', unsafe_allow_html=True)
    
    # Load or train models
    models_data = train_models(df)
    
    # Input form
    st.markdown('<h3 class="sub-header">📝 Enter Crop Parameters</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        rainfall = st.slider(
            "🌧️ Rainfall (mm)", 
            min_value=int(df['Rain Fall (mm)'].min()), 
            max_value=int(df['Rain Fall (mm)'].max()), 
            value=int(df['Rain Fall (mm)'].mean())
        )
        
        temperature = st.slider(
            "🌡️ Temperature (°C)", 
            min_value=int(df['Temperatue'].min()), 
            max_value=int(df['Temperatue'].max()), 
            value=int(df['Temperatue'].mean())
        )
        
        fertilizer = st.slider(
            "🧪 Fertilizer (kg)", 
            min_value=int(df['Fertilizer'].min()), 
            max_value=int(df['Fertilizer'].max()), 
            value=int(df['Fertilizer'].mean())
        )
    
    with col2:
        nitrogen = st.slider(
            "🟢 Nitrogen (N)", 
            min_value=int(df['Nitrogen (N)'].min()), 
            max_value=int(df['Nitrogen (N)'].max()), 
            value=int(df['Nitrogen (N)'].mean())
        )
        
        phosphorus = st.slider(
            "🔵 Phosphorus (P)", 
            min_value=int(df['Phosphorus (P)'].min()), 
            max_value=int(df['Phosphorus (P)'].max()), 
            value=int(df['Phosphorus (P)'].mean())
        )
        
        potassium = st.slider(
            "🟡 Potassium (K)", 
            min_value=int(df['Potassium (K)'].min()), 
            max_value=int(df['Potassium (K)'].max()), 
            value=int(df['Potassium (K)'].mean())
        )
    
    # Model selection
    model_choice = st.selectbox(
        "🤖 Choose Prediction Model:",
        ["Random Forest (Recommended)", "Decision Tree"]
    )
    
    # Prediction button
    if st.button("🎯 Predict Crop Yield", type="primary"):
        # Prepare input data
        input_data = np.array([[rainfall, temperature, fertilizer, nitrogen, phosphorus, potassium]])
        
        # Make prediction
        if model_choice == "Random Forest (Recommended)":
            prediction = models_data['rf_model'].predict(input_data)[0]
            model_r2 = models_data['rf_metrics']['r2']
        else:
            prediction = models_data['dt_model'].predict(input_data)[0]
            model_r2 = models_data['dt_metrics']['r2']
        
        # Display prediction
        st.markdown(
            f'<div class="prediction-result">'
            f'<h2>🌾 Predicted Yield: {prediction:.2f} Q/acre</h2>'
            f'<p>Model Accuracy (R²): {model_r2:.3f}</p>'
            f'</div>',
            unsafe_allow_html=True
        )
        
        # Performance context
        avg_yield = df['Yeild (Q/acre)'].mean()
        if prediction > avg_yield:
            st.success(f"🎉 Great! This prediction is {((prediction/avg_yield-1)*100):.1f}% above average yield ({avg_yield:.2f} Q/acre)")
        else:
            st.info(f"📊 This prediction is {((1-prediction/avg_yield)*100):.1f}% below average yield ({avg_yield:.2f} Q/acre)")
    
    # Feature importance
    st.markdown('<h3 class="sub-header">🎯 Feature Importance</h3>', unsafe_allow_html=True)
    
    rf_model = models_data['rf_model']
    feature_names = models_data['X_train'].columns
    feature_importance = rf_model.feature_importances_
    
    importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': feature_importance
    }).sort_values('Importance', ascending=True)
    
    fig = px.bar(
        importance_df, 
        x='Importance', 
        y='Feature', 
        orientation='h',
        title='Feature Importance in Random Forest Model'
    )
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def show_model_performance(df):
    """Display model performance metrics and comparisons"""
    st.markdown('<h2 class="sub-header">📈 Model Performance Analysis</h2>', unsafe_allow_html=True)
    
    # Load models and metrics
    models_data = train_models(df)
    
    # Performance metrics comparison
    st.markdown('<h3 class="sub-header">🏆 Model Comparison</h3>', unsafe_allow_html=True)
    
    metrics_df = pd.DataFrame({
        'Model': ['Decision Tree', 'Random Forest'],
        'R² Score': [models_data['dt_metrics']['r2'], models_data['rf_metrics']['r2']],
        'MSE': [models_data['dt_metrics']['mse'], models_data['rf_metrics']['mse']],
        'MAE': [models_data['dt_metrics']['mae'], models_data['rf_metrics']['mae']]
    })
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(metrics_df.round(4), use_container_width=True)
    
    with col2:
        # R² Score comparison
        fig = px.bar(
            metrics_df, 
            x='Model', 
            y='R² Score',
            title='Model Accuracy Comparison (R² Score)',
            color='R² Score',
            color_continuous_scale='Viridis'
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    # Prediction vs Actual plots
    st.markdown('<h3 class="sub-header">🎯 Prediction vs Actual</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Decision Tree
        fig = px.scatter(
            x=models_data['y_test'], 
            y=models_data['dt_pred'],
            title='Decision Tree: Predicted vs Actual',
            labels={'x': 'Actual Yield', 'y': 'Predicted Yield'}
        )
        fig.add_shape(
            type="line", line=dict(dash="dash"),
            x0=models_data['y_test'].min(), y0=models_data['y_test'].min(),
            x1=models_data['y_test'].max(), y1=models_data['y_test'].max(),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Random Forest
        fig = px.scatter(
            x=models_data['y_test'], 
            y=models_data['rf_pred'],
            title='Random Forest: Predicted vs Actual',
            labels={'x': 'Actual Yield', 'y': 'Predicted Yield'}
        )
        fig.add_shape(
            type="line", line=dict(dash="dash"),
            x0=models_data['y_test'].min(), y0=models_data['y_test'].min(),
            x1=models_data['y_test'].max(), y1=models_data['y_test'].max(),
        )
        st.plotly_chart(fig, use_container_width=True)
    
    # Residual analysis
    st.markdown('<h3 class="sub-header">📊 Residual Analysis</h3>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        dt_residuals = models_data['y_test'] - models_data['dt_pred']
        fig = px.histogram(
            x=dt_residuals, 
            title='Decision Tree: Residuals Distribution',
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        rf_residuals = models_data['y_test'] - models_data['rf_pred']
        fig = px.histogram(
            x=rf_residuals, 
            title='Random Forest: Residuals Distribution',
            nbins=20
        )
        st.plotly_chart(fig, use_container_width=True)

def show_data_explorer(df):
    """Display interactive data explorer"""
    st.markdown('<h2 class="sub-header">📋 Data Explorer</h2>', unsafe_allow_html=True)
    
    # Data overview
    st.markdown('<h3 class="sub-header">📊 Dataset Overview</h3>', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Records", len(df))
    with col2:
        st.metric("Features", len(df.columns) - 1)
    with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
    
    # Raw data display
    st.markdown('<h3 class="sub-header">🔍 Raw Data</h3>', unsafe_allow_html=True)
    
    # Filters
    col1, col2 = st.columns(2)
    
    with col1:
        yield_range = st.slider(
            "Filter by Yield Range (Q/acre)",
            min_value=float(df['Yeild (Q/acre)'].min()),
            max_value=float(df['Yeild (Q/acre)'].max()),
            value=(float(df['Yeild (Q/acre)'].min()), float(df['Yeild (Q/acre)'].max()))
        )
    
    with col2:
        temp_range = st.slider(
            "Filter by Temperature Range (°C)",
            min_value=float(df['Temperatue'].min()),
            max_value=float(df['Temperatue'].max()),
            value=(float(df['Temperatue'].min()), float(df['Temperatue'].max()))
        )
    
    # Apply filters
    filtered_df = df[
        (df['Yeild (Q/acre)'] >= yield_range[0]) & 
        (df['Yeild (Q/acre)'] <= yield_range[1]) &
        (df['Temperatue'] >= temp_range[0]) & 
        (df['Temperatue'] <= temp_range[1])
    ]
    
    st.dataframe(filtered_df, use_container_width=True)
    
    # Summary statistics
    st.markdown('<h3 class="sub-header">📈 Summary Statistics</h3>', unsafe_allow_html=True)
    st.dataframe(filtered_df.describe(), use_container_width=True)
    
    # Download filtered data
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"filtered_crop_data_{len(filtered_df)}_records.csv",
        mime="text/csv"
    )

if __name__ == "__main__":
    main()

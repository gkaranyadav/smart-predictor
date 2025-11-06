import streamlit as st

st.set_page_config(
    page_title="Home - Smart Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Smart Predictor - Home")
st.markdown("""
### Welcome to Your AI Assistant!

**Smart Predictor** helps you build machine learning models for various datasets:

🔍 **Data Analysis** - Understand your data with automated insights  
🤖 **Model Training** - Train multiple ML models with hyperparameter tuning  
📊 **Results & Evaluation** - Compare model performance and feature importance  
🚀 **Predictions** - Deploy models and make predictions on new data

### Getting Started:
1. Go to **Data Analysis** page to upload your dataset
2. Explore data statistics and visualizations  
3. Train models in **Model Training** page
4. View and compare results

---
*Currently optimized for: Diabetes Prediction, Weather Forecasting, and more coming soon!*
""")

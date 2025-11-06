import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="Model Training - Smart Predictor",
    page_icon="🤖", 
    layout="wide"
)

st.title("🤖 Model Training")
st.markdown("Train machine learning models with your data")

# Check if data is available
if 'current_dataset' not in st.session_state:
    st.warning("⚠️ No dataset found. Please go to **Data Analysis** page first to upload your data.")
    st.stop()

df = st.session_state.current_dataset

st.success(f"✅ Using dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# Target selection
st.header("🎯 Select Target Variable")
target_col = st.selectbox("Choose what you want to predict:", df.columns.tolist())

# Feature selection  
st.header("🔧 Select Features")
available_features = [col for col in df.columns if col != target_col]
selected_features = st.multiselect(
    "Choose features for training:",
    available_features,
    default=available_features
)

# Model selection
st.header("🤖 Choose Algorithms")
col1, col2, col3 = st.columns(3)
with col1:
    use_lr = st.checkbox("Logistic Regression", value=True)
with col2:
    use_rf = st.checkbox("Random Forest", value=True)
with col3:
    use_xgb = st.checkbox("XGBoost", value=True)

# Training button
if st.button("🚀 Train Models", type="primary", use_container_width=True):
    if not selected_features:
        st.error("❌ Please select at least one feature")
    elif not (use_lr or use_rf or use_xgb):
        st.error("❌ Please select at least one algorithm")
    else:
        st.success("✅ Starting model training...")
        st.info("This will connect to Databricks backend in the next version!")
        
        # Store training config
        st.session_state.training_config = {
            'target': target_col,
            'features': selected_features,
            'models': {
                'logistic_regression': use_lr,
                'random_forest': use_rf,
                'xgboost': use_xgb
            }
        }

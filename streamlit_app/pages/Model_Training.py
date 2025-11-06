import streamlit as st
import pandas as pd
import requests
import json
import time

st.set_page_config(
    page_title="Model Training - Smart Predictor",
    page_icon="🤖", 
    layout="wide"
)

st.title("🤖 Model Training")
st.markdown("Train machine learning models with hyperparameter tuning")

# Check if data is available from previous analysis
if 'current_dataset' not in st.session_state:
    st.session_state.current_dataset = None
if 'target_column' not in st.session_state:
    st.session_state.target_column = None

# Dataset selection section
st.header("📁 Dataset Setup")

if st.session_state.current_dataset is not None:
    st.success(f"✅ Using dataset from Data Analysis page")
    st.info(f"Dataset shape: {st.session_state.current_dataset.shape}")
    
    # Show target column selection
    all_columns = st.session_state.current_dataset.columns.tolist()
    target_col = st.selectbox(
        "Select target variable (what you want to predict):",
        all_columns,
        index=all_columns.index(st.session_state.target_column) if st.session_state.target_column in all_columns else 0
    )
    st.session_state.target_column = target_col
    
else:
    st.warning("⚠️ No dataset found. Please go to **Data Analysis** page first to upload your data.")
    st.stop()

# Model configuration section
st.header("⚙️ Model Configuration")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Algorithm Selection")
    use_lr = st.checkbox("Logistic Regression", value=True)
    use_rf = st.checkbox("Random Forest", value=True) 
    use_xgb = st.checkbox("XGBoost", value=True)

with col2:
    st.subheader("Training Settings")
    test_size = st.slider("Test Size (%)", 10, 40, 20)
    random_state = st.number_input("Random State", value=42)
    enable_cv = st.checkbox("Enable Cross-Validation", value=True)

with col3:
    st.subheader("Hyperparameter Tuning")
    enable_tuning = st.checkbox("Enable Auto-Tuning", value=True)
    if enable_tuning:
        tuning_method = st.radio("Tuning Method:", ["Grid Search", "Random Search"])
        max_tuning_time = st.slider("Max Tuning Time (min)", 1, 60, 10)

# Feature selection
st.header("🎯 Feature Selection")

# Show available features (exclude target)
available_features = [col for col in st.session_state.current_dataset.columns if col != st.session_state.target_column]

selected_features = st.multiselect(
    "Select features for training:",
    available_features,
    default=available_features[:min(5, len(available_features))]  # Select first 5 by default
)

if not selected_features:
    st.error("❌ Please select at least one feature for training")
    st.stop()

# Show dataset preview
st.header("📋 Data Preview")
col1, col2 = st.columns(2)

with col1:
    st.write("**Selected Features:**", selected_features)
    st.write("**Target Variable:**", st.session_state.target_column)

with col2:
    st.write("**Training Samples:**", f"{(100-test_size)}% of data")
    st.write("**Testing Samples:**", f"{test_size}% of data")

# Dataset sample
st.dataframe(st.session_state.current_dataset[selected_features + [st.session_state.target_column]].head(10), use_container_width=True)

# Training section
st.header("🚀 Start Training")

if st.button("🎯 Train Models", type="primary", use_container_width=True):
    if not use_lr and not use_rf and not use_xgb:
        st.error("❌ Please select at least one algorithm")
        st.stop()
    
    # Prepare training parameters
    training_config = {
        "dataset": st.session_state.current_dataset[selected_features + [st.session_state.target_column]].to_dict(),
        "target_column": st.session_state.target_column,
        "models": {
            "logistic_regression": use_lr,
            "random_forest": use_rf, 
            "xgboost": use_xgb
        },
        "training_settings": {
            "test_size": test_size / 100,
            "random_state": random_state,
            "cross_validation": enable_cv
        },
        "hyperparameter_tuning": {
            "enabled": enable_tuning,
            "method": tuning_method if enable_tuning else None,
            "max_time_minutes": max_tuning_time if enable_tuning else None
        }
    }
    
    # Show training progress
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # Simulate training process (we'll replace this with actual Databricks API call later)
    for i in range(5):
        progress = (i + 1) * 20
        progress_bar.progress(progress)
        
        if progress == 20:
            status_text.text("🔄 Loading and preprocessing data...")
        elif progress == 40:
            status_text.text("📊 Training Logistic Regression...")
        elif progress == 60: 
            status_text.text("🌲 Training Random Forest...")
        elif progress == 80:
            status_text.text("⚡ Training XGBoost...")
        elif progress == 100:
            status_text.text("📈 Evaluating models...")
        
        time.sleep(2)  # Simulate processing time
    
    # Show completion message
    st.success("✅ Model training completed!")
    
    # Store results in session state for Results page
    st.session_state.training_completed = True
    st.session_state.selected_features = selected_features
    st.session_state.models_trained = {
        "Logistic Regression": use_lr,
        "Random Forest": use_rf,
        "XGBoost": use_xgb
    }
    
    # Show next steps
    st.balloons()
    st.info("🎉 Go to the **Results** page to see model performance and comparisons!")

# Information section
st.markdown("---")
st.header("💡 Training Information")

with st.expander("About the Algorithms"):
    st.markdown("""
    **Logistic Regression**: Fast, interpretable, good baseline model  
    **Random Forest**: Robust, handles non-linear relationships, feature importance  
    **XGBoost**: State-of-the-art, often highest performance, good with complex patterns
    """)

with st.expander("About Hyperparameter Tuning"):
    st.markdown("""
    **Auto-Tuning** automatically finds the best parameters for each model:
    - **Grid Search**: Tests all parameter combinations (thorough but slower)
    - **Random Search**: Tests random combinations (faster, often good enough)
    
    Tuning can significantly improve model performance!
    """)

# Footer
st.markdown("---")
st.caption("💡 **Tip**: Enable hyperparameter tuning for better model performance, but it will take longer to train.")

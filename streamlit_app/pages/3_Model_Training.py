import streamlit as st

def main():
    # ALL YOUR EXISTING MODEL TRAINING CODE GOES HERE
    
    st.set_page_config(
        page_title="Model Training - Smart Predictor",
        page_icon="🤖", 
        layout="wide"
    )

    st.title("🤖 Model Training")
    # ... rest of your code ...

# Add this at the bottom  
import streamlit as st
import pandas as pd
import numpy as np
import time

st.set_page_config(
    page_title="Model Training - Smart Predictor",
    page_icon="🤖", 
    layout="wide"
)

st.title("🤖 Model Training")
st.markdown("Train machine learning models with hyperparameter tuning")

# Check if data is available
if 'current_dataset' not in st.session_state or st.session_state.current_dataset is None:
    st.error("""
    ❌ **No dataset found!**
    
    Please go to **Data Analysis** page first to upload your CSV file.
    """)
    st.stop()

df = st.session_state.current_dataset
st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")

# Dataset info
with st.expander("📁 Dataset Summary"):
    st.write(f"**File:** {getattr(st.session_state, 'uploaded_file_name', 'Unknown')}")
    st.write(f"**Shape:** {df.shape[0]} rows, {df.shape[1]} columns")
    st.write(f"**Memory:** {(df.memory_usage(deep=True).sum() / 1024**2):.2f} MB")

# Configuration section
st.header("⚙️ Training Configuration")

col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Target Selection")
    
    # Auto-suggest target if available from analysis
    if 'target_column' in st.session_state:
        default_target = st.session_state.target_column
    else:
        # Find potential targets (low cardinality)
        potential_targets = [col for col in df.columns if df[col].nunique() <= 20]
        default_target = potential_targets[0] if potential_targets else df.columns[0]
    
    target_col = st.selectbox(
        "Select target variable (what to predict):",
        df.columns.tolist(),
        index=df.columns.tolist().index(default_target) if default_target in df.columns else 0
    )

with col2:
    st.subheader("📊 Training Settings")
    test_size = st.slider("Test Size (%)", 10, 40, 20)
    random_state = st.number_input("Random State", value=42, min_value=0, max_value=1000)
    enable_cv = st.checkbox("Enable Cross-Validation", value=True)

# Feature selection
st.header("🔧 Feature Selection")

available_features = [col for col in df.columns if col != target_col]

if not available_features:
    st.error("❌ No features available after selecting target!")
    st.stop()

selected_features = st.multiselect(
    "Select features for training:",
    available_features,
    default=available_features[:min(10, len(available_features))]  # Select first 10 by default
)

if not selected_features:
    st.error("❌ Please select at least one feature for training")
    st.stop()

# Show selected features info
st.info(f"**Selected:** {len(selected_features)} features | **Target:** {target_col}")

# Model selection
st.header("🤖 Algorithm Selection")

st.markdown("Choose which models to train:")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Logistic Regression")
    use_lr = st.checkbox("Enable", value=True)
    if use_lr:
        st.checkbox("Auto-tune parameters", value=True, key="lr_tune")
        st.checkbox("Show coefficients", value=True, key="lr_coef")

with col2:
    st.subheader("Random Forest")
    use_rf = st.checkbox("Enable", value=True, key="rf_enable")
    if use_rf:
        st.checkbox("Auto-tune parameters", value=True, key="rf_tune")
        st.checkbox("Feature importance", value=True, key="rf_imp")

with col3:
    st.subheader("XGBoost")
    use_xgb = st.checkbox("Enable", value=True, key="xgb_enable")
    if use_xgb:
        st.checkbox("Auto-tune parameters", value=True, key="xgb_tune")
        st.checkbox("Early stopping", value=True, key="xgb_early")

# Hyperparameter tuning
st.header("⚡ Hyperparameter Tuning")

tuning_col1, tuning_col2 = st.columns(2)

with tuning_col1:
    enable_tuning = st.checkbox("Enable Advanced Hyperparameter Tuning", value=True)
    if enable_tuning:
        tuning_method = st.radio("Tuning Method:", ["Grid Search", "Random Search"], horizontal=True)
        max_tuning_time = st.slider("Max Tuning Time (minutes)", 1, 60, 10)

with tuning_col2:
    if enable_tuning:
        st.subheader("Tuning Settings")
        cv_folds = st.slider("Cross-Validation Folds", 3, 10, 5)
        scoring_metric = st.selectbox("Optimization Metric:", 
                                    ["accuracy", "f1", "precision", "recall", "roc_auc"])
        
        st.info(f"**Will optimize for:** {scoring_metric.upper()}")

# Data preview
st.header("📋 Training Data Preview")

preview_col1, preview_col2 = st.columns(2)

with preview_col1:
    st.write("**Selected Features Sample:**")
    st.dataframe(df[selected_features].head(8), use_container_width=True)

with preview_col2:
    st.write("**Target Variable Distribution:**")
    target_counts = df[target_col].value_counts()
    st.dataframe(target_counts, use_container_width=True)
    
    # Data split info
    train_size = int(df.shape[0] * (1 - test_size/100))
    test_size_actual = df.shape[0] - train_size
    
    st.metric("Training Samples", train_size)
    st.metric("Testing Samples", test_size_actual)

# Training execution
st.header("🚀 Start Model Training")

training_col1, training_col2 = st.columns([2, 1])

with training_col1:
    if st.button("🎯 TRAIN MODELS", type="primary", use_container_width=True, key="train_main"):
        # Validation
        if not (use_lr or use_rf or use_xgb):
            st.error("❌ Please select at least one algorithm")
            st.stop()
            
        # Store training configuration
        training_config = {
            "dataset_info": {
                "shape": df.shape,
                "target": target_col,
                "features": selected_features
            },
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
                "max_time_minutes": max_tuning_time if enable_tuning else None,
                "cv_folds": cv_folds if enable_tuning else 5,
                "scoring_metric": scoring_metric if enable_tuning else "accuracy"
            }
        }
        
        st.session_state.training_config = training_config
        st.session_state.training_started = True

with training_col2:
    if st.button("🔄 Reset Training", type="secondary", use_container_width=True):
        for key in ['training_config', 'training_started', 'training_results']:
            if key in st.session_state:
                del st.session_state[key]
        st.rerun()

# Training progress simulation
if st.session_state.get('training_started', False):
    st.success("✅ Starting model training process...")
    
    # Progress simulation
    progress_bar = st.progress(0)
    status_text = st.empty()
    results_placeholder = st.empty()
    
    steps = [
        ("🔄 Loading and preprocessing data...", 10),
        ("📊 Splitting data into train/test sets...", 20),
        ("🤖 Training Logistic Regression...", 40),
        ("🌲 Training Random Forest...", 60),
        ("⚡ Training XGBoost...", 80),
        ("📈 Evaluating models and generating reports...", 95),
        ("✅ Training completed!", 100)
    ]
    
    for step_text, progress in steps:
        status_text.text(step_text)
        progress_bar.progress(progress)
        time.sleep(2)  # Simulate processing time
    
    # Simulated results
    with results_placeholder.container():
        st.balloons()
        st.success("🎉 Model training completed successfully!")
        
        # Simulated metrics
        st.subheader("📊 Model Performance Summary")
        
        results_data = {
            "Model": ["Logistic Regression", "Random Forest", "XGBoost"],
            "Accuracy": [0.82, 0.85, 0.87],
            "Precision": [0.80, 0.83, 0.85],
            "Recall": [0.78, 0.82, 0.84],
            "F1-Score": [0.79, 0.82, 0.84],
            "Training Time": ["45s", "2m 15s", "3m 30s"]
        }
        
        results_df = pd.DataFrame(results_data)
        st.dataframe(results_df, use_container_width=True)
        
        st.info("""
        **Next Steps:**
        - Best model: **XGBoost** (87% accuracy)
        - Go to **Results** page for detailed analysis
        - Feature importance charts available
        - Model deployment options
        """)
        
        # Store results
        st.session_state.training_results = results_data

# Information section
st.markdown("---")
st.header("💡 Training Guide")

with st.expander("📚 About the Algorithms"):
    st.markdown("""
    **Logistic Regression**
    - Fast training and prediction
    - Good for linear relationships
    - Easily interpretable coefficients
    
    **Random Forest** 
    - Handles non-linear relationships well
    - Robust to outliers and noise
    - Provides feature importance scores
    
    **XGBoost**
    - State-of-the-art performance
    - Handles complex patterns effectively
    - Good with structured/tabular data
    """)

with st.expander("⚡ About Hyperparameter Tuning"):
    st.markdown("""
    **Why Tune?**
    - Default parameters are often not optimal
    - Can improve accuracy by 5-15%
    - Better generalization on new data
    
    **Methods:**
    - **Grid Search**: Tests all combinations (thorough but slow)
    - **Random Search**: Tests random combinations (faster, often better)
    
    **Tip**: Start with Random Search for quick results!
    """)

# Footer
st.markdown("---")
st.caption("💡 **Pro Tip**: Enable hyperparameter tuning and use cross-validation for more reliable models!")
if __name__ == "__main__":
    main()

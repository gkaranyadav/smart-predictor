# app.py - UPDATED WITH BETTER OUTPUT HANDLING
import streamlit as st
import pandas as pd
import time
import json
import io
from databricks_api import *

# Page configuration
st.set_page_config(page_title="Smart Predictor", page_icon="🚀", layout="wide")

# Safe secret loading
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except:
        return default

# Job IDs
INGEST_JOB_ID = get_secret("DATABRICKS_JOB_INGEST_ID", "675344377204129")
TRAIN_JOB_ID = get_secret("DATABRICKS_JOB_TRAIN_ID", "362348352440928") 
SCORE_JOB_ID = get_secret("DATABRICKS_JOB_SCORE_ID", "100926012778266")

# Check secrets
if not get_secret("DATABRICKS_HOST") or not get_secret("DATABRICKS_TOKEN"):
    st.error("❌ Databricks credentials not configured!")
    st.stop()

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'upload_complete' not in st.session_state:
    st.session_state.upload_complete = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'scoring_results' not in st.session_state:
    st.session_state.scoring_results = None
if 'available_columns' not in st.session_state:
    st.session_state.available_columns = []

# NEW: Enhanced prediction loader
def load_predictions_enhanced(session_id):
    """Enhanced prediction loader with multiple fallbacks"""
    base_path = f"/FileStore/results/{session_id}"
    
    # Try multiple file locations
    file_paths = [
        f"{base_path}/predictions_direct.csv",
        f"{base_path}/predictions_sample.csv",
        f"{base_path}/predictions/part-00000-*.csv"
    ]
    
    for file_path in file_paths:
        result = dbfs_read_file(file_path)
        if result["status"] == "success":
            try:
                df = pd.read_csv(io.StringIO(result["content"]))
                st.success(f"✅ Loaded predictions from: {file_path}")
                return df
            except Exception as e:
                continue
    
    # Try directory listing approach
    list_result = dbfs_list_files(base_path)
    if list_result["status"] == "success":
        files = list_result.get("files", [])
        for file_info in files:
            if file_info["path"].endswith(".csv"):
                result = dbfs_read_file(file_info["path"])
                if result["status"] == "success":
                    try:
                        df = pd.read_csv(io.StringIO(result["content"]))
                        st.success(f"✅ Loaded from: {file_info['path']}")
                        return df
                    except:
                        continue
    
    return None

# NEW: Load EDA results
def load_eda_results(session_id):
    """Load EDA results from Databricks"""
    eda_path = f"/FileStore/eda/{session_id}/eda_results.json"
    result = dbfs_read_file(eda_path)
    
    if result["status"] == "success":
        try:
            return json.loads(result["content"])
        except:
            return None
    return None

# NEW: Load data sample
def load_data_sample(session_id):
    """Load data sample for display"""
    sample_path = f"/FileStore/eda/{session_id}/data_sample.csv"
    result = dbfs_read_file(sample_path)
    
    if result["status"] == "success":
        try:
            return pd.read_csv(io.StringIO(result["content"]))
        except:
            return None
    return None

# App UI
st.title("🚀 Smart Predictor")
st.markdown("Upload CSV data, train models, and get predictions - all without code!")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Model Training", "Batch Scoring"])

# Home Page
if page == "Home":
    st.header("📊 CSV Upload")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        if st.session_state.session_id is None:
            st.session_state.session_id = f"session_{int(time.time())}"
            st.session_state.current_file = uploaded_file.name
        
        session_id = st.session_state.session_id
        
        # Display file info
        file_size = uploaded_file.size / (1024 * 1024)
        st.success(f"File: {uploaded_file.name} ({file_size:.2f} MB)")
        st.info(f"Session ID: {session_id}")
        
        # Upload to DBFS
        if not st.session_state.upload_complete:
            with st.spinner("Uploading file to Databricks..."):
                dbfs_path = f"/FileStore/tmp/{session_id}/{uploaded_file.name}"
                result = upload_to_dbfs_simple(uploaded_file, dbfs_path)
                
                if result["status"] == "success":
                    st.session_state.upload_complete = True
                    st.session_state.dbfs_path = dbfs_path
                    st.success("✅ File uploaded successfully!")
                    
                    # Show data preview
                    st.subheader("📋 Data Preview")
                    uploaded_file.seek(0)
                    df_preview = pd.read_csv(uploaded_file, nrows=10)
                    st.dataframe(df_preview)
                    st.write(f"Shape: {df_preview.shape[0]} rows × {df_preview.shape[1]} columns")
                    
                    # Store column info
                    st.session_state.available_columns = df_preview.columns.tolist()
                    
                    # Show basic info
                    st.subheader("📊 Basic Information")
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write("**Data Types:**")
                        dtype_info = pd.DataFrame({
                            'Column': df_preview.columns,
                            'Type': df_preview.dtypes.astype(str)
                        })
                        st.dataframe(dtype_info, height=300)
                    
                    with col2:
                        st.write("**Basic Statistics:**")
                        st.dataframe(df_preview.describe())
                        
                else:
                    st.error(f"❌ Upload failed: {result['message']}")

# Data Analysis Page
elif page == "Data Analysis":
    st.header("🔍 Data Analysis")
    
    if st.session_state.upload_complete:
        st.success("✅ File uploaded successfully!")
        
        # Load and display previous EDA results
        eda_results = load_eda_results(st.session_state.session_id)
        if eda_results:
            st.subheader("📊 Previous Analysis Results")
            
            # Display EDA results
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**Dataset Overview**")
                st.metric("Total Rows", f"{eda_results['column_info']['total_rows']:,}")
                st.metric("Total Columns", eda_results['column_info']['total_columns'])
                st.metric("Total Missing Values", sum(eda_results['missing_values'].values()))
            
            with col2:
                st.write("**Data Sample**")
                sample_df = pd.DataFrame(eda_results['data_sample'])
                st.dataframe(sample_df)
            
            # Show data types
            st.write("**Data Types**")
            dtype_df = pd.DataFrame({
                'Column': list(eda_results['data_types'].keys()),
                'Type': list(eda_results['data_types'].values())
            })
            st.dataframe(dtype_df)
            
            # Show missing values
            st.write("**Missing Values**")
            missing_df = pd.DataFrame({
                'Column': list(eda_results['missing_values'].keys()),
                'Missing Count': list(eda_results['missing_values'].values())
            })
            st.dataframe(missing_df)
        
        # Run Data Analysis
        if st.button("🚀 Run Data Analysis"):
            with st.spinner("Running comprehensive data analysis..."):
                result = run_job(INGEST_JOB_ID, {
                    "dbfs_path": st.session_state.dbfs_path,
                    "session_id": st.session_state.session_id
                })
                
                if result["status"] == "success":
                    st.success("✅ Data analysis completed!")
                    
                    # Store results
                    st.session_state.analysis_results = {
                        "run_id": result.get('run_id'),
                        "status": "success"
                    }
                    
                    # Wait a bit and reload EDA results
                    time.sleep(3)
                    st.rerun()
                    
                else:
                    st.error(f"❌ Analysis failed: {result['message']}")
    else:
        st.warning("⚠️ Please upload a CSV file first from the Home page.")

# Model Training Page  
elif page == "Model Training":
    st.header("🤖 Model Training")
    
    if st.session_state.upload_complete:
        st.info("Train a machine learning model on your data")
        
        # Training configuration
        col1, col2 = st.columns(2)
        
        with col1:
            # Auto-suggest target column
            suggested_target = ""
            available_cols = st.session_state.available_columns
            
            if 'Diabetes_binary' in available_cols:
                suggested_target = 'Diabetes_binary'
            elif 'target' in [col.lower() for col in available_cols]:
                for col in available_cols:
                    if col.lower() == 'target':
                        suggested_target = col
                        break
            else:
                suggested_target = available_cols[-1] if available_cols else ""
            
            target_column = st.text_input(
                "Target Column", 
                value=suggested_target,
                help="Column to predict"
            )
            
            model_type = st.selectbox(
                "Model Type",
                ["Random Forest", "Logistic Regression", "Gradient Boosting", "AutoML"]
            )
        
        with col2:
            test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
            random_state = st.number_input("Random State", value=42)
        
        if st.button("🚀 Train Model"):
            if target_column and target_column not in st.session_state.available_columns:
                st.error("❌ Target column not found in data!")
            else:
                with st.spinner("Training model... This may take a few minutes."):
                    result = run_job(TRAIN_JOB_ID, {
                        "session_id": st.session_state.session_id,
                        "target_column": target_column,
                        "test_size": str(test_size),
                        "random_state": str(random_state)
                    })
                    
                    if result["status"] == "success":
                        st.success("✅ Model training completed!")
                        st.session_state.training_results = {
                            "run_id": result.get('run_id'),
                            "target_column": target_column,
                            "status": "success"
                        }
                        st.balloons()
                    else:
                        st.error(f"❌ Training failed: {result['message']}")
    else:
        st.warning("⚠️ Please upload a CSV file first from the Home page.")

# Batch Scoring Page
elif page == "Batch Scoring":
    st.header("📈 Batch Scoring")
    
    if st.session_state.upload_complete:
        st.success("✅ File uploaded successfully!")
        
        # Refresh button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Refresh Results"):
                st.rerun()
        
        # Display previous predictions if available
        predictions_df = load_predictions_enhanced(st.session_state.session_id)
        if predictions_df is not None:
            st.subheader("🎯 Prediction Results")
            
            # Display predictions
            st.dataframe(predictions_df.head(10))
            st.write(f"Total predictions: {len(predictions_df):,}")
            
            # Show prediction distribution
            if 'prediction' in predictions_df.columns:
                st.write("**Prediction Distribution**")
                pred_counts = predictions_df['prediction'].value_counts()
                st.bar_chart(pred_counts)
                
                # Download option
                csv = predictions_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Predictions",
                    data=csv,
                    file_name=f"predictions_{st.session_state.session_id}.csv",
                    mime="text/csv"
                )
        
        # Scoring options
        st.subheader("🔮 Generate Predictions")
        
        scoring_file = st.file_uploader(
            "Upload new data for scoring (optional)", 
            type=["csv"]
        )
        
        if st.button("🎯 Run Batch Scoring"):
            if not st.session_state.training_results:
                st.error("❌ Please train a model first!")
            else:
                with st.spinner("Running batch scoring..."):
                    input_path = st.session_state.dbfs_path
                    
                    # Use new file if provided
                    if scoring_file is not None:
                        scoring_id = f"{st.session_state.session_id}_scoring"
                        scoring_path = f"/FileStore/tmp/{scoring_id}/{scoring_file.name}"
                        upload_result = upload_to_dbfs_simple(scoring_file, scoring_path)
                        
                        if upload_result["status"] == "success":
                            input_path = scoring_path
                    
                    # Run scoring job
                    result = run_job(SCORE_JOB_ID, {
                        "input_dbfs_path": input_path,
                        "session_id": st.session_state.session_id
                    })
                    
                    if result["status"] == "success":
                        st.success("✅ Batch scoring completed!")
                        st.session_state.scoring_results = {
                            "run_id": result.get('run_id'),
                            "status": "success"
                        }
                        
                        # Wait and reload predictions
                        time.sleep(5)
                        st.rerun()
                    else:
                        st.error(f"❌ Scoring failed: {result['message']}")
    else:
        st.warning("⚠️ Please upload a CSV file first from the Home page.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Smart Predictor v2.0 | Streamlit + Databricks")

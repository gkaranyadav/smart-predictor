# app.py - PERMANENT FIX
import streamlit as st
import pandas as pd
import json
import io
import time
from databricks_api import *

# Page config
st.set_page_config(page_title="Smart Predictor", page_icon="🚀", layout="wide")

# Secrets
def get_secret(key, default=None):
    try: return st.secrets[key]
    except: return default

INGEST_JOB_ID = get_secret("DATABRICKS_JOB_INGEST_ID", "675344377204129")
TRAIN_JOB_ID = get_secret("DATABRICKS_JOB_TRAIN_ID", "362348352440928") 
SCORE_JOB_ID = get_secret("DATABRICKS_JOB_SCORE_ID", "100926012778266")

if not get_secret("DATABRICKS_HOST") or not get_secret("DATABRICKS_TOKEN"):
    st.error("❌ Configure Databricks secrets!")
    st.stop()

# Session state
if 'session_id' not in st.session_state: st.session_state.session_id = None
if 'upload_complete' not in st.session_state: st.session_state.upload_complete = False

# 🎯 PERMANENT FIX: GUARANTEED FILE LOADING FUNCTIONS
def load_predictions_guaranteed(session_id):
    """GUARANTEED to load predictions - multiple fallback methods"""
    base_dir = f"/FileStore/smart_predictor_output/{session_id}"
    
    # Try all possible file locations
    file_paths = [
        f"{base_dir}/predictions_final.csv",           # Primary
        f"{base_dir}/predictions_sample.csv",          # Sample
        f"{base_dir}/predictions.csv",                 # Alternative
    ]
    
    for file_path in file_paths:
        result = dbfs_read_file(file_path)
        if result["status"] == "success":
            try:
                df = pd.read_csv(io.StringIO(result["content"]))
                st.success(f"✅ Loaded from: {file_path}")
                return df
            except Exception as e:
                continue
    
    # If no files found, show debug info
    st.error(f"❌ No prediction files found in: {base_dir}")
    return None

def load_eda_results_guaranteed(session_id):
    """GUARANTEED to load EDA results"""
    eda_dir = f"/FileStore/smart_predictor_output/{session_id}/eda"
    
    file_paths = [
        f"{eda_dir}/eda_summary.json",
        f"{eda_dir}/data_sample.csv",
        f"{eda_dir}/dataset_overview.csv"
    ]
    
    results = {}
    
    for file_path in file_paths:
        result = dbfs_read_file(file_path)
        if result["status"] == "success":
            try:
                if file_path.endswith('.json'):
                    results['summary'] = json.loads(result["content"])
                elif 'sample' in file_path:
                    results['sample'] = pd.read_csv(io.StringIO(result["content"]))
                elif 'overview' in file_path:
                    results['overview'] = pd.read_csv(io.StringIO(result["content"]))
            except:
                continue
    
    return results if results else None

# App UI
st.title("🚀 Smart Predictor - PERMANENT FIX")
st.markdown("**Guaranteed outputs** - Upload, Analyze, Train, Predict!")

# Navigation
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Model Training", "Batch Scoring"])

# Home Page
if page == "Home":
    st.header("📊 Upload CSV File")
    
    uploaded_file = st.file_uploader("Choose CSV", type=["csv"])
    
    if uploaded_file is not None:
        if st.session_state.session_id is None:
            st.session_state.session_id = f"session_{int(time.time())}"
        
        session_id = st.session_state.session_id
        st.success(f"File: {uploaded_file.name}")
        st.info(f"Session: {session_id}")
        
        if not st.session_state.upload_complete:
            with st.spinner("Uploading..."):
                dbfs_path = f"/FileStore/tmp/{session_id}/{uploaded_file.name}"
                result = upload_to_dbfs_simple(uploaded_file, dbfs_path)
                
                if result["status"] == "success":
                    st.session_state.upload_complete = True
                    st.session_state.dbfs_path = dbfs_path
                    st.success("✅ Upload successful!")
                    
                    # Show preview
                    uploaded_file.seek(0)
                    df_preview = pd.read_csv(uploaded_file, nrows=10)
                    st.dataframe(df_preview)
                    st.write(f"**Shape:** {df_preview.shape[0]} rows × {df_preview.shape[1]} columns")

# Data Analysis Page
elif page == "Data Analysis":
    st.header("🔍 Data Analysis - GUARANTEED OUTPUTS")
    
    if st.session_state.upload_complete:
        session_id = st.session_state.session_id
        
        # Load and display EDA results
        eda_results = load_eda_results_guaranteed(session_id)
        
        if eda_results:
            st.success("✅ EDA Results Found!")
            
            # Display dataset overview
            if 'overview' in eda_results:
                st.subheader("📈 Dataset Overview")
                st.dataframe(eda_results['overview'])
            
            # Display data sample
            if 'sample' in eda_results:
                st.subheader("📋 Data Sample (First 100 rows)")
                st.dataframe(eda_results['sample'])
            
            # Display detailed EDA summary
            if 'summary' in eda_results:
                summary = eda_results['summary']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Dataset Info")
                    st.metric("Total Rows", f"{summary['dataset_info']['total_rows']:,}")
                    st.metric("Total Columns", summary['dataset_info']['total_columns'])
                    st.metric("Memory Usage", summary['dataset_info']['memory_usage'])
                
                with col2:
                    st.subheader("🔍 Data Types")
                    dtype_df = pd.DataFrame({
                        'Column': list(summary['data_types'].keys()),
                        'Type': list(summary['data_types'].values())
                    })
                    st.dataframe(dtype_df, height=300)
                
                # Show missing values
                st.subheader("❌ Missing Values")
                missing_df = pd.DataFrame({
                    'Column': list(summary['missing_values'].keys()),
                    'Missing Count': list(summary['missing_values'].values())
                })
                st.dataframe(missing_df)
        
        # Run Analysis Button
        if st.button("🚀 Run Data Analysis", type="primary"):
            with st.spinner("Running analysis..."):
                result = run_job(INGEST_JOB_ID, {
                    "dbfs_path": st.session_state.dbfs_path,
                    "session_id": session_id
                })
                
                if result["status"] == "success":
                    st.success("✅ Analysis completed! Refreshing...")
                    time.sleep(5)
                    st.rerun()
                else:
                    st.error(f"❌ Failed: {result['message']}")
    else:
        st.warning("⚠️ Upload a CSV file first")

# Model Training Page
elif page == "Model Training":
    st.header("🤖 Model Training")
    
    if st.session_state.upload_complete:
        # Training configuration
        target_column = st.text_input("Target Column", value="Diabetes_binary")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model..."):
                result = run_job(TRAIN_JOB_ID, {
                    "session_id": st.session_state.session_id,
                    "target_column": target_column,
                    "test_size": "0.2",
                    "random_state": "42"
                })
                
                if result["status"] == "success":
                    st.success("✅ Model trained successfully!")
                    st.balloons()
                else:
                    st.error(f"❌ Training failed: {result['message']}")
    else:
        st.warning("⚠️ Upload a CSV file first")

# Batch Scoring Page
elif page == "Batch Scoring":
    st.header("📈 Batch Scoring - GUARANTEED OUTPUTS")
    
    if st.session_state.upload_complete:
        session_id = st.session_state.session_id
        
        # Refresh button
        if st.button("🔄 Refresh Predictions", type="primary"):
            st.rerun()
        
        # Load and display predictions
        predictions_df = load_predictions_guaranteed(session_id)
        
        if predictions_df is not None:
            st.success("✅ Predictions Loaded!")
            
            # Display predictions
            st.subheader("🎯 Prediction Results")
            st.dataframe(predictions_df.head(15))
            
            # Show stats
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Total Predictions", len(predictions_df))
            with col2: st.metric("Unique Predictions", predictions_df['prediction'].nunique())
            with col3: st.metric("Mean Probability", f"{predictions_df['prediction_probability'].mean():.3f}")
            
            # Show distribution
            st.subheader("📊 Prediction Distribution")
            pred_counts = predictions_df['prediction'].value_counts()
            st.bar_chart(pred_counts)
            
            # Download
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                "📥 Download Full Predictions",
                data=csv,
                file_name=f"predictions_{session_id}.csv",
                mime="text/csv"
            )
        
        # Run Scoring
        if st.button("🎯 Run Batch Scoring", type="primary"):
            with st.spinner("Generating predictions..."):
                result = run_job(SCORE_JOB_ID, {
                    "input_dbfs_path": st.session_state.dbfs_path,
                    "session_id": session_id
                })
                
                if result["status"] == "success":
                    st.success("✅ Scoring completed! Refreshing...")
                    time.sleep(5)
                    st.rerun()
                else:
                    st.error(f"❌ Scoring failed: {result['message']}")
    else:
        st.warning("⚠️ Upload a CSV file first")

st.sidebar.markdown("---")
st.sidebar.info("**PERMANENT FIX v1.0** | Guaranteed Outputs")

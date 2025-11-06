# app.py - SIMPLIFIED AND ROBUST
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
    st.error("❌ Configure Databricks secrets in Streamlit Cloud!")
    st.stop()

# Session state
if 'session_id' not in st.session_state: st.session_state.session_id = None
if 'upload_complete' not in st.session_state: st.session_state.upload_complete = False

# Helper functions
def load_file_safe(file_path):
    """Safely load any file from DBFS"""
    result = dbfs_read_file(file_path)
    if result["status"] == "success":
        return result["content"]
    return None

def load_predictions(session_id):
    """Load predictions with multiple fallbacks"""
    base_dir = f"/FileStore/smart_predictor/{session_id}/predictions"
    
    # Try different file locations
    file_paths = [
        f"{base_dir}/predictions.csv",
        f"{base_dir}/predictions_sample.csv"
    ]
    
    for file_path in file_paths:
        content = load_file_safe(file_path)
        if content:
            try:
                df = pd.read_csv(io.StringIO(content))
                st.success(f"✅ Loaded from: {file_path}")
                return df
            except:
                continue
    
    st.error("❌ No predictions found")
    return None

def load_eda_results(session_id):
    """Load EDA results"""
    base_dir = f"/FileStore/smart_predictor/{session_id}"
    
    results = {}
    
    # Load EDA summary
    eda_content = load_file_safe(f"{base_dir}/eda_summary.json")
    if eda_content:
        results['summary'] = json.loads(eda_content)
    
    # Load data sample
    sample_content = load_file_safe(f"{base_dir}/data_sample.csv")
    if sample_content:
        results['sample'] = pd.read_csv(io.StringIO(sample_content))
    
    # Load numeric stats
    stats_content = load_file_safe(f"{base_dir}/numeric_stats.csv")
    if stats_content:
        results['stats'] = pd.read_csv(io.StringIO(stats_content))
    
    return results if results else None

# App UI
st.title("🚀 Smart Predictor - NO DELTA")
st.markdown("**Simple & Robust** - Upload CSV, Analyze, Train, Predict")

# Navigation
page = st.sidebar.radio("Go to", ["Upload Data", "Data Analysis", "Model Training", "Batch Scoring"])

# Upload Data Page
if page == "Upload Data":
    st.header("📊 Upload CSV File")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    
    if uploaded_file is not None:
        if st.session_state.session_id is None:
            st.session_state.session_id = f"session_{int(time.time())}"
        
        session_id = st.session_state.session_id
        st.success(f"File: {uploaded_file.name}")
        st.info(f"Session ID: {session_id}")
        
        if not st.session_state.upload_complete:
            with st.spinner("Uploading to Databricks..."):
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
                else:
                    st.error(f"❌ Upload failed: {result['message']}")

# Data Analysis Page
elif page == "Data Analysis":
    st.header("🔍 Data Analysis")
    
    if st.session_state.upload_complete:
        session_id = st.session_state.session_id
        
        # Load existing EDA results
        eda_results = load_eda_results(session_id)
        
        if eda_results:
            st.success("✅ EDA Results Found!")
            
            # Display data sample
            if 'sample' in eda_results:
                st.subheader("📋 Data Sample")
                st.dataframe(eda_results['sample'].head(15))
            
            # Display EDA summary
            if 'summary' in eda_results:
                summary = eda_results['summary']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Dataset Info")
                    st.metric("Total Rows", f"{summary['dataset_info']['total_rows']:,}")
                    st.metric("Total Columns", summary['dataset_info']['total_columns'])
                
                with col2:
                    st.subheader("🔍 Data Types")
                    for col, dtype in list(summary['data_types'].items())[:10]:
                        st.write(f"**{col}**: {dtype}")
            
            # Display statistics
            if 'stats' in eda_results:
                st.subheader("📈 Numeric Statistics")
                st.dataframe(eda_results['stats'])
        
        # Run Analysis Button
        if st.button("🚀 Run Data Analysis", type="primary"):
            with st.spinner("Analyzing data..."):
                result = run_job(INGEST_JOB_ID, {
                    "dbfs_path": st.session_state.dbfs_path,
                    "session_id": session_id
                })
                
                if result["status"] == "success":
                    st.success("✅ Analysis completed! Refreshing...")
                    time.sleep(5)
                    st.rerun()
                else:
                    st.error(f"❌ Analysis failed: {result['message']}")
    else:
        st.warning("⚠️ Please upload a CSV file first")

# Model Training Page
elif page == "Model Training":
    st.header("🤖 Model Training")
    
    if st.session_state.upload_complete:
        session_id = st.session_state.session_id
        
        # Simple training configuration
        target_column = st.text_input("Target Column", value="Diabetes_binary")
        
        if st.button("🚀 Train Model", type="primary"):
            with st.spinner("Training model... This may take a few minutes."):
                result = run_job(TRAIN_JOB_ID, {
                    "session_id": session_id,
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
        st.warning("⚠️ Please upload a CSV file first")

# Batch Scoring Page
elif page == "Batch Scoring":
    st.header("📈 Batch Scoring")
    
    if st.session_state.upload_complete:
        session_id = st.session_state.session_id
        
        # Refresh button
        if st.button("🔄 Refresh Results"):
            st.rerun()
        
        # Load and display predictions
        predictions_df = load_predictions(session_id)
        
        if predictions_df is not None:
            st.success("✅ Predictions Loaded!")
            
            # Display predictions
            st.subheader("🎯 Prediction Results")
            st.dataframe(predictions_df.head(15))
            
            # Show stats
            col1, col2 = st.columns(2)
            with col1: 
                st.metric("Total Predictions", len(predictions_df))
                st.metric("Unique Predictions", predictions_df['prediction'].nunique())
            with col2:
                st.metric("Mean Probability", f"{predictions_df['prediction_probability'].mean():.3f}")
            
            # Download
            csv = predictions_df.to_csv(index=False)
            st.download_button(
                "📥 Download Predictions",
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
        st.warning("⚠️ Please upload a CSV file first")

st.sidebar.markdown("---")
st.sidebar.info("**No Delta Tables** | Simple & Robust")

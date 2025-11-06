# app.py - COMPLETE WORKING VERSION
import streamlit as st
import pandas as pd
import json
import io
import time
from databricks_api import *
from utils import gen_session_id, safe_dbfs_path

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
if 'current_file' not in st.session_state: st.session_state.current_file = None
if 'analysis_results' not in st.session_state: st.session_state.analysis_results = None
if 'training_results' not in st.session_state: st.session_state.training_results = None
if 'scoring_results' not in st.session_state: st.session_state.scoring_results = None

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
st.title("🚀 Smart Predictor - COMPLETE WORKING VERSION")
st.markdown("**Large File Support** - Upload, Analyze, Train, Predict")

# Navigation
page = st.sidebar.radio("Go to", ["Upload Data", "Data Analysis", "Model Training", "Batch Scoring"])

# Upload Data Page
if page == "Upload Data":
    st.header("📊 Upload CSV File")
    
    uploaded_file = st.file_uploader("Choose CSV file", type=["csv"])
    
    if uploaded_file is not None:
        if st.session_state.session_id is None:
            st.session_state.session_id = gen_session_id()
            st.session_state.current_file = uploaded_file.name
        
        session_id = st.session_state.session_id
        
        # Display file info
        file_size = uploaded_file.size / (1024 * 1024)
        st.success(f"File: {uploaded_file.name} ({file_size:.2f} MB)")
        st.info(f"Session ID: {session_id}")
        
        if not st.session_state.upload_complete:
            with st.spinner("Uploading to Databricks..."):
                dbfs_path = safe_dbfs_path(session_id, uploaded_file.name)
                result = upload_to_dbfs_simple(uploaded_file, dbfs_path)
                
                if result["status"] == "success":
                    st.session_state.upload_complete = True
                    st.session_state.dbfs_path = dbfs_path
                    st.success(f"✅ {result['message']}")
                    
                    # Show preview
                    st.subheader("Data Preview")
                    uploaded_file.seek(0)
                    df_preview = pd.read_csv(uploaded_file, nrows=10)
                    st.dataframe(df_preview)
                    st.write(f"**Shape:** {df_preview.shape[0]} rows × {df_preview.shape[1]} columns")
                    
                    # Show basic info
                    st.subheader("Basic Information")
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
        session_id = st.session_state.session_id
        st.success("✅ File uploaded successfully!")
        
        # Load existing EDA results
        eda_results = load_eda_results(session_id)
        
        if eda_results:
            st.success("✅ EDA Results Found!")
            
            # Display data sample
            if 'sample' in eda_results:
                st.subheader("📋 Data Sample (First 1000 rows)")
                st.dataframe(eda_results['sample'].head(15))
            
            # Display EDA summary
            if 'summary' in eda_results:
                summary = eda_results['summary']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.subheader("📊 Dataset Overview")
                    st.metric("Total Rows", f"{summary['dataset_info']['total_rows']:,}")
                    st.metric("Total Columns", summary['dataset_info']['total_columns'])
                    st.metric("Total Missing", f"{sum(summary['missing_values'].values()):,}")
                
                with col2:
                    st.subheader("🔍 Column Types")
                    for col, dtype in list(summary['data_types'].items())[:10]:
                        st.write(f"**{col}**: `{dtype}`")
            
            # Display statistics
            if 'stats' in eda_results:
                st.subheader("📈 Numeric Statistics")
                st.dataframe(eda_results['stats'])
        
        # Run Analysis Button
        if st.button("🚀 Run Data Analysis", type="primary"):
            with st.spinner("Analyzing data... This may take a few minutes."):
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
        st.info("Train a machine learning model on your data")
        
        # Training configuration
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
                    st.session_state.training_results = {
                        "run_id": result.get('run_id'),
                        "target_column": target_column
                    }
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
        st.success("✅ File uploaded successfully!")
        
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
            col1, col2, col3 = st.columns(3)
            with col1: 
                st.metric("Total Predictions", len(predictions_df))
            with col2:
                st.metric("Unique Predictions", predictions_df['prediction'].nunique())
            with col3:
                st.metric("Mean Probability", f"{predictions_df['prediction_probability'].mean():.3f}")
            
            # Show distribution
            if 'prediction' in predictions_df.columns:
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
            if not st.session_state.training_results:
                st.error("❌ Please train a model first!")
            else:
                with st.spinner("Generating predictions... This may take a few minutes."):
                    result = run_job(SCORE_JOB_ID, {
                        "input_dbfs_path": st.session_state.dbfs_path,
                        "session_id": session_id
                    })
                    
                    if result["status"] == "success":
                        st.success("✅ Scoring completed! Refreshing...")
                        st.session_state.scoring_results = {
                            "run_id": result.get('run_id')
                        }
                        time.sleep(5)
                        st.rerun()
                    else:
                        st.error(f"❌ Scoring failed: {result['message']}")
    else:
        st.warning("⚠️ Please upload a CSV file first")

st.sidebar.markdown("---")
st.sidebar.info("**Working Version** | Large File Support")

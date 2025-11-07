# app.py - SIMPLE WORKING VERSION
import streamlit as st
import pandas as pd
import time
import json
import io
from databricks_api import upload_to_dbfs_simple, run_job, get_task_output
from utils import gen_session_id, safe_dbfs_path

# Page configuration
st.set_page_config(
    page_title="Smart Predictor",
    page_icon="🚀",
    layout="wide"
)

# Safe secret loading with fallbacks
def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        return default

# Job IDs
INGEST_JOB_ID = get_secret("DATABRICKS_JOB_INGEST_ID", "675344377204129")
TRAIN_JOB_ID = get_secret("DATABRICKS_JOB_TRAIN_ID", "362348352440928")
SCORE_JOB_ID = get_secret("DATABRICKS_JOB_SCORE_ID", "100926012778266")

# Check if secrets are configured
if not get_secret("DATABRICKS_HOST") or not get_secret("DATABRICKS_TOKEN"):
    st.error("Databricks credentials not configured!")
    st.stop()

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'upload_complete' not in st.session_state:
    st.session_state.upload_complete = False
if 'predictions_data' not in st.session_state:
    st.session_state.predictions_data = None

# SIMPLE: Get predictions from task output
def get_predictions_simple(run_id):
    """Simple prediction loader from task output"""
    try:
        output_result = get_task_output(run_id)
        
        if output_result["status"] == "success":
            logs = output_result.get("logs", "")
            
            # Look for our markers
            if "PREDICTIONS_JSON_START" in logs and "PREDICTIONS_JSON_END" in logs:
                start_idx = logs.find("PREDICTIONS_JSON_START") + len("PREDICTIONS_JSON_START")
                end_idx = logs.find("PREDICTIONS_JSON_END")
                
                if start_idx < end_idx:
                    json_str = logs[start_idx:end_idx].strip()
                    try:
                        predictions_data = json.loads(json_str)
                        
                        if predictions_data.get("status") == "success":
                            # Convert to DataFrame
                            df = pd.DataFrame(predictions_data["sample_predictions"])
                            st.session_state.prediction_metadata = predictions_data
                            st.success("✅ Predictions loaded successfully!")
                            return df
                    except Exception as e:
                        st.error(f"Error parsing predictions: {e}")
            
            # Debug: Check what's in logs
            if logs:
                st.info(f"Logs length: {len(logs)}")
                # Show relevant part of logs
                if "PREDICTIONS" in logs:
                    relevant_part = logs[logs.find("PREDICTIONS"):min(len(logs), logs.find("PREDICTIONS") + 1000)]
                    st.text_area("Relevant logs:", relevant_part, height=150)
        
        return None
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Display predictions
def display_predictions_simple(predictions_df, metadata):
    """Simple prediction display"""
    if predictions_df is None or len(predictions_df) == 0:
        st.error("No predictions to display")
        return
    
    st.subheader("🎯 Prediction Results")
    
    # Show stats
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Predictions", f"{metadata.get('total_predictions', 0):,}")
    with col2:
        st.metric("Sample Size", len(predictions_df))
    with col3:
        stats = metadata.get('prediction_stats', {})
        st.metric("Prediction 0", f"{stats.get('prediction_0', 0):,}")
    
    # Show sample data
    st.write("**Sample Predictions:**")
    st.dataframe(predictions_df)
    
    # Show distribution
    if 'prediction' in predictions_df.columns:
        st.write("**Prediction Distribution:**")
        pred_counts = predictions_df['prediction'].value_counts()
        st.bar_chart(pred_counts)
    
    # Download button
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample Predictions",
        data=csv,
        file_name=f"predictions_{metadata.get('session_id', 'sample')}.csv",
        mime="text/csv"
    )

# App
st.title("🚀 Smart Predictor")
st.markdown("Upload CSV → Train Model → Get Predictions")

# Sidebar
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Upload Data", "Train Model", "Get Predictions"])

# Upload Data
if page == "Upload Data":
    st.header("📊 Upload CSV File")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        if st.session_state.session_id is None:
            st.session_state.session_id = gen_session_id()
        
        session_id = st.session_state.session_id
        file_size = uploaded_file.size / (1024 * 1024)
        
        st.success(f"File: {uploaded_file.name} ({file_size:.2f} MB)")
        st.info(f"Session ID: {session_id}")
        
        if not st.session_state.upload_complete:
            with st.spinner("Uploading file..."):
                dbfs_path = safe_dbfs_path(session_id, uploaded_file.name)
                result = upload_to_dbfs_simple(uploaded_file, dbfs_path)
                
                if result["status"] == "success":
                    st.session_state.upload_complete = True
                    st.session_state.dbfs_path = dbfs_path
                    st.success("✅ File uploaded successfully!")
                    
                    # Show preview
                    uploaded_file.seek(0)
                    df_preview = pd.read_csv(uploaded_file, nrows=5)
                    st.write("**Data Preview:**")
                    st.dataframe(df_preview)
                else:
                    st.error(f"Upload failed: {result['message']}")

# Train Model
elif page == "Train Model":
    st.header("🤖 Train Model")
    
    if st.session_state.upload_complete:
        st.success("✅ File uploaded successfully!")
        
        # Simple training form
        target_column = st.text_input("Target Column", value="Diabetes_binary")
        
        if st.button("🚀 Train Model"):
            with st.spinner("Training model..."):
                result = run_job(TRAIN_JOB_ID, {
                    "session_id": st.session_state.session_id,
                    "target_column": target_column,
                    "test_size": "0.2",
                    "random_state": "42"
                })
                
                if result["status"] == "success":
                    st.success(f"✅ Model trained! Run ID: {result.get('run_id')}")
                    st.balloons()
                else:
                    st.error(f"Training failed: {result['message']}")
    else:
        st.warning("Please upload a CSV file first.")

# Get Predictions
elif page == "Get Predictions":
    st.header("📈 Get Predictions")
    
    if st.session_state.upload_complete:
        st.success("✅ File uploaded successfully!")
        
        # Show previous predictions if available
        if st.session_state.predictions_data is not None:
            st.subheader("📊 Previous Predictions")
            display_predictions_simple(
                st.session_state.predictions_data, 
                st.session_state.get('prediction_metadata', {})
            )
        
        # Option to score new data
        scoring_file = st.file_uploader(
            "Upload new data for scoring (optional)", 
            type=["csv"]
        )
        
        if st.button("🎯 Get Predictions"):
            with st.spinner("Generating predictions..."):
                # Use new file or original file
                input_dbfs_path = st.session_state.dbfs_path
                if scoring_file is not None:
                    scoring_session_id = st.session_state.session_id + "_scoring"
                    scoring_dbfs_path = safe_dbfs_path(scoring_session_id, scoring_file.name)
                    upload_result = upload_to_dbfs_simple(scoring_file, scoring_dbfs_path)
                    if upload_result["status"] == "success":
                        input_dbfs_path = scoring_dbfs_path
                
                # Run scoring job
                result = run_job(SCORE_JOB_ID, {
                    "input_dbfs_path": input_dbfs_path,
                    "session_id": st.session_state.session_id
                })
                
                if result["status"] == "success":
                    run_id = result.get('run_id')
                    st.success(f"✅ Scoring completed! Run ID: {run_id}")
                    
                    # Get predictions immediately
                    with st.spinner("Loading predictions..."):
                        time.sleep(5)  # Wait a bit
                        predictions_df = get_predictions_simple(run_id)
                        
                        if predictions_df is not None:
                            st.session_state.predictions_data = predictions_df
                            display_predictions_simple(
                                predictions_df, 
                                st.session_state.prediction_metadata
                            )
                            st.balloons()
                        else:
                            st.info("💡 Predictions generated! Try refreshing in a moment.")
                else:
                    st.error(f"Scoring failed: {result['message']}")
    else:
        st.warning("Please upload a CSV file first.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info("Smart Predictor v1.0")

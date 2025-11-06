# app.py
import streamlit as st
import pandas as pd
import time
import json
from databricks_api import (
    dbfs_put_single, dbfs_upload_chunked, upload_to_dbfs_simple, 
    run_job, get_job_output, get_task_runs, get_task_output, 
    get_run_details, dbfs_read_file, dbfs_file_exists
)
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

# Job IDs from Streamlit Cloud secrets with fallbacks
INGEST_JOB_ID = get_secret("DATABRICKS_JOB_INGEST_ID", "675344377204129")
TRAIN_JOB_ID = get_secret("DATABRICKS_JOB_TRAIN_ID", "362348352440928")
SCORE_JOB_ID = get_secret("DATABRICKS_JOB_SCORE_ID", "100926012778266")

# Check if secrets are configured
if not get_secret("DATABRICKS_HOST") or not get_secret("DATABRICKS_TOKEN"):
    st.error("""
    ❌ **Databricks credentials not configured!**
    
    Please add these secrets in Streamlit Cloud:
    1. Go to your app settings
    2. Click on 'Secrets' 
    3. Add:
    
    ```toml
    DATABRICKS_HOST = "https://dbc-484c2988-d6e6.cloud.databricks.com"
    DATABRICKS_TOKEN = "dapi1234567890abcdef..."
    DATABRICKS_JOB_INGEST_ID = "675344377204129"
    DATABRICKS_JOB_TRAIN_ID = "362348352440928"
    DATABRICKS_JOB_SCORE_ID = "100926012778266"
    ```
    """)
    st.stop()

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'upload_complete' not in st.session_state:
    st.session_state.upload_complete = False
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'training_results' not in st.session_state:
    st.session_state.training_results = None
if 'scoring_results' not in st.session_state:
    st.session_state.scoring_results = None

# App title and description
st.title("🚀 Smart Predictor")
st.markdown("""
Upload your CSV data, train machine learning models, and get predictions - all without writing code!
""")

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Analysis", "Model Training", "Batch Scoring"])

# Home Page
if page == "Home":
    st.header("📊 CSV Upload")
    
    uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])
    
    if uploaded_file is not None:
        # Generate session ID for this upload
        if st.session_state.session_id is None:
            st.session_state.session_id = gen_session_id()
            st.session_state.current_file = uploaded_file.name
        
        session_id = st.session_state.session_id
        
        # Display file info
        file_size = uploaded_file.size / (1024 * 1024)  # Size in MB
        st.success(f"File: {uploaded_file.name} ({file_size:.2f} MB)")
        st.info(f"Session ID: {session_id}")
        
        # Upload file to DBFS
        if not st.session_state.upload_complete:
            with st.spinner("Uploading file to Databricks..."):
                dbfs_path = safe_dbfs_path(session_id, uploaded_file.name)
                
                # Use the simple upload method that handles both small and large files
                result = upload_to_dbfs_simple(uploaded_file, dbfs_path)
                
                if result["status"] == "success":
                    st.session_state.upload_complete = True
                    st.session_state.dbfs_path = dbfs_path
                    st.success(f"✅ {result['message']}")
                    
                    # Show preview of uploaded data
                    st.subheader("Data Preview")
                    try:
                        uploaded_file.seek(0)  # Reset file pointer
                        df_preview = pd.read_csv(uploaded_file, nrows=5)
                        st.dataframe(df_preview)
                        st.write(f"Shape: {len(df_preview)} rows x {len(df_preview.columns)} columns")
                        
                        # Show column names and types
                        st.subheader("Data Types")
                        col_info = pd.DataFrame({
                            'Column': df_preview.columns,
                            'Type': [str(dtype) for dtype in df_preview.dtypes]
                        })
                        st.dataframe(col_info)
                        
                        # Show basic statistics
                        st.subheader("Basic Statistics")
                        st.dataframe(df_preview.describe())
                        
                    except Exception as e:
                        st.error(f"Error previewing data: {str(e)}")
                else:
                    st.error(f"❌ {result['message']}")

# Data Analysis Page
elif page == "Data Analysis":
    st.header("🔍 Data Analysis")
    
    if st.session_state.upload_complete:
        st.success("✅ File uploaded successfully!")
        
        # Display previous results if available
        if st.session_state.analysis_results:
            st.subheader("📊 Previous Analysis Results")
            
            # Display task outputs if available
            if "task_outputs" in st.session_state.analysis_results:
                task_outputs = st.session_state.analysis_results["task_outputs"]
                for task_key, task_data in task_outputs.items():
                    with st.expander(f"Task: {task_key}"):
                        if "output" in task_data:
                            output = task_data["output"]
                            if "notebook_output" in output and output["notebook_output"]:
                                st.write("Notebook Output:")
                                st.code(str(output["notebook_output"]))
                            if "logs" in output and output["logs"]:
                                st.write("Logs:")
                                st.text_area(f"Logs - {task_key}", output["logs"], height=150, key=f"logs_{task_key}")
        
        # Trigger Ingest Job for EDA
        if st.button("Run Data Analysis"):
            with st.spinner("Running data analysis..."):
                # USING JOB PARAMETERS (not notebook_params)
                result = run_job(INGEST_JOB_ID, {
                    "dbfs_path": st.session_state.dbfs_path,
                    "session_id": st.session_state.session_id
                })
                
                if result["status"] == "success":
                    st.success(f"✅ Data analysis completed! Run ID: {result.get('run_id', 'N/A')}")
                    
                    # Get job output and display results
                    run_id = result.get('run_id')
                    if run_id:
                        with st.spinner("Fetching analysis results..."):
                            # First try to get task outputs (for multi-task jobs)
                            task_result = get_task_runs(run_id)
                            
                            if task_result["status"] == "success":
                                st.session_state.analysis_results = task_result
                                
                                st.subheader("📈 Analysis Results")
                                
                                # Display task information
                                task_outputs = task_result.get("task_outputs", {})
                                run_info = task_result.get("run_info", {})
                                
                                st.info(f"Job completed with {len(task_outputs)} task(s)")
                                
                                # Display each task output
                                for task_key, task_data in task_outputs.items():
                                    with st.expander(f"📋 Task: {task_key}", expanded=True):
                                        if "output" in task_data:
                                            output = task_data["output"]
                                            
                                            # Display notebook output
                                            notebook_output = output.get("notebook_output", {})
                                            if notebook_output:
                                                result_text = notebook_output.get("result", "")
                                                if result_text:
                                                    st.success(f"Result: {result_text}")
                                            
                                            # Display logs
                                            logs = output.get("logs", "")
                                            if logs:
                                                st.write("Execution Logs:")
                                                st.text_area("Logs", logs, height=200, key=f"logs_{task_key}")
                                            
                                            # Display metadata
                                            metadata = output.get("metadata", {})
                                            if metadata:
                                                st.write("Task Metadata:")
                                                st.json(metadata)
                                        
                                        elif "error" in task_data:
                                            st.error(f"Error: {task_data['error']}")
                                
                                # Try to get sample data
                                sample_path = f"/FileStore/tmp/{st.session_state.session_id}/sample.csv"
                                if dbfs_file_exists(sample_path):
                                    sample_result = dbfs_read_file(sample_path)
                                    if sample_result["status"] == "success":
                                        try:
                                            # Display sample data
                                            from io import StringIO
                                            sample_df = pd.read_csv(StringIO(sample_result["content"]))
                                            st.subheader("📊 Sample Data (First 10K rows)")
                                            st.dataframe(sample_df.head(10))
                                            st.write(f"Sample shape: {sample_df.shape}")
                                            
                                            # Show basic statistics
                                            st.subheader("📈 Basic Statistics")
                                            st.dataframe(sample_df.describe())
                                            
                                        except Exception as e:
                                            st.error(f"Error displaying sample data: {str(e)}")
                                
                                st.balloons()
                                
                            else:
                                # Fallback: try direct output (for single-task jobs)
                                st.warning("Multi-task job detected, trying alternative approach...")
                                output_result = get_job_output(run_id)
                                
                                if output_result["status"] == "success":
                                    st.session_state.analysis_results = output_result
                                    st.subheader("📈 Analysis Results")
                                    
                                    # Display notebook output
                                    notebook_output = output_result.get("notebook_output", {})
                                    if notebook_output:
                                        st.write("Notebook Output:")
                                        st.code(str(notebook_output))
                                    
                                    # Display logs
                                    logs = output_result.get("logs", "")
                                    if logs:
                                        st.write("Job Logs:")
                                        st.text_area("Logs", logs, height=200)
                                    
                                    st.balloons()
                                else:
                                    st.error(f"❌ Failed to get job output: {output_result['message']}")
                    else:
                        st.info("Analysis completed. Check Databricks workspace for detailed results.")
                else:
                    st.error(f"❌ Data analysis failed: {result['message']}")
    else:
        st.warning("⚠️ Please upload a CSV file first from the Home page.")

# Model Training Page
elif page == "Model Training":
    st.header("🤖 Model Training")
    
    if st.session_state.upload_complete:
        st.info("Train a machine learning model on your uploaded data")
        
        # Display previous training results if available
        if st.session_state.training_results:
            st.subheader("📊 Previous Training Results")
            
            if "task_outputs" in st.session_state.training_results:
                task_outputs = st.session_state.training_results["task_outputs"]
                for task_key, task_data in task_outputs.items():
                    with st.expander(f"Task: {task_key}"):
                        if "output" in task_data:
                            output = task_data["output"]
                            if "notebook_output" in output and output["notebook_output"]:
                                st.code(str(output["notebook_output"]))
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Configuration")
            model_type = st.selectbox(
                "Model Type",
                ["AutoML (Recommended)", "Random Forest", "Logistic Regression", "Gradient Boosting"]
            )
            target_column = st.text_input("Target Column (leave empty for auto-detection)", "")
        
        with col2:
            st.subheader("Advanced Options")
            test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2)
            random_state = st.number_input("Random State", value=42)
        
        if st.button("🚀 Train Model"):
            with st.spinner("Training model... This may take several minutes."):
                # USING JOB PARAMETERS (not notebook_params)
                result = run_job(TRAIN_JOB_ID, {
                    "session_id": st.session_state.session_id,
                    "target_column": target_column if target_column else "",
                    "test_size": str(test_size),
                    "random_state": str(random_state)
                })
                
                if result["status"] == "success":
                    st.success(f"✅ Model training completed! Run ID: {result.get('run_id', 'N/A')}")
                    
                    # Get training results
                    run_id = result.get('run_id')
                    if run_id:
                        with st.spinner("Fetching training results..."):
                            # Try task outputs first (for multi-task jobs)
                            task_result = get_task_runs(run_id)
                            
                            if task_result["status"] == "success":
                                st.session_state.training_results = task_result
                                
                                st.subheader("🎯 Training Results")
                                st.success("Model trained and registered in MLflow!")
                                
                                # Display task outputs
                                task_outputs = task_result.get("task_outputs", {})
                                for task_key, task_data in task_outputs.items():
                                    with st.expander(f"Training Task: {task_key}"):
                                        if "output" in task_data:
                                            output = task_data["output"]
                                            notebook_output = output.get("notebook_output", {})
                                            if notebook_output:
                                                st.code(str(notebook_output))
                                
                                # Display model info
                                model_name = f"smart_predictor_model_{st.session_state.session_id}"
                                st.write(f"**Model Name:** {model_name}")
                                st.write(f"**Model URI:** models:/{model_name}/latest")
                                
                                st.balloons()
                            else:
                                # Fallback to direct output
                                output_result = get_job_output(run_id)
                                if output_result["status"] == "success":
                                    st.session_state.training_results = output_result
                                    st.subheader("🎯 Training Results")
                                    st.success("Model trained and registered in MLflow!")
                                    
                                    # Display model info
                                    model_name = f"smart_predictor_model_{st.session_state.session_id}"
                                    st.write(f"**Model Name:** {model_name}")
                                    st.write(f"**Model URI:** models:/{model_name}/latest")
                                    
                                    # Display notebook output
                                    notebook_output = output_result.get("notebook_output", {})
                                    if notebook_output:
                                        st.write("Training Output:")
                                        st.code(str(notebook_output))
                                    
                                    st.balloons()
                                else:
                                    st.error(f"❌ Failed to get training results: {output_result['message']}")
                else:
                    st.error(f"❌ Model training failed: {result['message']}")
    else:
        st.warning("⚠️ Please upload a CSV file first from the Home page.")

# Batch Scoring Page
elif page == "Batch Scoring":
    st.header("📈 Batch Scoring")
    
    if st.session_state.upload_complete:
        st.info("Generate predictions using your trained model")
        
        # Display previous scoring results if available
        if st.session_state.scoring_results:
            st.subheader("📊 Previous Scoring Results")
            
            if "task_outputs" in st.session_state.scoring_results:
                task_outputs = st.session_state.scoring_results["task_outputs"]
                for task_key, task_data in task_outputs.items():
                    with st.expander(f"Task: {task_key}"):
                        if "output" in task_data:
                            output = task_data["output"]
                            if "notebook_output" in output and output["notebook_output"]:
                                st.code(str(output["notebook_output"]))
        
        # Option to upload new data for scoring or use existing
        scoring_file = st.file_uploader(
            "Upload new data for scoring (optional - uses training data if not provided)", 
            type=["csv"]
        )
        
        if st.button("🎯 Run Batch Scoring"):
            with st.spinner("Running batch scoring..."):
                # Use new file if provided, otherwise use original file
                input_dbfs_path = st.session_state.dbfs_path
                if scoring_file is not None:
                    # Upload scoring file
                    scoring_dbfs_path = safe_dbfs_path(
                        st.session_state.session_id + "_scoring", 
                        scoring_file.name
                    )
                    
                    upload_result = upload_to_dbfs_simple(scoring_file, scoring_dbfs_path)
                    
                    if upload_result["status"] == "success":
                        input_dbfs_path = scoring_dbfs_path
                    else:
                        st.error(f"❌ Scoring file upload failed: {upload_result['message']}")
                
                # USING JOB PARAMETERS (not notebook_params)
                result = run_job(SCORE_JOB_ID, {
                    "input_dbfs_path": input_dbfs_path,
                    "session_id": st.session_state.session_id
                })
                
                if result["status"] == "success":
                    st.success(f"✅ Batch scoring completed! Run ID: {result.get('run_id', 'N/A')}")
                    
                    # Get scoring results
                    run_id = result.get('run_id')
                    if run_id:
                        with st.spinner("Fetching scoring results..."):
                            # Try task outputs first (for multi-task jobs)
                            task_result = get_task_runs(run_id)
                            
                            if task_result["status"] == "success":
                                st.session_state.scoring_results = task_result
                                
                                st.subheader("🎯 Scoring Results")
                                st.success("Predictions generated successfully!")
                                
                                # Display task outputs
                                task_outputs = task_result.get("task_outputs", {})
                                for task_key, task_data in task_outputs.items():
                                    with st.expander(f"Scoring Task: {task_key}"):
                                        if "output" in task_data:
                                            output = task_data["output"]
                                            notebook_output = output.get("notebook_output", {})
                                            if notebook_output:
                                                st.code(str(notebook_output))
                                
                                # Display predictions info
                                predictions_path = f"/FileStore/results/{st.session_state.session_id}/predictions.csv"
                                st.write(f"**Predictions saved to:** {predictions_path}")
                                
                                # Try to download and display predictions
                                if dbfs_file_exists(predictions_path):
                                    pred_result = dbfs_read_file(predictions_path)
                                    if pred_result["status"] == "success":
                                        try:
                                            # Read the CSV content
                                            from io import StringIO
                                            predictions_df = pd.read_csv(StringIO(pred_result["content"]))
                                            st.write("**Sample Predictions:**")
                                            st.dataframe(predictions_df.head(10))
                                            
                                            # Show prediction distribution
                                            if 'prediction' in predictions_df.columns:
                                                st.write("**Prediction Distribution:**")
                                                pred_counts = predictions_df['prediction'].value_counts()
                                                st.bar_chart(pred_counts)
                                                
                                            # Download button
                                            csv = predictions_df.to_csv(index=False)
                                            st.download_button(
                                                label="📥 Download Predictions",
                                                data=csv,
                                                file_name=f"predictions_{st.session_state.session_id}.csv",
                                                mime="text/csv"
                                            )
                                        except Exception as e:
                                            st.error(f"Error displaying predictions: {str(e)}")
                                
                                st.balloons()
                            else:
                                # Fallback to direct output
                                output_result = get_job_output(run_id)
                                if output_result["status"] == "success":
                                    st.session_state.scoring_results = output_result
                                    st.subheader("🎯 Scoring Results")
                                    st.success("Predictions generated successfully!")
                                    
                                    # Display predictions info
                                    predictions_path = f"/FileStore/results/{st.session_state.session_id}/predictions.csv"
                                    st.write(f"**Predictions saved to:** {predictions_path}")
                                    
                                    # Try to download and display predictions
                                    if dbfs_file_exists(predictions_path):
                                        pred_result = dbfs_read_file(predictions_path)
                                        if pred_result["status"] == "success":
                                            try:
                                                # Read the CSV content
                                                from io import StringIO
                                                predictions_df = pd.read_csv(StringIO(pred_result["content"]))
                                                st.write("**Sample Predictions:**")
                                                st.dataframe(predictions_df.head(10))
                                                
                                                # Show prediction distribution
                                                if 'prediction' in predictions_df.columns:
                                                    st.write("**Prediction Distribution:**")
                                                    pred_counts = predictions_df['prediction'].value_counts()
                                                    st.bar_chart(pred_counts)
                                                    
                                                # Download button
                                                csv = predictions_df.to_csv(index=False)
                                                st.download_button(
                                                    label="📥 Download Predictions",
                                                    data=csv,
                                                    file_name=f"predictions_{st.session_state.session_id}.csv",
                                                    mime="text/csv"
                                                )
                                            except Exception as e:
                                                st.error(f"Error displaying predictions: {str(e)}")
                                    
                                    st.balloons()
                                else:
                                    st.error(f"❌ Failed to get scoring results: {output_result['message']}")
                else:
                    st.error(f"❌ Batch scoring failed: {result['message']}")
    else:
        st.warning("⚠️ Please upload a CSV file first from the Home page.")

# Footer
st.sidebar.markdown("---")
st.sidebar.info(
    "Smart Predictor v1.0 | "
    "Streamlit Frontend + Databricks Backend"
)

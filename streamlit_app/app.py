# app.py - COMPLETE UPDATED VERSION WITH FIXED PARSING
import streamlit as st
import pandas as pd
import time
import json
import io
from databricks_api import dbfs_put_single, dbfs_upload_chunked, upload_to_dbfs_simple, run_job, get_task_output, dbfs_read_file, dbfs_file_exists, dbfs_list_files
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
if 'available_columns' not in st.session_state:
    st.session_state.available_columns = []
if 'predictions_data' not in st.session_state:
    st.session_state.predictions_data = None
if 'prediction_metadata' not in st.session_state:
    st.session_state.prediction_metadata = None

# NEW: Get predictions from TASK output (for multi-task jobs)
def get_predictions_from_job_output(run_id):
    """Get predictions directly from task output instead of job output"""
    try:
        output_result = get_task_output(run_id)  # Use get_task_output for multi-task jobs
        
        if output_result["status"] == "success":
            logs = output_result.get("logs", "")
            
            # ✅ FIXED: Look for our markers
            if "=== STREAMLIT_PREDICTIONS ===" in logs:
                start_idx = logs.find("=== STREAMLIT_PREDICTIONS ===") + len("=== STREAMLIT_PREDICTIONS ===")
                end_idx = logs.find("=== END_STREAMLIT_PREDICTIONS ===")
                
                if start_idx != -1 and end_idx != -1 and start_idx < end_idx:
                    json_str = logs[start_idx:end_idx].strip()
                    try:
                        predictions_data = json.loads(json_str)
                        
                        # Convert back to DataFrame
                        if "sample_predictions" in predictions_data:
                            df = pd.DataFrame(predictions_data["sample_predictions"])
                            
                            # Add metadata to session state
                            st.session_state.prediction_metadata = {
                                "total_predictions": predictions_data.get("total_predictions", 0),
                                "prediction_stats": predictions_data.get("prediction_stats", {}),
                                "timestamp": predictions_data.get("timestamp", ""),
                                "target_column": predictions_data.get("target_column", ""),
                                "model_run_id": predictions_data.get("model_run_id", "")
                            }
                            
                            st.success("✅ Predictions loaded from task output!")
                            return df
                    except Exception as e:
                        st.error(f"Error parsing predictions: {e}")
                        return None
            
            # Debug: Show what we actually got
            st.warning("No predictions found in task output")
            st.info(f"Debug - Logs length: {len(logs)}")
            if len(logs) > 0:
                # Show last part of logs to see what markers are there
                last_1000 = logs[-1000:]
                st.text_area("Last 1000 chars of logs:", last_1000, height=200)
                if "STREAMLIT" in last_1000:
                    st.info("Found STREAMLIT marker in logs!")
            return None
        else:
            st.error(f"Failed to get task output: {output_result['message']}")
            return None
            
    except Exception as e:
        st.error(f"Error getting task output: {str(e)}")
        return None

# Enhanced debug function
def enhanced_debug_prediction_files(session_id):
    """Enhanced debug function to find prediction files"""
    st.subheader("🔍 Enhanced File Debug")
    
    # Try ALL possible path formats
    base_paths = [
        f"/FileStore/results/{session_id}",
        f"/FileStore/tmp/{session_id}",
    ]
    
    found_files = []
    
    for base_path in base_paths:
        st.write(f"**Checking:** `{base_path}`")
        
        # Try to list directory
        list_result = dbfs_list_files(base_path)
        if list_result["status"] == "success":
            files = list_result.get("files", [])
            st.success(f"✅ FOUND! {len(files)} files in {base_path}")
            
            for file_info in files:
                file_path = file_info['path']
                file_size = file_info.get('file_size', 0)
                st.write(f"- `{file_path}` (size: {file_size} bytes)")
                
                # Try to read CSV files
                if file_path.endswith('.csv') and not file_info['is_dir']:
                    file_result = dbfs_read_file(file_path)
                    if file_result["status"] == "success":
                        st.success(f"✅ CAN READ: {file_path}")
                        try:
                            df = pd.read_csv(io.StringIO(file_result["content"]))
                            st.write(f"📊 Data: {len(df)} rows, {len(df.columns)} columns")
                            st.dataframe(df.head(3))
                            found_files.append({
                                'path': file_path,
                                'data': df,
                                'size': file_size
                            })
                        except Exception as e:
                            st.error(f"❌ Parse error: {e}")
                    else:
                        st.error(f"❌ Cannot read: {file_result['message']}")
                elif file_path.endswith('.txt'):
                    # Read success markers
                    file_result = dbfs_read_file(file_path)
                    if file_result["status"] == "success":
                        st.info(f"📝 {file_path}: {file_result['content'][:100]}...")
        else:
            st.error(f"❌ Directory not found: {list_result['message']}")
    
    if found_files:
        st.success(f"🎯 Found {len(found_files)} prediction files!")
        return found_files[0]['data']  # Return first successful dataframe
    else:
        st.error("🚨 NO PREDICTION FILES FOUND IN ANY LOCATION!")
        return None

# Smart prediction loader
def load_predictions_smart(session_id):
    """Smart prediction loader with multiple fallbacks"""
    st.info("🔍 Searching for prediction files...")
    
    # Try multiple file patterns in order of preference
    file_patterns = [
        f"/FileStore/results/{session_id}/predictions.csv",
        f"/FileStore/results/{session_id}/predictions_sample.csv",
        f"/FileStore/results/{session_id}/predictions_direct.csv",
        f"/FileStore/tmp/{session_id}/predictions.csv",
    ]
    
    for file_path in file_patterns:
        result = dbfs_read_file(file_path)
        if result["status"] == "success":
            try:
                df = pd.read_csv(io.StringIO(result["content"]))
                st.success(f"✅ Loaded from: {file_path}")
                return df
            except Exception as e:
                st.warning(f"⚠️ Could not parse {file_path}: {e}")
                continue
    
    # If direct files not found, try enhanced debug
    st.warning("⚠️ Direct file access failed. Starting enhanced search...")
    return enhanced_debug_prediction_files(session_id)

# Display predictions
def display_predictions(predictions_df, session_id):
    """Display predictions with download option"""
    if predictions_df is None or len(predictions_df) == 0:
        st.error("❌ No predictions data to display")
        return False
        
    st.subheader("🎯 Prediction Results")
    
    # Display basic info
    st.write(f"**Sample Size:** {len(predictions_df):,} rows")
    st.write(f"**Columns:** {', '.join(predictions_df.columns.tolist())}")
    
    # Show metadata if available
    if st.session_state.prediction_metadata:
        metadata = st.session_state.prediction_metadata
        st.info(f"📊 Full dataset: {metadata.get('total_predictions', 0):,} total predictions")
        if 'prediction_stats' in metadata:
            stats = metadata['prediction_stats']
            st.write(f"**Prediction 0:** {stats.get('prediction_0', 0):,}")
            st.write(f"**Prediction 1:** {stats.get('prediction_1', 0):,}")
        if 'target_column' in metadata:
            st.write(f"**Target Column:** {metadata['target_column']}")
    
    # Display sample predictions
    st.write("**Sample Predictions (first 10 rows):**")
    st.dataframe(predictions_df.head(10))
    
    # Show prediction distribution
    if 'prediction' in predictions_df.columns:
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Prediction Distribution:**")
            pred_counts = predictions_df['prediction'].value_counts()
            st.bar_chart(pred_counts)
            
        with col2:
            st.write("**Prediction Statistics:**")
            st.write(f"Sample size: {len(predictions_df):,}")
            st.write(f"Unique predictions: {predictions_df['prediction'].nunique()}")
            
            # Show accuracy if we have actual values
            if 'Diabetes_binary' in predictions_df.columns:
                accuracy = (predictions_df['prediction'] == predictions_df['Diabetes_binary']).mean()
                st.write(f"**Accuracy vs actual:** {accuracy:.4f}")
            elif 'prediction_probability' in predictions_df.columns:
                avg_prob = predictions_df['prediction_probability'].mean()
                st.write(f"**Average confidence:** {avg_prob:.4f}")
    
    # Download button
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Sample Predictions CSV",
        data=csv,
        file_name=f"predictions_sample_{session_id}.csv",
        mime="text/csv",
        key=f"download_{session_id}"
    )
    
    return True

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
                        
                        # Store available columns for later use
                        st.session_state.available_columns = df_preview.columns.tolist()
                        
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
                        
                        # Show column suggestions for diabetes dataset
                        if 'diabetes_binary' in [col.lower() for col in st.session_state.available_columns]:
                            st.info("💡 **Diabetes Dataset Detected**: Suggested target column: 'Diabetes_binary'")
                        elif 'target' in [col.lower() for col in st.session_state.available_columns]:
                            st.info("💡 **Target Column Detected**: Suggested target column: 'target'")
                        elif 'label' in [col.lower() for col in st.session_state.available_columns]:
                            st.info("💡 **Label Column Detected**: Suggested target column: 'label'")
                            
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
            st.json(st.session_state.analysis_results)
        
        # Trigger Ingest Job for EDA
        if st.button("Run Data Analysis"):
            with st.spinner("Running data analysis..."):
                result = run_job(INGEST_JOB_ID, {
                    "dbfs_path": st.session_state.dbfs_path,
                    "session_id": st.session_state.session_id
                })
                
                if result["status"] == "success":
                    st.success(f"✅ Data analysis completed! Run ID: {result.get('run_id', 'N/A')}")
                    
                    # Store basic results
                    st.session_state.analysis_results = {
                        "run_id": result.get('run_id'),
                        "status": "success",
                        "message": "Analysis completed successfully."
                    }
                    
                    st.subheader("📈 Analysis Results")
                    st.success("✅ Data analysis completed successfully!")
                    st.info("""
                    **Next Steps:**
                    1. Check your Databricks workspace for detailed analysis results
                    2. The data has been processed and stored in Delta format
                    3. You can now proceed to Model Training
                    """)
                    
                    st.balloons()
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
            st.json(st.session_state.training_results)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Training Configuration")
            model_type = st.selectbox(
                "Model Type",
                ["AutoML (Recommended)", "Random Forest", "Logistic Regression", "Gradient Boosting"]
            )
            
            # Initialize suggested_target
            suggested_target = ""
            
            # Show available columns and suggest target
            if st.session_state.available_columns:
                st.info(f"📋 Available columns: {', '.join(st.session_state.available_columns)}")
                
                # Auto-suggest target column
                if 'Diabetes_binary' in st.session_state.available_columns:
                    suggested_target = 'Diabetes_binary'
                elif 'diabetes_binary' in [col.lower() for col in st.session_state.available_columns]:
                    for col in st.session_state.available_columns:
                        if col.lower() == 'diabetes_binary':
                            suggested_target = col
                            break
                elif 'target' in [col.lower() for col in st.session_state.available_columns]:
                    for col in st.session_state.available_columns:
                        if col.lower() == 'target':
                            suggested_target = col
                            break
                elif 'label' in [col.lower() for col in st.session_state.available_columns]:
                    for col in st.session_state.available_columns:
                        if col.lower() == 'label':
                            suggested_target = col
                            break
                else:
                    suggested_target = st.session_state.available_columns[-1]
                
                if suggested_target:
                    st.success(f"💡 Suggested target column: **{suggested_target}**")
            
            target_column = st.text_input(
                "Target Column (leave empty for auto-detection)", 
                value=suggested_target if suggested_target else "",
                help="The column you want to predict. For diabetes dataset, use 'Diabetes_binary'"
            )
        
        with col2:
            st.subheader("Advanced Options")
            test_size = st.slider("Test Size Ratio", 0.1, 0.5, 0.2)
            random_state = st.number_input("Random State", value=42)
        
        if st.button("🚀 Train Model"):
            if target_column and target_column not in st.session_state.available_columns:
                st.error("Please select a valid target column from the available columns.")
            else:
                with st.spinner("Training model... This may take several minutes."):
                    result = run_job(TRAIN_JOB_ID, {
                        "session_id": st.session_state.session_id,
                        "target_column": target_column if target_column else "",
                        "test_size": str(test_size),
                        "random_state": str(random_state)
                    })
                    
                    if result["status"] == "success":
                        st.success(f"✅ Model training completed! Run ID: {result.get('run_id', 'N/A')}")
                        
                        # Store basic results
                        st.session_state.training_results = {
                            "run_id": result.get('run_id'),
                            "status": "success",
                            "message": "Model trained and registered in MLflow",
                            "model_name": f"smart_predictor_model_{st.session_state.session_id}",
                            "target_column": target_column
                        }
                        
                        st.subheader("🎯 Training Results")
                        st.success("Model trained and registered in MLflow!")
                        st.info("""
                        **Next Steps:**
                        1. Model has been trained and registered in MLflow
                        2. You can now proceed to Batch Scoring to make predictions
                        3. Check MLflow in your Databricks workspace for detailed metrics
                        """)
                        
                        st.balloons()
                    else:
                        st.error(f"❌ Model training failed: {result['message']}")
    else:
        st.warning("⚠️ Please upload a CSV file first from the Home page.")

# Batch Scoring Page
elif page == "Batch Scoring":
    st.header("📈 Batch Scoring")
    
    if st.session_state.upload_complete:
        st.success("✅ File uploaded successfully!")
        st.info("Generate predictions using your trained model")
        
        # Add refresh button
        col1, col2 = st.columns([3, 1])
        with col2:
            if st.button("🔄 Refresh Predictions", key="refresh_predictions"):
                st.rerun()
        
        # Enhanced debug button
        if st.button("🔍 Enhanced File Debug"):
            predictions_df = enhanced_debug_prediction_files(st.session_state.session_id)
            if predictions_df is not None:
                st.session_state.predictions_data = predictions_df
                display_success = display_predictions(predictions_df, st.session_state.session_id)
                if display_success:
                    st.balloons()
        
        # Display previous scoring results if available
        if st.session_state.scoring_results:
            st.subheader("📊 Previous Scoring Results")
            st.json(st.session_state.scoring_results)
            
            # Try to load and display previous predictions
            st.info("🔄 Loading predictions...")
            
            # NEW: Try task output first, then files
            run_id = st.session_state.scoring_results.get("run_id")
            predictions_df = None
            
            if run_id:
                predictions_df = get_predictions_from_job_output(run_id)
            
            # If task output failed, try files
            if predictions_df is None:
                predictions_df = load_predictions_smart(st.session_state.session_id)
            
            # Display predictions if found
            if predictions_df is not None:
                st.session_state.predictions_data = predictions_df
                display_success = display_predictions(predictions_df, st.session_state.session_id)
                if display_success:
                    st.balloons()
            else:
                st.warning("📁 Predictions not found in automatic search.")
        
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
                    scoring_session_id = st.session_state.session_id + "_scoring"
                    scoring_dbfs_path = safe_dbfs_path(scoring_session_id, scoring_file.name)
                    
                    upload_result = upload_to_dbfs_simple(scoring_file, scoring_dbfs_path)
                    
                    if upload_result["status"] == "success":
                        input_dbfs_path = scoring_dbfs_path
                        st.success(f"✅ Scoring file uploaded: {scoring_dbfs_path}")
                    else:
                        st.error(f"❌ Scoring file upload failed: {upload_result['message']}")
                        st.info("Using original training data for scoring...")
                
                # Verify training was completed first
                if not st.session_state.training_results:
                    st.error("❌ Please train a model first before running batch scoring!")
                    st.info("Go to the Model Training page and train a model first.")
                else:
                    result = run_job(SCORE_JOB_ID, {
                        "input_dbfs_path": input_dbfs_path,
                        "session_id": st.session_state.session_id
                    })
                    
                    if result["status"] == "success":
                        run_id = result.get('run_id')
                        st.success(f"✅ Batch scoring completed! Run ID: {run_id}")
                        
                        # Store basic results
                        st.session_state.scoring_results = {
                            "run_id": run_id,
                            "status": "success",
                            "message": "Predictions generated successfully"
                        }
                        
                        st.subheader("🎯 Scoring Results")
                        st.success("Predictions generated successfully!")
                        
                        # NEW: Try to get predictions from TASK output immediately
                        if run_id:
                            with st.spinner("Loading predictions from task output..."):
                                predictions_df = get_predictions_from_job_output(run_id)
                                
                                if predictions_df is not None:
                                    st.session_state.predictions_data = predictions_df
                                    display_success = display_predictions(predictions_df, st.session_state.session_id)
                                    
                                    if display_success:
                                        st.balloons()
                                else:
                                    st.warning("Predictions not in task output. Trying file search...")
                                    # Fall back to file search
                                    predictions_df = load_predictions_smart(st.session_state.session_id)
                                    if predictions_df is not None:
                                        st.session_state.predictions_data = predictions_df
                                        display_predictions(predictions_df, st.session_state.session_id)
                                    else:
                                        st.info("💡 Predictions were generated successfully! Check back in a moment.")
                        else:
                            st.info("💡 Predictions generated! Check back in a moment.")
                        
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

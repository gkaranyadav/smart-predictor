# app.py - COMPLETE UPDATED VERSION
import streamlit as st
import pandas as pd
import time
import json
import io
from databricks_api import dbfs_put_single, dbfs_upload_chunked, upload_to_dbfs_simple, run_job, get_job_output, dbfs_read_file, dbfs_file_exists, dbfs_list_files
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

# NEW: Debug function for prediction files
def debug_prediction_files(session_id):
    """Debug function to see what files actually exist"""
    st.subheader("🐛 Debug File Search")
    
    base_path = f"/FileStore/results/{session_id}"
    st.info(f"Looking in: {base_path}")
    
    # List main directory
    list_result = dbfs_list_files(base_path)
    if list_result["status"] == "success":
        files = list_result.get("files", [])
        st.write(f"Found {len(files)} files/directories:")
        for file_info in files:
            st.write(f"- {file_info['path']} (is_dir: {file_info['is_dir']})")
            
            # If it's a directory, list its contents too
            if file_info['is_dir']:
                sub_result = dbfs_list_files(file_info['path'])
                if sub_result["status"] == "success":
                    sub_files = sub_result.get("files", [])
                    for sub_file in sub_files:
                        st.write(f"  ↳ {sub_file['path']} (size: {sub_file['file_size']})")
    else:
        st.error(f"Could not list directory: {list_result['message']}")

# NEW: Smart prediction loading functions
def load_predictions_from_directory(directory_path):
    """Load predictions from Spark output directory"""
    try:
        # List all files in the directory
        files_result = dbfs_list_files(directory_path)
        if files_result["status"] != "success":
            return None
            
        files = files_result.get("files", [])
        if not files:
            return None
        
        # Try different file patterns
        file_patterns = [
            f"{directory_path}/predictions_direct.csv",  # Direct CSV
            f"{directory_path}/predictions_sample.csv",  # Sample CSV
        ]
        
        # Also look for Spark part files
        for file_info in files:
            if "part-" in file_info["path"] and file_info["path"].endswith(".csv"):
                file_patterns.append(file_info["path"])
        
        for file_path in file_patterns:
            result = dbfs_read_file(file_path)
            if result["status"] == "success":
                try:
                    df = pd.read_csv(io.StringIO(result["content"]))
                    st.success(f"✅ Loaded predictions from: {file_path}")
                    return df
                except Exception as e:
                    continue
        
        return None
        
    except Exception as e:
        st.error(f"Error loading predictions: {str(e)}")
        return None

def load_predictions_smart(session_id):
    """Smart prediction loader with multiple fallbacks"""
    base_path = f"/FileStore/results/{session_id}"
    
    # Try multiple file locations in order of preference
    file_paths = [
        f"{base_path}/predictions_direct.csv",      # Primary - direct CSV
        f"{base_path}/predictions_sample.csv",      # Secondary - sample
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
    
    # If direct files not found, try directory approach
    return load_predictions_from_directory(base_path)

def display_predictions(predictions_df, session_id):
    """Display predictions with download option"""
    if predictions_df is None or len(predictions_df) == 0:
        return False
        
    st.subheader("🎯 Prediction Results")
    
    # Display sample predictions
    st.write("**Sample Predictions:**")
    st.dataframe(predictions_df.head(10))
    
    # Show prediction distribution
    if 'prediction' in predictions_df.columns:
        st.write("**Prediction Distribution:**")
        pred_counts = predictions_df['prediction'].value_counts()
        st.bar_chart(pred_counts)
        
        # Show prediction statistics
        st.write("**Prediction Statistics:**")
        st.write(f"Total predictions: {len(predictions_df):,}")
        st.write(f"Unique predictions: {predictions_df['prediction'].nunique()}")
        
        # Show accuracy if we have actual values
        if 'Diabetes_binary' in predictions_df.columns:
            accuracy = (predictions_df['prediction'] == predictions_df['Diabetes_binary']).mean()
            st.write(f"**Accuracy vs actual:** {accuracy:.4f}")
    
    # Download button
    csv = predictions_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Full Predictions CSV",
        data=csv,
        file_name=f"predictions_{session_id}.csv",
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
                            # For now, just show basic success message since we can't get multi-task outputs
                            st.session_state.analysis_results = {
                                "run_id": run_id,
                                "status": "success",
                                "message": "Analysis completed successfully. Check Databricks workspace for detailed results."
                            }
                            
                            st.subheader("📈 Analysis Results")
                            st.success("✅ Data analysis completed successfully!")
                            st.info("""
                            **Next Steps:**
                            1. Check your Databricks workspace for detailed analysis results
                            2. The data has been processed and stored in Delta format
                            3. You can now proceed to Model Training
                            """)
                            
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
                    # Find the exact case
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
                    # Suggest the last column (common for target)
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
            
            # Show target column tips
            st.info("""
            **Target Column Tips:**
            - For classification: Choose a column with limited unique values
            - For regression: Choose a numeric column
            - Common target names: 'target', 'label', 'class', 'outcome'
            """)
        
        # Validation
        if target_column and target_column not in st.session_state.available_columns:
            st.error(f"❌ Target column '{target_column}' not found in available columns!")
            st.info(f"Available columns: {', '.join(st.session_state.available_columns)}")
        
        if st.button("🚀 Train Model"):
            if target_column and target_column not in st.session_state.available_columns:
                st.error("Please select a valid target column from the available columns.")
            else:
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
                        
                        # Display model info
                        model_name = f"smart_predictor_model_{st.session_state.session_id}"
                        st.write(f"**Model Name:** {model_name}")
                        st.write(f"**Model URI:** models:/{model_name}/latest")
                        st.write(f"**Target Column:** {target_column if target_column else 'Auto-detected'}")
                        st.write(f"**Run ID:** {result.get('run_id', 'N/A')}")
                        
                        st.info("""
                        **Next Steps:**
                        1. Model has been trained and registered in MLflow
                        2. You can now proceed to Batch Scoring to make predictions
                        3. Check MLflow in your Databricks workspace for detailed metrics
                        """)
                        
                        st.balloons()
                    else:
                        st.error(f"❌ Model training failed: {result['message']}")
                        st.info("""
                        **Troubleshooting Tips:**
                        1. Make sure the target column exists in your data
                        2. Check that the target column has appropriate values for ML
                        3. Verify your Databricks job is configured correctly
                        """)
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
        
        # NEW: Add debug button
        if st.button("🐛 Debug File Search"):
            debug_prediction_files(st.session_state.session_id)
        
        # Display previous scoring results if available
        if st.session_state.scoring_results:
            st.subheader("📊 Previous Scoring Results")
            st.json(st.session_state.scoring_results)
            
            # Try to load and display previous predictions
            st.info("🔄 Loading predictions...")
            
            # NEW: Use smart prediction loader
            predictions_df = load_predictions_smart(st.session_state.session_id)
            
            # Display predictions if found
            if predictions_df is not None:
                st.session_state.predictions_data = predictions_df
                display_success = display_predictions(predictions_df, st.session_state.session_id)
                if display_success:
                    st.balloons()
            else:
                st.warning("📁 Predictions not found in automatic search.")
                st.info("""
                **Manual Access Instructions:**
                1. Go to **Databricks Workspace** → **Data** → **DBFS**
                2. Navigate to: `FileStore/results/{session_id}/`
                3. Look for CSV files (may be in subdirectories)
                4. Download and check the files manually
                """)
        
        # Option to upload new data for scoring or use existing
        scoring_file = st.file_uploader(
            "Upload new data for scoring (optional - uses training data if not provided)", 
            type=["csv"]
        )
        
        if st.button("🎯 Run Batch Scoring"):
            # Skip file existence check and proceed directly
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
                    # USING JOB PARAMETERS (not notebook_params)
                    result = run_job(SCORE_JOB_ID, {
                        "input_dbfs_path": input_dbfs_path,
                        "session_id": st.session_state.session_id
                    })
                    
                    if result["status"] == "success":
                        st.success(f"✅ Batch scoring completed! Run ID: {result.get('run_id', 'N/A')}")
                        
                        # Store basic results
                        st.session_state.scoring_results = {
                            "run_id": result.get('run_id'),
                            "status": "success",
                            "message": "Predictions generated successfully",
                            "predictions_path": f"/FileStore/results/{st.session_state.session_id}/"
                        }
                        
                        st.subheader("🎯 Scoring Results")
                        st.success("Predictions generated successfully!")
                        
                        # Display predictions info
                        predictions_base_path = f"/FileStore/results/{st.session_state.session_id}"
                        st.write(f"**Predictions saved to:** {predictions_base_path}/")
                        st.write(f"**Run ID:** {result.get('run_id', 'N/A')}")
                        
                        # Try to load predictions immediately
                        st.info("🔍 Loading predictions...")
                        # NEW: Use smart prediction loader
                        predictions_df = load_predictions_smart(st.session_state.session_id)
                        
                        if predictions_df is not None:
                            st.session_state.predictions_data = predictions_df
                            display_success = display_predictions(predictions_df, st.session_state.session_id)
                            if display_success:
                                st.balloons()
                        else:
                            st.warning("⏳ Predictions are being processed...")
                            st.info("""
                            **Next Steps:**
                            1. Wait a few moments and click the **🔄 Refresh Predictions** button above
                            2. Predictions are being saved by Spark (this takes a moment)
                            3. Your predictions will appear here automatically when ready
                            """)
                        
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

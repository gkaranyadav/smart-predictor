# app.py
import streamlit as st
import pandas as pd
import time
from databricks_api import dbfs_put_single, dbfs_upload_chunked, run_job
from utils import gen_session_id, safe_dbfs_path

# Page configuration
st.set_page_config(
    page_title="Smart Predictor",
    page_icon="🚀",
    layout="wide"
)

# Initialize session state
if 'session_id' not in st.session_state:
    st.session_state.session_id = None
if 'current_file' not in st.session_state:
    st.session_state.current_file = None
if 'upload_complete' not in st.session_state:
    st.session_state.upload_complete = False

# Job IDs from Streamlit Cloud secrets
INGEST_JOB_ID = st.secrets["DATABRICKS_JOB_INGEST_ID"]
TRAIN_JOB_ID = st.secrets["DATABRICKS_JOB_TRAIN_ID"]
SCORE_JOB_ID = st.secrets["DATABRICKS_JOB_SCORE_ID"]

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
                
                # Decide chunked or single upload based on file size
                if uploaded_file.size > 2 * 1024 * 1024:  # 2MB threshold
                    result = dbfs_upload_chunked(dbfs_path, uploaded_file, overwrite=True)
                else:
                    result = dbfs_put_single(dbfs_path, uploaded_file, overwrite=True)
                
                if result["status"] == "success":
                    st.session_state.upload_complete = True
                    st.session_state.dbfs_path = dbfs_path
                    st.success(f"✅ {result['message']}")
                else:
                    st.error(f"❌ {result['message']}")
        
        # Show preview of uploaded data
        if st.session_state.upload_complete:
            st.subheader("Data Preview")
            try:
                uploaded_file.seek(0)  # Reset file pointer
                df_preview = pd.read_csv(uploaded_file, nrows=5)
                st.dataframe(df_preview)
                st.write(f"Shape: {len(df_preview)} rows x {len(df_preview.columns)} columns")
            except Exception as e:
                st.error(f"Error previewing data: {str(e)}")

# Data Analysis Page
elif page == "Data Analysis":
    st.header("🔍 Data Analysis")
    
    if st.session_state.upload_complete:
        st.success("✅ File uploaded successfully!")
        
        # Trigger Ingest Job for EDA
        if st.button("Run Data Analysis"):
            with st.spinner("Running data analysis..."):
                result = run_job(INGEST_JOB_ID, {
                    "dbfs_path": st.session_state.dbfs_path,
                    "session_id": st.session_state.session_id
                })
                
                if result["status"] == "success":
                    st.success(f"✅ Data analysis completed! Run ID: {result.get('run_id', 'N/A')}")
                    # Here you would typically fetch and display the analysis results
                    st.info("Analysis results would be displayed here once the job completes.")
                else:
                    st.error(f"❌ Data analysis failed: {result['message']}")
    else:
        st.warning("⚠️ Please upload a CSV file first from the Home page.")

# Model Training Page
elif page == "Model Training":
    st.header("🤖 Model Training")
    
    if st.session_state.upload_complete:
        st.info("Train a machine learning model on your uploaded data")
        
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
                result = run_job(TRAIN_JOB_ID, {
                    "session_id": st.session_state.session_id,
                    "target_column": target_column,
                    "test_size": str(test_size),
                    "random_state": str(random_state)
                })
                
                if result["status"] == "success":
                    st.success(f"✅ Model training completed! Run ID: {result.get('run_id', 'N/A')}")
                    st.balloons()
                else:
                    st.error(f"❌ Model training failed: {result['message']}")
    else:
        st.warning("⚠️ Please upload a CSV file first from the Home page.")

# Batch Scoring Page
elif page == "Batch Scoring":
    st.header("📈 Batch Scoring")
    
    if st.session_state.upload_complete:
        st.info("Generate predictions using your trained model")
        
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
                    
                    if scoring_file.size > 2 * 1024 * 1024:
                        upload_result = dbfs_upload_chunked(scoring_dbfs_path, scoring_file, overwrite=True)
                    else:
                        upload_result = dbfs_put_single(scoring_dbfs_path, scoring_file, overwrite=True)
                    
                    if upload_result["status"] == "success":
                        input_dbfs_path = scoring_dbfs_path
                    else:
                        st.error(f"❌ Scoring file upload failed: {upload_result['message']}")
                        # Continue with original file instead of returning
                
                # Run scoring job
                result = run_job(SCORE_JOB_ID, {
                    "input_dbfs_path": input_dbfs_path,
                    "session_id": st.session_state.session_id
                })
                
                if result["status"] == "success":
                    st.success(f"✅ Batch scoring completed! Run ID: {result.get('run_id', 'N/A')}")
                    st.info("Predictions would be available for download here once the job completes.")
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

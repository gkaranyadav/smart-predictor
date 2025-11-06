import streamlit as st
import pandas as pd
import time
from databricks_api import (
    dbfs_upload_chunked, 
    trigger_databricks_job, 
    get_job_status,
    get_job_result,
    validate_databricks_config
)

st.set_page_config(
    page_title="Smart Predictor",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Smart Predictor - AI Assistant")
st.markdown("Upload your dataset and train ML models with Databricks backend")

# File upload section
uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"])

if uploaded_file is not None:
    try:
        # First, load locally for preview
        df = pd.read_csv(uploaded_file)
        st.session_state.current_dataset = df
        st.session_state.uploaded_file_name = uploaded_file.name
        
        st.success(f"✅ File loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Upload to Databricks when user clicks button
        if st.button("🚀 Upload to Databricks & Train Models", type="primary"):
            if not validate_databricks_config():
                st.stop()
                
            # Upload file to DBFS
            dbfs_path = f"/FileStore/uploaded_data/{uploaded_file.name}"
            uploaded_file.seek(0)  # Reset file pointer
            
            with st.spinner("Uploading to Databricks..."):
                if dbfs_upload_chunked(dbfs_path, uploaded_file):
                    # Trigger training job
                    parameters = {
                        "file_path": dbfs_path,
                        "file_name": uploaded_file.name
                    }
                    
                    run_id = trigger_databricks_job(
                        st.secrets["DATABRICKS_JOB_TRAIN_ID"], 
                        parameters
                    )
                    
                    if run_id:
                        st.session_state.current_run_id = run_id
                        st.success("🎯 Training job started! Check progress below...")
                        
                        # Monitor job progress
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        while True:
                            result = get_job_result(run_id)
                            if result and result.get("status") != "running":
                                break
                                
                            status_text.text("🔄 Training in progress...")
                            progress_bar.progress(50)
                            time.sleep(5)
                        
                        # Show results
                        if result and "notebook_output" in result:
                            st.success("✅ Training completed!")
                            st.json(result["notebook_output"]["result"])
                        else:
                            st.error("❌ Training failed or timed out")
                
    except Exception as e:
        st.error(f"Error: {str(e)}")

# Show current status
if 'current_dataset' in st.session_state:
    st.info(f"📊 Dataset ready: {st.session_state.current_dataset.shape}")

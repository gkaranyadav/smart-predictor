import streamlit as st
import pandas as pd
import requests
import json
import time
import base64

# Page configuration
st.set_page_config(
    page_title="Databricks ML Pipeline", 
    layout="wide",
    page_icon="🚀"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'job_status' not in st.session_state:
        st.session_state.job_status = 'not_started'
    if 'job_id' not in st.session_state:
        st.session_state.job_id = None
    if 'results' not in st.session_state:
        st.session_state.results = None

def get_databricks_config():
    """Get Databricks configuration from secrets"""
    try:
        return {
            'host': st.secrets["DATABRICKS"]["HOST"].rstrip('/'),
            'token': st.secrets["DATABRICKS"]["TOKEN"]
        }
    except Exception as e:
        st.error(f"❌ Error loading Databricks configuration: {e}")
        return None

def upload_file_to_dbfs(file_content, file_name, config):
    """Upload file to DBFS using Databricks API"""
    try:
        # Encode file content
        encoded_content = base64.b64encode(file_content).decode()
        
        # DBFS API endpoint
        url = f"{config['host']}/api/2.0/dbfs/put"
        
        headers = {
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "path": f"/FileStore/uploads/{file_name}",
            "contents": encoded_content,
            "overwrite": True
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return f"dbfs:/FileStore/uploads/{file_name}"
        else:
            st.error(f"File upload failed: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error uploading file: {e}")
        return None

def trigger_databricks_job(config, file_path, model_type, enable_tuning, test_size):
    """Trigger Databricks job via API"""
    try:
        url = f"{config['host']}/api/2.0/jobs/run-now"
        
        headers = {
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json"
        }
        
        # Job parameters
        data = {
            "job_id": 123456,  # We'll set this after creating the job
            "notebook_params": {
                "input_path": file_path,
                "output_path": "/FileStore/results",
                "model_type": model_type,
                "enable_tuning": str(enable_tuning).lower(),
                "test_size": str(test_size)
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            return response.json()["run_id"]
        else:
            st.error(f"Job trigger failed: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error triggering job: {e}")
        return None

def get_job_status(config, run_id):
    """Get job status"""
    try:
        url = f"{config['host']}/api/2.0/jobs/runs/get?run_id={run_id}"
        
        headers = {
            "Authorization": f"Bearer {config['token']}"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()["state"]["life_cycle_state"]
        else:
            return "UNKNOWN"
            
    except Exception as e:
        return "ERROR"

def run_pipeline(uploaded_file, model_name, enable_tuning, test_size):
    """Trigger the Databricks pipeline"""
    try:
        config = get_databricks_config()
        if not config:
            return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Upload file
        status_text.info("📤 Uploading file to Databricks...")
        file_content = uploaded_file.getvalue()
        file_path = upload_file_to_dbfs(file_content, uploaded_file.name, config)
        progress_bar.progress(25)
        
        if not file_path:
            return
        
        # Step 2: Trigger job
        status_text.info("🚀 Starting ML pipeline on Databricks...")
        
        # Map model names to internal codes
        model_mapping = {
            "Logistic Regression": "logistic",
            "Random Forest": "random_forest", 
            "XGBoost": "xgboost",
            "LightGBM": "lightgbm",
            "Neural Network": "neural_net"
        }
        
        model_code = model_mapping[model_name]
        run_id = trigger_databricks_job(config, file_path, model_code, enable_tuning, test_size)
        progress_bar.progress(50)
        
        if not run_id:
            return
        
        # Store in session state
        st.session_state.job_id = run_id
        st.session_state.job_status = 'running'
        
        # Step 3: Poll for completion
        status_text.info("🔄 Pipeline running... This may take a few minutes.")
        
        max_attempts = 60  # 5 minutes max
        for attempt in range(max_attempts):
            status = get_job_status(config, run_id)
            
            if status in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
                break
                
            progress = 50 + (attempt / max_attempts) * 45
            progress_bar.progress(min(progress, 95))
            time.sleep(5)  # Wait 5 seconds between checks
        
        progress_bar.progress(100)
        
        if status == "TERMINATED":
            status_text.success("✅ Pipeline completed successfully!")
            st.session_state.job_status = 'completed'
        else:
            status_text.error(f"❌ Pipeline ended with status: {status}")
            st.session_state.job_status = 'failed'
        
    except Exception as e:
        st.error(f"❌ Error in pipeline: {e}")
        st.session_state.job_status = 'failed'

def main():
    initialize_session_state()
    
    st.title("🚀 Databricks ML Pipeline")
    st.markdown("Upload your dataset and train ML models on Databricks!")
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = {
            "Logistic Regression": "logistic",
            "Random Forest": "random_forest", 
            "XGBoost": "xgboost",
            "LightGBM": "lightgbm",
            "Neural Network": "neural_net"
        }
        
        selected_model = st.selectbox(
            "Select Model",
            options=list(model_options.keys()),
            index=1
        )
        
        # Hyperparameter tuning option
        enable_tuning = st.checkbox("Enable Hyperparameter Tuning", value=False)
        
        # Test size
        test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        
        st.markdown("---")
        st.header("📊 Dataset Info")
        st.info("Upload a CSV file for analysis and modeling")
    
    # Main area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your dataset in CSV format"
        )
        
        if uploaded_file is not None:
            # Preview data
            try:
                df_preview = pd.read_csv(uploaded_file, nrows=5)
                st.subheader("Data Preview")
                st.dataframe(df_preview)
                
                st.subheader("Dataset Info")
                st.write(f"📏 **Shape:** {df_preview.shape}")
                st.write(f"🎯 **Columns:** {', '.join(df_preview.columns.tolist())}")
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with col2:
        st.header("🚀 Pipeline Controls")
        
        if uploaded_file is not None:
            # Run pipeline button
            if st.button("🎯 Run ML Pipeline", type="primary", use_container_width=True):
                run_pipeline(uploaded_file, selected_model, enable_tuning, test_size)
            
            # Show status
            if st.session_state.job_status == 'running':
                st.info("🔄 Pipeline is running on Databricks...")
            elif st.session_state.job_status == 'completed':
                st.success("✅ Pipeline completed! Check Databricks for results.")
            elif st.session_state.job_status == 'failed':
                st.error("❌ Pipeline failed. Check Databricks logs for details.")
        
        else:
            st.info("Please upload a CSV file to start the pipeline")

if __name__ == "__main__":
    main()

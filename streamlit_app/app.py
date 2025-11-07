import streamlit as st
import pandas as pd
import requests
import json
import time
import base64
import io

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
        config = {
            'host': st.secrets["DATABRICKS"]["HOST"].rstrip('/'),
            'token': st.secrets["DATABRICKS"]["TOKEN"],
            'job_id': st.secrets["DATABRICKS"]["JOB_ID"]
        }
        return config
    except Exception as e:
        st.error(f"❌ Error loading Databricks configuration: {e}")
        return None

def upload_file_chunk_to_dbfs(chunk_content, chunk_path, config):
    """Upload a single file chunk to DBFS"""
    try:
        # Encode chunk content
        encoded_content = base64.b64encode(chunk_content).decode()
        
        # DBFS API endpoint
        url = f"{config['host']}/api/2.0/dbfs/put"
        
        headers = {
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "path": chunk_path,
            "contents": encoded_content,
            "overwrite": True
        }
        
        response = requests.post(url, headers=headers, json=data)
        return response.status_code == 200
        
    except Exception as e:
        st.error(f"Error uploading chunk: {e}")
        return False

def split_and_upload_large_file(file_content, file_name, config):
    """Split large file into chunks and upload to DBFS"""
    try:
        # Calculate chunk size (2.5MB to be safe under 10MB limit)
        CHUNK_SIZE = 2 * 1024 * 1024  # 2MB in bytes
        
        # Split file content into chunks
        chunks = []
        total_chunks = (len(file_content) + CHUNK_SIZE - 1) // CHUNK_SIZE
        
        chunk_paths = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i in range(total_chunks):
            start_idx = i * CHUNK_SIZE
            end_idx = min((i + 1) * CHUNK_SIZE, len(file_content))
            chunk_content = file_content[start_idx:end_idx]
            
            # Upload this chunk
            chunk_name = f"{file_name}_chunk_{i:03d}.csv"
            chunk_path = f"/FileStore/uploads/{chunk_name}"
            
            status_text.info(f"📤 Uploading chunk {i+1}/{total_chunks}...")
            
            if upload_file_chunk_to_dbfs(chunk_content, chunk_path, config):
                chunk_paths.append(f"dbfs:{chunk_path}")
                progress_bar.progress((i + 1) / total_chunks)
            else:
                st.error(f"❌ Failed to upload chunk {i+1}")
                return None
        
        status_text.success(f"✅ All {total_chunks} chunks uploaded successfully!")
        return chunk_paths
        
    except Exception as e:
        st.error(f"❌ Error splitting and uploading file: {e}")
        return None

def trigger_databricks_job(config, chunk_paths, model_type, enable_tuning, test_size):
    """Trigger Databricks job via API"""
    try:
        url = f"{config['host']}/api/2.0/jobs/run-now"
        
        headers = {
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json"
        }
        
        # Job parameters
        data = {
            "job_id": config['job_id'],
            "notebook_params": {
                "chunk_paths": json.dumps(chunk_paths),  # Pass as JSON string
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
            st.error(f"❌ Job trigger failed: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error triggering job: {e}")
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
            st.error("❌ Cannot start pipeline - Databricks configuration missing")
            return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Split and upload file in chunks
        status_text.info("📤 Splitting and uploading file to Databricks...")
        file_content = uploaded_file.getvalue()
        
        chunk_paths = split_and_upload_large_file(file_content, uploaded_file.name, config)
        progress_bar.progress(50)
        
        if not chunk_paths:
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
        run_id = trigger_databricks_job(config, chunk_paths, model_code, enable_tuning, test_size)
        progress_bar.progress(75)
        
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
                
            progress = 75 + (attempt / max_attempts) * 20
            progress_bar.progress(min(progress, 95))
            time.sleep(5)  # Wait 5 seconds between checks
        
        progress_bar.progress(100)
        
        if status == "TERMINATED":
            status_text.success("✅ Pipeline completed successfully!")
            st.session_state.job_status = 'completed'
            
            # Show results section
            show_results_section(config, run_id)
        else:
            status_text.error(f"❌ Pipeline ended with status: {status}")
            st.session_state.job_status = 'failed'
        
    except Exception as e:
        st.error(f"❌ Error in pipeline: {e}")
        st.session_state.job_status = 'failed'

def show_results_section(config, run_id):
    """Display results from the completed job"""
    try:
        st.markdown("---")
        st.header("📊 Results")
        
        # Get job output
        url = f"{config['host']}/api/2.0/jobs/runs/get-output?run_id={run_id}"
        headers = {"Authorization": f"Bearer {config['token']}"}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            output = response.json()
            
            if "notebook_output" in output:
                st.subheader("Job Output")
                st.text(output["notebook_output"])
            else:
                st.info("Check Databricks workspace for detailed results and MLflow experiments")
                
        else:
            st.warning("Could not fetch job output. Check Databricks workspace for results.")
            
    except Exception as e:
        st.error(f"Error fetching results: {e}")

def main():
    initialize_session_state()
    
    st.title("🚀 Databricks ML Pipeline")
    st.markdown("Upload your dataset and train ML models on Databricks!")
    
    # Check configuration first
    config = get_databricks_config()
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        if not config:
            st.error("Please configure Databricks secrets in Streamlit Cloud")
            return
        
        # Model selection
        model_options = [
            "Logistic Regression",
            "Random Forest", 
            "XGBoost",
            "LightGBM",
            "Neural Network"
        ]
        
        selected_model = st.selectbox(
            "Select Model",
            options=model_options,
            index=1
        )
        
        # Hyperparameter tuning option
        enable_tuning = st.checkbox("Enable Hyperparameter Tuning", value=False)
        
        # Test size - user selects from 10% to 40%
        test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        
        st.markdown("---")
        st.header("📊 Dataset Info")
        st.info("Upload a CSV file for analysis and modeling")
        
        # Display current configuration
        st.markdown("### Current Settings")
        st.write(f"**Model:** {selected_model}")
        st.write(f"**Test Size:** {test_size}%")
        st.write(f"**Hyperparameter Tuning:** {'Yes' if enable_tuning else 'No'}")
        st.write(f"**Job ID:** {config['job_id']}")
    
    # Main area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Data Upload")
        
        # File upload
        uploaded_file = st.file_uploader(
            "Choose a CSV file",
            type=['csv'],
            help="Upload your dataset in CSV format (any size supported)"
        )
        
        if uploaded_file is not None:
            # Preview data
            try:
                df_preview = pd.read_csv(uploaded_file, nrows=5)
                st.subheader("Data Preview")
                st.dataframe(df_preview)
                
                st.subheader("Dataset Info")
                st.write(f"📏 **Shape:** {df_preview.shape}")
                st.write(f"📊 **File Size:** {len(uploaded_file.getvalue()) / (1024*1024):.2f} MB")
                st.write(f"🎯 **Columns:** {len(df_preview.columns)}")
                st.write(f"🔍 **Sample Columns:** {', '.join(df_preview.columns.tolist()[:5])}...")
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with col2:
        st.header("🚀 Pipeline Controls")
        
        if uploaded_file is not None:
            # Display current configuration
            st.subheader("Pipeline Configuration")
            st.write(f"**Dataset:** {uploaded_file.name}")
            st.write(f"**Model:** {selected_model}")
            st.write(f"**Test Size:** {test_size}%")
            st.write(f"**Tuning:** {'Enabled' if enable_tuning else 'Disabled'}")
            
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

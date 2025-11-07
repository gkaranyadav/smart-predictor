import streamlit as st
import pandas as pd
import requests
import json
import time
import base64
import os
import tempfile

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
    if 'uploaded_file_path' not in st.session_state:
        st.session_state.uploaded_file_path = None
    if 'dbfs_file_path' not in st.session_state:
        st.session_state.dbfs_file_path = None

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

def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location and return path"""
    try:
        # Create temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.csv') as tmp_file:
            # Write uploaded file content to temp file
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"❌ Error saving file: {e}")
        return None

def upload_file_to_dbfs(config, local_file_path, filename):
    """Upload file directly to DBFS using Databricks API"""
    try:
        # Read file content
        with open(local_file_path, 'rb') as f:
            content = f.read()
        
        # DBFS path
        dbfs_path = f"/FileStore/uploads/{filename}"
        full_dbfs_path = f"dbfs:{dbfs_path}"
        
        st.info(f"📤 Uploading {filename} to DBFS...")
        
        # Use the simpler single-call API approach
        url = f"{config['host']}/api/2.0/dbfs/put"
        
        headers = {
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json"
        }
        
        # Convert content to base64
        content_b64 = base64.b64encode(content).decode('utf-8')
        
        data = {
            "path": dbfs_path,
            "contents": content_b64,
            "overwrite": True
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            st.success(f"✅ File uploaded successfully to: {full_dbfs_path}")
            return full_dbfs_path
        else:
            st.error(f"❌ Failed to upload file: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error uploading file to DBFS: {e}")
        return None

def trigger_databricks_job(config, dbfs_file_path, model_type, enable_tuning, test_size):
    """Trigger Databricks job via API"""
    try:
        url = f"{config['host']}/api/2.0/jobs/run-now"
        
        headers = {
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json"
        }
        
        # Job parameters
        data = {
            "job_id": int(config['job_id']),
            "notebook_params": {
                "input_path": dbfs_file_path,
                "output_path": "dbfs:/FileStore/results",
                "model_type": model_type,
                "enable_tuning": str(enable_tuning).lower(),
                "test_size": str(test_size)
            }
        }
        
        st.info("🚀 Triggering Databricks job...")
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            run_id = response.json()["run_id"]
            st.success(f"✅ Job triggered successfully! Run ID: {run_id}")
            return run_id
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
            result = response.json()
            state = result["state"]
            return state["life_cycle_state"], state.get("result_state", "UNKNOWN"), state.get("state_message", "")
        else:
            return "UNKNOWN", "FAILED", f"API Error: {response.text}"
            
    except Exception as e:
        return "ERROR", "FAILED", str(e)

def get_job_output(config, run_id):
    """Get job output"""
    try:
        url = f"{config['host']}/api/2.0/jobs/runs/get-output?run_id={run_id}"
        headers = {"Authorization": f"Bearer {config['token']}"}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception as e:
        st.error(f"Error fetching job output: {e}")
        return None

def run_pipeline(dbfs_file_path, model_name, enable_tuning, test_size):
    """Trigger the Databricks pipeline"""
    try:
        config = get_databricks_config()
        if not config:
            st.error("❌ Cannot start pipeline - Databricks configuration missing")
            return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Map model names to internal codes
        model_mapping = {
            "Logistic Regression": "logistic",
            "Random Forest": "random_forest", 
            "XGBoost": "xgboost",
            "LightGBM": "lightgbm",
            "Neural Network": "neural_net"
        }
        
        model_code = model_mapping[model_name]
        
        # Step 1: Trigger job
        status_text.info("🚀 Starting ML pipeline on Databricks...")
        run_id = trigger_databricks_job(config, dbfs_file_path, model_code, enable_tuning, test_size)
        progress_bar.progress(30)
        
        if not run_id:
            st.error("❌ Failed to trigger Databricks job.")
            return
        
        # Store in session state
        st.session_state.job_id = run_id
        st.session_state.job_status = 'running'
        
        # Step 2: Poll for completion
        status_text.info("🔄 Pipeline running... This may take a few minutes.")
        
        max_attempts = 120  # 10 minutes max (5 seconds per check)
        for attempt in range(max_attempts):
            life_cycle_state, result_state, message = get_job_status(config, run_id)
            
            # Update progress based on state
            if life_cycle_state == "PENDING":
                progress = 0.3 + (attempt / max_attempts) * 0.2
            elif life_cycle_state == "RUNNING":
                progress = 0.5 + (attempt / max_attempts) * 0.4
            else:
                progress = 0.9
            
            progress_bar.progress(min(progress, 0.9))
            
            # Display current status
            status_text.info(f"🔄 Current status: {life_cycle_state} - {message}")
            
            if life_cycle_state in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
                break
                
            time.sleep(5)  # Wait 5 seconds between checks
        
        progress_bar.progress(1.0)
        
        if life_cycle_state == "TERMINATED" and result_state == "SUCCESS":
            status_text.success("✅ Pipeline completed successfully!")
            st.session_state.job_status = 'completed'
            
            # Show results section
            show_results_section(config, run_id)
        else:
            status_text.error(f"❌ Pipeline ended with status: {life_cycle_state} - {result_state}")
            if message:
                st.error(f"Error message: {message}")
            st.session_state.job_status = 'failed'
        
    except Exception as e:
        st.error(f"❌ Error in pipeline: {e}")
        st.session_state.job_status = 'failed'

def show_results_section(config, run_id):
    """Display results from the completed job"""
    try:
        st.markdown("---")
        st.header("📊 Pipeline Results")
        
        # Get job output
        output = get_job_output(config, run_id)
        
        if output:
            if "notebook_output" in output and output["notebook_output"]:
                st.subheader("Job Logs")
                st.text_area("Execution Logs", output["notebook_output"]["result"] if isinstance(output["notebook_output"], dict) else output["notebook_output"], height=300)
            else:
                st.success("✅ Pipeline completed successfully!")
                
                # Show MLflow and DBFS links
                st.info("""
                **Check Databricks for detailed results:**
                - 📈 **MLflow Experiments**: Model metrics and parameters
                - 📊 **DBFS Results**: EDA reports and model artifacts  
                - 📋 **Job Runs**: Detailed execution logs
                """)
                
                # Try to get results from DBFS
                try:
                    results_url = f"{config['host']}/api/2.0/dbfs/read"
                    headers = {"Authorization": f"Bearer {config['token']}"}
                    results_data = {
                        "path": "/FileStore/results/results.json"
                    }
                    
                    response = requests.get(results_url, headers=headers, json=results_data)
                    if response.status_code == 200:
                        results_content = base64.b64decode(response.json()["data"]).decode('utf-8')
                        results = json.loads(results_content)
                        
                        if results.get("status") == "success":
                            st.subheader("📈 Model Performance")
                            metrics = results.get("metrics", {})
                            
                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric("Accuracy", f"{metrics.get('accuracy', 0):.4f}")
                            with col2:
                                st.metric("Precision", f"{metrics.get('precision', 0):.4f}")
                            with col3:
                                st.metric("Recall", f"{metrics.get('recall', 0):.4f}")
                            with col4:
                                st.metric("F1 Score", f"{metrics.get('f1_score', 0):.4f}")
                            
                            if "roc_auc" in metrics:
                                st.metric("ROC AUC", f"{metrics.get('roc_auc', 0):.4f}")
                except Exception as e:
                    st.warning("Could not load detailed results from DBFS")
                    
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
            st.info("""
            **Required secrets in Streamlit Cloud:**
            ```
            [DATABRICKS]
            HOST = "https://your-workspace.cloud.databricks.com"
            TOKEN = "dapiyour-token"
            JOB_ID = "1234567890"
            ```
            """)
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
        st.info("""
        **Process:**
        1. Upload CSV file
        2. File automatically uploaded to Databricks DBFS
        3. Run ML Pipeline
        """)
        
        # Display current configuration
        st.markdown("### Current Settings")
        st.write(f"**Model:** {selected_model}")
        st.write(f"**Test Size:** {test_size}%")
        st.write(f"**Hyperparameter Tuning:** {'Yes' if enable_tuning else 'No'}")
    
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
                st.write(f"📊 **File Size:** {len(uploaded_file.getvalue()) / (1024*1024):.2f} MB")
                st.write(f"🎯 **Columns:** {len(df_preview.columns)}")
                st.write(f"🔍 **Sample Columns:** {', '.join(df_preview.columns.tolist()[:5])}...")
                
                # Save file locally first
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    st.session_state.uploaded_file_path = file_path
                    
                    # Upload to DBFS
                    if st.button("📤 Upload to Databricks DBFS", type="secondary"):
                        with st.spinner("Uploading file to Databricks..."):
                            dbfs_path = upload_file_to_dbfs(config, file_path, uploaded_file.name)
                            if dbfs_path:
                                st.session_state.dbfs_file_path = dbfs_path
                                st.success("✅ Ready to run pipeline!")
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            st.info("👆 Please upload a CSV file to begin")
    
    with col2:
        st.header("🚀 Pipeline Controls")
        
        if st.session_state.dbfs_file_path:
            # Display current configuration
            st.subheader("Pipeline Configuration")
            st.write(f"**Dataset:** {os.path.basename(st.session_state.dbfs_file_path)}")
            st.write(f"**DBFS Path:** `{st.session_state.dbfs_file_path}`")
            st.write(f"**Model:** {selected_model}")
            st.write(f"**Test Size:** {test_size}%")
            st.write(f"**Tuning:** {'Enabled' if enable_tuning else 'Disabled'}")
            
            # Run pipeline button
            if st.button("🎯 Run ML Pipeline", type="primary", use_container_width=True):
                run_pipeline(st.session_state.dbfs_file_path, selected_model, enable_tuning, test_size)
            
            # Show status
            if st.session_state.job_status == 'running':
                st.info("🔄 Pipeline is running on Databricks...")
            elif st.session_state.job_status == 'completed':
                st.success("✅ Pipeline completed! Check results above.")
            elif st.session_state.job_status == 'failed':
                st.error("❌ Pipeline failed. Check Databricks logs for details.")
        
        elif st.session_state.uploaded_file_path and not st.session_state.dbfs_file_path:
            st.warning("⚠️ Please upload the file to Databricks DBFS first using the button above.")
        else:
            st.info("Please upload a CSV file and transfer it to Databricks DBFS")

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import requests
import json
import time
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
    if 'file_uploaded_to_dbfs' not in st.session_state:
        st.session_state.file_uploaded_to_dbfs = False

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

def check_file_in_dbfs(config, filename):
    """Check if file exists in DBFS"""
    try:
        url = f"{config['host']}/api/2.0/dbfs/list"
        headers = {
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "path": "/FileStore/uploads/"
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            files = response.json().get("files", [])
            for file_info in files:
                if file_info["path"].endswith(filename):
                    return True
            return False
        else:
            st.error(f"❌ Error checking DBFS: {response.text}")
            return False
            
    except Exception as e:
        st.error(f"❌ Error checking file existence: {e}")
        return False

def trigger_databricks_job(config, filename, model_type, enable_tuning, test_size):
    """Trigger Databricks job via API"""
    try:
        url = f"{config['host']}/api/2.0/jobs/run-now"
        
        headers = {
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json"
        }
        
        # DBFS path
        dbfs_path = f"dbfs:/FileStore/uploads/{filename}"
        
        # Job parameters
        data = {
            "job_id": int(config['job_id']),
            "notebook_params": {
                "input_path": dbfs_path,
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

def run_pipeline(filename, model_name, enable_tuning, test_size):
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
        
        # Step 1: Check if file exists in DBFS
        status_text.info("🔍 Checking if file exists in Databricks DBFS...")
        if not check_file_in_dbfs(config, filename):
            st.error(f"❌ File '{filename}' not found in Databricks DBFS!")
            st.error("Please run the CLI command above to upload your file first.")
            return
        
        progress_bar.progress(30)
        
        # Step 2: Trigger job
        status_text.info("🚀 Starting ML pipeline on Databricks...")
        run_id = trigger_databricks_job(config, filename, model_code, enable_tuning, test_size)
        progress_bar.progress(50)
        
        if not run_id:
            st.error("❌ Failed to trigger Databricks job.")
            return
        
        # Store in session state
        st.session_state.job_id = run_id
        st.session_state.job_status = 'running'
        
        # Step 3: Poll for completion
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
        **Three-Step Process:**
        1. Upload CSV file below
        2. Run CLI command to upload to Databricks
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
        st.header("📁 Step 1: Data Upload")
        
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
                file_size_mb = len(uploaded_file.getvalue()) / (1024*1024)
                st.write(f"📊 **File Size:** {file_size_mb:.2f} MB")
                st.write(f"🎯 **Columns:** {len(df_preview.columns)}")
                st.write(f"🔍 **Sample Columns:** {', '.join(df_preview.columns.tolist()[:5])}...")
                
                # Save file locally first
                file_path = save_uploaded_file(uploaded_file)
                if file_path:
                    st.session_state.uploaded_file_path = file_path
                    
                    st.header("📤 Step 2: Upload to Databricks")
                    st.info("""
                    **Run this command in your terminal to upload the file to Databricks:**
                    """)
                    
                    # CLI command with proper formatting
                    cli_command = f'databricks fs cp "{file_path}" "dbfs:/FileStore/uploads/{uploaded_file.name}"'
                    
                    st.code(cli_command, language="bash")
                    
                    # Additional instructions
                    st.info("""
                    **Instructions:**
                    1. Open your terminal/command prompt
                    2. Ensure you have Databricks CLI installed and configured
                    3. Run the command above
                    4. Come back here and click 'Check Upload Status' below
                    """)
                    
                    # Check upload status button
                    if st.button("🔄 Check Upload Status", type="secondary"):
                        if check_file_in_dbfs(config, uploaded_file.name):
                            st.success(f"✅ File '{uploaded_file.name}' found in Databricks DBFS!")
                            st.session_state.file_uploaded_to_dbfs = True
                        else:
                            st.error(f"❌ File '{uploaded_file.name}' not found in DBFS!")
                            st.error("Please run the CLI command above first.")
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            st.info("👆 Please upload a CSV file to begin")
    
    with col2:
        st.header("🚀 Step 3: Run ML Pipeline")
        
        if st.session_state.get('file_uploaded_to_dbfs') and uploaded_file is not None:
            # Display current configuration
            st.subheader("Pipeline Configuration")
            st.write(f"**Dataset:** {uploaded_file.name}")
            st.write(f"**DBFS Path:** `dbfs:/FileStore/uploads/{uploaded_file.name}`")
            st.write(f"**Model:** {selected_model}")
            st.write(f"**Test Size:** {test_size}%")
            st.write(f"**Tuning:** {'Enabled' if enable_tuning else 'Disabled'}")
            
            # Run pipeline button
            if st.button("🎯 Run ML Pipeline", type="primary", use_container_width=True):
                run_pipeline(uploaded_file.name, selected_model, enable_tuning, test_size)
            
            # Show status
            if st.session_state.job_status == 'running':
                st.info("🔄 Pipeline is running on Databricks...")
            elif st.session_state.job_status == 'completed':
                st.success("✅ Pipeline completed! Check results above.")
            elif st.session_state.job_status == 'failed':
                st.error("❌ Pipeline failed. Check Databricks logs for details.")
        
        elif uploaded_file is not None and not st.session_state.get('file_uploaded_to_dbfs'):
            st.warning("""
            ⚠️ **File not yet uploaded to Databricks**
            
            Please complete Step 2:
            1. Run the CLI command shown in Step 2
            2. Click 'Check Upload Status' to verify
            3. Then you can run the pipeline
            """)
        else:
            st.info("Please complete Steps 1 and 2 first")

if __name__ == "__main__":
    main()

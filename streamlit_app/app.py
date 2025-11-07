import streamlit as st
import pandas as pd
import requests
import json
import time
from io import BytesIO

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
    if 'run_id' not in st.session_state:
        st.session_state.run_id = None
    if 'uploaded_file' not in st.session_state:
        st.session_state.uploaded_file = None

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

def trigger_databricks_job(config, model_type, enable_tuning, test_size):
    """Trigger Databricks job via API - NO FILE UPLOAD NEEDED!"""
    try:
        url = f"{config['host']}/api/2.0/jobs/run-now"
        
        headers = {
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json"
        }
        
        # Job parameters - Databricks notebook handles file upload internally
        data = {
            "job_id": int(config['job_id']),
            "notebook_params": {
                "model_type": model_type,
                "enable_tuning": str(enable_tuning).lower(),
                "test_size": str(test_size),
                "output_path": "dbfs:/FileStore/results"
            }
        }
        
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            run_id = response.json()["run_id"]
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
            return {
                "life_cycle_state": state["life_cycle_state"],
                "result_state": state.get("result_state", "UNKNOWN"),
                "state_message": state.get("state_message", "")
            }
        else:
            return {
                "life_cycle_state": "UNKNOWN",
                "result_state": "FAILED", 
                "state_message": f"API Error: {response.text}"
            }
            
    except Exception as e:
        return {
            "life_cycle_state": "ERROR",
            "result_state": "FAILED",
            "state_message": str(e)
        }

def run_pipeline(model_name, enable_tuning, test_size):
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
            "Neural Network": "neural_net"
        }
        
        model_code = model_mapping[model_name]
        
        # Step 1: Trigger job
        status_text.info("🚀 Starting ML pipeline on Databricks...")
        run_id = trigger_databricks_job(config, model_code, enable_tuning, test_size)
        progress_bar.progress(30)
        
        if not run_id:
            st.error("❌ Failed to start Databricks job.")
            return
        
        # Store in session state
        st.session_state.run_id = run_id
        st.session_state.job_status = 'running'
        
        # Step 2: Poll for completion
        status_text.info("🔄 Pipeline running... This may take a few minutes.")
        
        max_attempts = 120  # 10 minutes max
        for attempt in range(max_attempts):
            status_info = get_job_status(config, run_id)
            life_cycle_state = status_info["life_cycle_state"]
            
            # Update progress based on state
            if life_cycle_state == "PENDING":
                progress = 0.3 + (attempt / max_attempts) * 0.2
                status_text.info("⏳ Job queued...")
            elif life_cycle_state == "RUNNING":
                progress = 0.5 + (attempt / max_attempts) * 0.4
                status_text.info("🤖 Processing data and training model...")
            else:
                progress = 0.9
            
            progress_bar.progress(min(progress, 0.9))
            
            if life_cycle_state in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
                break
                
            time.sleep(5)
        
        progress_bar.progress(1.0)
        
        if life_cycle_state == "TERMINATED" and status_info["result_state"] == "SUCCESS":
            status_text.success("✅ Pipeline completed successfully!")
            st.session_state.job_status = 'completed'
            show_results_section(config, run_id)
        else:
            status_text.error(f"❌ Pipeline ended with status: {life_cycle_state}")
            st.error(f"Error: {status_info['state_message']}")
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
        url = f"{config['host']}/api/2.0/jobs/runs/get-output?run_id={run_id}"
        headers = {"Authorization": f"Bearer {config['token']}"}
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            output = response.json()
            if "notebook_output" in output and output["notebook_output"]:
                st.subheader("Execution Logs")
                logs = output["notebook_output"]["result"] if isinstance(output["notebook_output"], dict) else output["notebook_output"]
                st.text_area("Logs", logs, height=200)
            
            # Try to get results from DBFS
            try:
                results_url = f"{config['host']}/api/2.0/dbfs/read"
                headers = {"Authorization": f"Bearer {config['token']}"}
                results_data = {"path": "/FileStore/results/results.json"}
                
                response = requests.get(results_url, headers=headers, json=results_data)
                if response.status_code == 200:
                    import base64
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
                        
                        st.success("🎉 Model training completed successfully!")
            except Exception as e:
                st.info("📊 Check Databricks MLflow for detailed metrics")
                
        else:
            st.info("📋 Check Databricks workspace for complete results")
            
    except Exception as e:
        st.error(f"Error fetching results: {e}")

def main():
    initialize_session_state()
    
    st.title("🚀 Databricks ML Pipeline")
    st.markdown("""
    ### New Workflow:
    1. **Configure settings** in this Streamlit app
    2. **Click "Start ML Pipeline"** 
    3. **Upload your CSV file** directly in Databricks UI
    4. **View results** here when complete
    """)
    
    # Check configurations
    databricks_config = get_databricks_config()
        
    if not databricks_config:
        st.error("❌ Databricks configuration missing.")
        return
    
    # Sidebar for configuration
    with st.sidebar:
        st.header("⚙️ Configuration")
        
        # Model selection
        model_options = [
            "Logistic Regression",
            "Random Forest", 
            "Neural Network"
        ]
        
        selected_model = st.selectbox("Select Model", options=model_options, index=1)
        enable_tuning = st.checkbox("Enable Hyperparameter Tuning", value=False)
        test_size = st.slider("Test Set Size (%)", 10, 40, 20)
        
        st.markdown("---")
        st.header("📊 Current Setup")
        st.success("✅ File upload handled in Databricks!")
        st.write(f"**Model:** {selected_model}")
        st.write(f"**Test Size:** {test_size}%")
        st.write(f"**Tuning:** {'Yes' if enable_tuning else 'No'}")
        
        st.markdown("---")
        st.header("ℹ️ How It Works")
        st.info("""
        **New Process:**
        1. Configure settings here
        2. Start pipeline → Opens Databricks
        3. Upload CSV in Databricks UI
        4. Pipeline runs automatically
        5. View results back here
        
        **Benefits:**
        - ✅ No file size limits
        - ✅ Native Databricks file upload
        - ✅ Simple and reliable
        - ✅ Professional workflow
        """)
    
    # Main area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📋 Pipeline Info")
        st.info("""
        **Important Note:**
        - File upload now happens **directly in Databricks**
        - No file size limits!
        - More reliable and professional
        
        **When you start the pipeline:**
        1. Databricks job will start
        2. You'll see file upload UI in Databricks
        3. Upload your CSV file there
        4. Pipeline continues automatically
        """)
        
        # Optional: Still show file preview if user wants
        uploaded_file = st.file_uploader(
            "Optional: Preview CSV (not uploaded)", 
            type=['csv'],
            help="This is just for preview - actual upload happens in Databricks"
        )
        
        if uploaded_file is not None:
            try:
                df_preview = pd.read_csv(uploaded_file, nrows=5)
                st.subheader("Data Preview")
                st.dataframe(df_preview)
                st.write(f"📏 **Shape:** {df_preview.shape}")
            except Exception as e:
                st.error(f"Error reading file: {e}")
    
    with col2:
        st.header("🚀 Start ML Pipeline")
        
        st.warning("""
        ⚠️ **Before starting:**
        - Make sure your Databricks job is configured
        - The notebook should have file upload widget
        - You'll upload the file in Databricks UI
        """)
        
        st.write("**Pipeline Configuration:**")
        st.write(f"- **Model:** {selected_model}")
        st.write(f"- **Test Size:** {test_size}%")
        st.write(f"- **Hyperparameter Tuning:** {'Yes' if enable_tuning else 'No'}")
        st.write(f"- **File Upload:** In Databricks UI")
        
        if st.button("🎯 Start ML Pipeline", type="primary", use_container_width=True):
            run_pipeline(selected_model, enable_tuning, test_size)
        
        # Show status
        if st.session_state.job_status == 'running':
            st.info("🔄 Pipeline is running...")
            st.info("💡 Check Databricks workspace to upload your file!")
        elif st.session_state.job_status == 'completed':
            st.success("✅ Pipeline completed!")
        elif st.session_state.job_status == 'failed':
            st.error("❌ Pipeline failed.")

if __name__ == "__main__":
    main()

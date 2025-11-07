import streamlit as st
import pandas as pd
import requests
import json
import time
import os

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
    if 'run_id' not in st.session_state:
        st.session_state.run_id = None
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

def trigger_databricks_job(config, model_type, enable_tuning, test_size):
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
                "model_type": model_type,
                "enable_tuning": str(enable_tuning).lower(),
                "test_size": str(test_size),
                "output_path": "dbfs:/FileStore/results"
            }
        }
        
        st.info("🚀 Starting Databricks job...")
        response = requests.post(url, headers=headers, json=data)
        
        if response.status_code == 200:
            run_id = response.json()["run_id"]
            st.success(f"✅ Job started successfully! Run ID: {run_id}")
            return run_id
        else:
            st.error(f"❌ Job trigger failed: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"❌ Error triggering job: {e}")
        return None

def get_job_status(config, run_id):
    """Get job status with detailed information"""
    try:
        url = f"{config['host']}/api/2.0/jobs/runs/get?run_id={run_id}"
        
        headers = {
            "Authorization": f"Bearer {config['token']}"
        }
        
        response = requests.get(url, headers=headers)
        
        if response.status_code == 200:
            result = response.json()
            state = result["state"]
            
            # Get task run state for more detailed status
            task_states = []
            if "tasks" in result:
                for task in result["tasks"]:
                    task_states.append({
                        "task_key": task.get("task_key", "unknown"),
                        "state": task.get("state", {}).get("life_cycle_state", "UNKNOWN"),
                        "result_state": task.get("state", {}).get("result_state", "UNKNOWN")
                    })
            
            return {
                "life_cycle_state": state["life_cycle_state"],
                "result_state": state.get("result_state", "UNKNOWN"),
                "state_message": state.get("state_message", ""),
                "task_states": task_states
            }
        else:
            return {
                "life_cycle_state": "UNKNOWN",
                "result_state": "FAILED", 
                "state_message": f"API Error: {response.text}",
                "task_states": []
            }
            
    except Exception as e:
        return {
            "life_cycle_state": "ERROR",
            "result_state": "FAILED",
            "state_message": str(e),
            "task_states": []
        }

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

def get_detailed_status_message(status_info):
    """Convert job status to user-friendly messages"""
    state = status_info["life_cycle_state"]
    
    status_messages = {
        "PENDING": "🔄 Job is queued and waiting for resources...",
        "RUNNING": "🔄 Job is running...",
        "TERMINATING": "🔄 Job is finishing up...",
        "TERMINATED": "✅ Job completed!",
        "SKIPPED": "⚠️ Job was skipped",
        "INTERNAL_ERROR": "❌ Internal error occurred",
    }
    
    base_message = status_messages.get(state, f"🔄 Current status: {state}")
    
    # Add task-specific status
    task_messages = []
    for task in status_info.get("task_states", []):
        task_state = task["state"]
        if task_state == "PENDING":
            task_messages.append("📁 Waiting for file upload...")
        elif task_state == "RUNNING":
            task_messages.append("🤖 Processing uploaded file...")
        elif task_state == "TERMINATED" and task["result_state"] == "SUCCESS":
            task_messages.append("✅ File processed successfully!")
    
    if task_messages:
        base_message += " | " + " | ".join(task_messages)
    
    return base_message

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
        detailed_status = st.empty()
        
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
        detailed_status.info("Initializing job...")
        
        run_id = trigger_databricks_job(config, model_code, enable_tuning, test_size)
        progress_bar.progress(10)
        
        if not run_id:
            st.error("❌ Failed to start Databricks job.")
            return
        
        # Store in session state
        st.session_state.run_id = run_id
        st.session_state.job_status = 'running'
        
        # Step 2: Poll for completion with detailed status
        status_text.info("🔄 Job started! Waiting for file upload in Databricks...")
        
        max_attempts = 180  # 15 minutes max (5 seconds per check)
        for attempt in range(max_attempts):
            status_info = get_job_status(config, run_id)
            life_cycle_state = status_info["life_cycle_state"]
            
            # Get user-friendly status message
            status_message = get_detailed_status_message(status_info)
            detailed_status.info(status_message)
            
            # Update progress based on state
            if life_cycle_state == "PENDING":
                progress = 0.1 + (attempt / max_attempts) * 0.3
                status_text.info("📁 Please upload your file in the Databricks job now...")
            elif life_cycle_state == "RUNNING":
                progress = 0.4 + (attempt / max_attempts) * 0.5
                status_text.info("🤖 Processing your file and training model...")
            else:
                progress = 0.9
            
            progress_bar.progress(min(progress, 0.9))
            
            if life_cycle_state in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
                break
                
            time.sleep(5)  # Wait 5 seconds between checks
        
        progress_bar.progress(1.0)
        
        if life_cycle_state == "TERMINATED" and status_info["result_state"] == "SUCCESS":
            status_text.success("✅ Pipeline completed successfully!")
            detailed_status.success("All steps completed! Showing results...")
            st.session_state.job_status = 'completed'
            
            # Show results section
            show_results_section(config, run_id)
        else:
            status_text.error(f"❌ Pipeline ended with status: {life_cycle_state}")
            detailed_status.error(f"Result: {status_info['result_state']} - {status_info['state_message']}")
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
                st.subheader("Execution Logs")
                logs = output["notebook_output"]["result"] if isinstance(output["notebook_output"], dict) else output["notebook_output"]
                st.text_area("Logs", logs, height=200)
            
            # Try to get results from DBFS
            try:
                results_url = f"{config['host']}/api/2.0/dbfs/read"
                headers = {"Authorization": f"Bearer {config['token']}"}
                results_data = {
                    "path": "/FileStore/results/results.json"
                }
                
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
                st.info("📊 Check Databricks MLflow for detailed metrics and visualizations")
                
        else:
            st.info("📋 Check Databricks workspace for complete results and visualizations")
            
    except Exception as e:
        st.error(f"Error fetching results: {e}")

def main():
    initialize_session_state()
    
    st.title("🚀 Databricks ML Pipeline")
    st.markdown("Start the pipeline and upload your file directly in Databricks!")
    
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
        st.header("📊 Workflow")
        st.info("""
        **Process:**
        1. Click 'Start ML Pipeline' below
        2. Upload your file in Databricks job
        3. Wait for processing
        4. View results here
        """)
        
        # Display current configuration
        st.markdown("### Current Settings")
        st.write(f"**Model:** {selected_model}")
        st.write(f"**Test Size:** {test_size}%")
        st.write(f"**Tuning:** {'Yes' if enable_tuning else 'No'}")
    
    # Main area
    st.header("🚀 Start ML Pipeline")
    
    st.info("""
    **How it works:**
    - Click the button below to start the Databricks job
    - The job will open in Databricks and wait for file upload
    - Upload your CSV file directly in the Databricks interface
    - The pipeline will process your file and train the model
    - Results will appear here when complete
    """)
    
    # Start pipeline button
    if st.button("🎯 Start ML Pipeline", type="primary", use_container_width=True):
        run_pipeline(selected_model, enable_tuning, test_size)
    
    # Show current status
    if st.session_state.job_status == 'running':
        st.info("🔄 Pipeline is running... Check Databricks for file upload prompt.")
    elif st.session_state.job_status == 'completed':
        st.success("✅ Pipeline completed! Results shown above.")
    elif st.session_state.job_status == 'failed':
        st.error("❌ Pipeline failed. Check Databricks logs for details.")

if __name__ == "__main__":
    main()

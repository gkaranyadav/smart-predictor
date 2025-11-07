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
    if 'volume_file_path' not in st.session_state:
        st.session_state.volume_file_path = None
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

def upload_to_volumes(uploaded_file, config):
    """Upload file to Databricks Volumes via temporary storage"""
    try:
        # Generate unique file name
        timestamp = int(time.time())
        file_name = f"{timestamp}_{uploaded_file.name}"
        
        # Use Volumes path - CHANGE THIS TO YOUR ACTUAL VOLUME PATH
        volume_path = f"/Volumes/demo_ml/main/ml_pipeline/{file_name}"
        
        with st.spinner(f"📤 Uploading {uploaded_file.name} to Databricks Volumes..."):
            # Read file content
            uploaded_file.seek(0)
            file_content = uploaded_file.read()
            
            # Step 1: Create the file in Volumes
            create_url = f"{config['host']}/api/2.0/fs/files{volume_path}"
            headers = {
                "Authorization": f"Bearer {config['token']}",
                "Content-Type": "application/octet-stream"
            }
            
            # Create empty file first
            response = requests.put(create_url, headers=headers, data=b"")
            
            if response.status_code not in [200, 409]:  # 409 = already exists (ok)
                st.error(f"❌ Failed to create file: {response.text}")
                return None
            
            # Step 2: Upload content using simple approach
            # For large files, we'll use a different approach - convert to DataFrame
            st.info("🔄 Converting and uploading data...")
            
            # Convert CSV to JSON and upload as multiple parts if needed
            uploaded_file.seek(0)
            df = pd.read_csv(uploaded_file)
            
            # Save as multiple smaller files if too large
            if len(df) > 10000:  # If more than 10k rows, split
                st.info("📦 Splitting large file into chunks...")
                chunk_size = 5000
                chunks = [df[i:i + chunk_size] for i in range(0, len(df), chunk_size)]
                
                for i, chunk in enumerate(chunks):
                    chunk_path = f"{volume_path}/chunk_{i}.csv"
                    create_chunk_url = f"{config['host']}/api/2.0/fs/files{chunk_path}"
                    requests.put(create_chunk_url, headers=headers, data=b"")
                    
                    # Upload chunk content
                    chunk_csv = chunk.to_csv(index=False)
                    requests.post(
                        f"{config['host']}/api/2.0/fs/files{chunk_path}",
                        headers=headers,
                        data=chunk_csv.encode('utf-8')
                    )
                
                st.success(f"✅ File uploaded as {len(chunks)} chunks to Volumes!")
                return f"{volume_path}/*.csv"
            
            else:
                # Upload as single file
                csv_content = df.to_csv(index=False)
                response = requests.post(
                    f"{config['host']}/api/2.0/fs/files{volume_path}",
                    headers=headers,
                    data=csv_content.encode('utf-8')
                )
                
                if response.status_code == 200:
                    st.success("✅ File successfully uploaded to Volumes!")
                    st.info(f"**Volume Path:** `{volume_path}`")
                    return volume_path
                else:
                    st.error(f"❌ Upload failed: {response.text}")
                    return None
        
    except Exception as e:
        st.error(f"❌ Error uploading to Volumes: {str(e)}")
        return None

def trigger_databricks_job(config, volume_file_path, model_type, enable_tuning, test_size):
    """Trigger Databricks job via API"""
    try:
        url = f"{config['host']}/api/2.0/jobs/run-now"
        
        headers = {
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json"
        }
        
        # Job parameters for Volumes
        data = {
            "job_id": int(config['job_id']),
            "notebook_params": {
                "volume_file_path": volume_file_path,  # Changed to volume path
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

def run_pipeline(volume_file_path, model_name, enable_tuning, test_size):
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
        run_id = trigger_databricks_job(config, volume_file_path, model_code, enable_tuning, test_size)
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
    st.markdown("Upload your dataset and train ML models using **Databricks Volumes**!")
    
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
        st.success("✅ Using Databricks Volumes (No size limits!)")
        st.write(f"**Model:** {selected_model}")
        st.write(f"**Test Size:** {test_size}%")
        st.write(f"**Tuning:** {'Yes' if enable_tuning else 'No'}")
        
        st.markdown("---")
        st.header("ℹ️ Volume Setup Required")
        st.info("""
        **Before using:**
        1. Create a Volume in your Databricks workspace:
           - Catalog: `demo_ml`
           - Schema: `main` 
           - Volume: `ml_pipeline`
        2. Or update the volume path in code
        """)
    
    # Main area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📁 Upload Dataset")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV file", 
            type=['csv'],
            help="Select your dataset file (any size supported!)"
        )
        
        if uploaded_file is not None:
            # Store file in session state
            st.session_state.uploaded_file = uploaded_file
            
            # Preview data
            try:
                df_preview = pd.read_csv(uploaded_file, nrows=5)
                st.subheader("Data Preview")
                st.dataframe(df_preview)
                
                st.subheader("Dataset Info")
                st.write(f"📏 **Shape:** {df_preview.shape}")
                file_size = len(uploaded_file.getvalue()) / (1024*1024)
                st.write(f"📊 **File Size:** {file_size:.2f} MB")
                st.write(f"🎯 **Columns:** {len(df_preview.columns)}")
                st.write(f"🔍 **Sample Columns:** {', '.join(df_preview.columns.tolist()[:3])}...")
                
                # Upload to Volumes
                if not st.session_state.volume_file_path:
                    if st.button("📤 Upload to Databricks Volumes", type="primary", use_container_width=True):
                        volume_path = upload_to_volumes(uploaded_file, databricks_config)
                        if volume_path:
                            st.session_state.volume_file_path = volume_path
                            st.rerun()
                
            except Exception as e:
                st.error(f"Error reading file: {e}")
        else:
            st.info("👆 Please upload a CSV file to begin")
    
    with col2:
        st.header("🚀 Run ML Pipeline")
        
        if st.session_state.volume_file_path:
            st.success("✅ File uploaded to Volumes successfully!")
            st.info(f"**Volume Path:** `{st.session_state.volume_file_path}`")
            
            st.write("**Pipeline Configuration:**")
            st.write(f"- **Model:** {selected_model}")
            st.write(f"- **Test Size:** {test_size}%")
            st.write(f"- **Hyperparameter Tuning:** {'Yes' if enable_tuning else 'No'}")
            st.write(f"- **Storage:** Databricks Volumes")
            
            if st.button("🎯 Start ML Pipeline", type="primary", use_container_width=True):
                run_pipeline(st.session_state.volume_file_path, selected_model, enable_tuning, test_size)
            
            # Show status
            if st.session_state.job_status == 'running':
                st.info("🔄 Pipeline is running...")
            elif st.session_state.job_status == 'completed':
                st.success("✅ Pipeline completed!")
            elif st.session_state.job_status == 'failed':
                st.error("❌ Pipeline failed.")
        
        elif st.session_state.uploaded_file is not None:
            st.info("👆 Click 'Upload to Databricks Volumes' to proceed")
        else:
            st.info("📁 Please upload a CSV file to begin")

if __name__ == "__main__":
    main()

import streamlit as st
import requests
import json
import time
import pandas as pd

# Page configuration
st.set_page_config(
    page_title="Smart Predictor ML Platform", 
    layout="wide",
    page_icon="🚀"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'job_status' not in st.session_state:
        st.session_state.job_status = 'not_started'
    if 'run_id' not in st.session_state:
        st.session_state.run_id = None

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

def trigger_databricks_job(config):
    """Trigger Databricks Auto-ML job"""
    try:
        url = f"{config['host']}/api/2.0/jobs/run-now"
        
        headers = {
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json"
        }
        
        # No parameters needed - everything is automated
        data = {
            "job_id": int(config['job_id'])
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

def run_auto_ml_pipeline():
    """Trigger the Auto-ML pipeline"""
    try:
        config = get_databricks_config()
        if not config:
            st.error("❌ Cannot start pipeline - Databricks configuration missing")
            return
        
        # Show progress
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # Step 1: Trigger job
        status_text.info("🚀 Starting Auto-ML Pipeline on Databricks...")
        run_id = trigger_databricks_job(config)
        progress_bar.progress(20)
        
        if not run_id:
            st.error("❌ Failed to start Databricks job.")
            return
        
        # Store in session state
        st.session_state.run_id = run_id
        st.session_state.job_status = 'running'
        
        # Step 2: Poll for completion
        status_text.info("🔄 Auto-ML Pipeline running... This may take a few minutes.")
        
        max_attempts = 180  # 15 minutes max for auto-ML
        for attempt in range(max_attempts):
            status_info = get_job_status(config, run_id)
            life_cycle_state = status_info["life_cycle_state"]
            
            # Update progress based on state
            if life_cycle_state == "PENDING":
                progress = 0.2 + (attempt / max_attempts) * 0.3
                status_text.info("⏳ Job queued...")
            elif life_cycle_state == "RUNNING":
                progress = 0.5 + (attempt / max_attempts) * 0.4
                status_text.info("🤖 Auto-ML: Detecting target, preprocessing, training models...")
            else:
                progress = 0.9
            
            progress_bar.progress(min(progress, 0.9))
            
            if life_cycle_state in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
                break
                
            time.sleep(5)
        
        progress_bar.progress(1.0)
        
        if life_cycle_state == "TERMINATED" and status_info["result_state"] == "SUCCESS":
            status_text.success("✅ Auto-ML Pipeline completed successfully!")
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
    """Display results from the completed Auto-ML job"""
    try:
        st.markdown("---")
        st.header("📊 Auto-ML Results")
        
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
                results_data = {"path": "/FileStore/auto_ml_results/results.json"}
                
                response = requests.get(results_url, headers=headers, json=results_data)
                if response.status_code == 200:
                    import base64
                    results_content = base64.b64decode(response.json()["data"]).decode('utf-8')
                    results = json.loads(results_content)
                    
                    if results.get("status") == "success":
                        display_auto_ml_results(results)
                        
            except Exception as e:
                st.info("📊 Check Databricks workspace for complete results")
                
        else:
            st.info("📋 Check Databricks workspace for complete results")
            
    except Exception as e:
        st.error(f"Error fetching results: {e}")

def display_auto_ml_results(results):
    """Display comprehensive Auto-ML results"""
    
    # Model Performance
    st.subheader("🎯 Model Performance Comparison")
    
    if 'model_comparison' in results:
        model_metrics = results['model_comparison']
        metrics_df = pd.DataFrame(model_metrics).T
        
        # Show best model
        best_model = results.get('best_model', {})
        st.success(f"🏆 **Best Model**: {best_model.get('name', 'N/A')} "
                  f"(Accuracy: {best_model.get('accuracy', 0):.4f})")
        
        # Display metrics table
        st.dataframe(metrics_df)
    
    # Dataset Info
    st.subheader("📋 Dataset Analysis")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"**Dataset**: {results.get('dataset_name', 'N/A')}")
        st.info(f"**Target Column**: {results.get('target_column', 'N/A')}")
    
    with col2:
        if 'dataset_info' in results:
            st.info(f"**Samples**: {results['dataset_info'].get('rows', 'N/A'):,}")
            st.info(f"**Features**: {results['dataset_info'].get('features', 'N/A')}")
    
    with col3:
        if 'problem_type' in results:
            st.info(f"**Problem Type**: {results['problem_type']}")
            st.info(f"**Target Distribution**: {results.get('target_distribution', 'N/A')}")
    
    # Feature Importance
    if 'feature_importance' in results:
        st.subheader("🔍 Top Feature Importance")
        features = results['feature_importance']
        if features:
            top_features = list(features.items())[:10]
            for feature, importance in top_features:
                st.write(f"`{feature}`: {importance:.4f}")

def main():
    initialize_session_state()
    
    st.title("🚀 Smart Predictor - Auto ML Platform")
    st.markdown("### Fully Automated Machine Learning - Upload any dataset and get predictions!")
    
    # Check configurations
    databricks_config = get_databricks_config()
        
    if not databricks_config:
        st.error("❌ Databricks configuration missing.")
        return
    
    # Sidebar for information
    with st.sidebar:
        st.header("⚡ Auto-ML Features")
        st.info("""
        **What happens automatically:**
        - 📁 File upload & table creation
        - 🎯 Smart target detection  
        - 🔧 Auto preprocessing
        - 🤖 Multiple model training
        - 📊 Model comparison
        - 🏆 Best model selection
        """)
        
        st.header("📁 Supported Datasets")
        st.info("""
        **Any CSV file with:**
        - Classification or Regression
        - Any number of features
        - Any data types
        - Automatic target detection
        """)
    
    # Main area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🎯 How It Works")
        st.info("""
        1. **Upload your CSV** in Databricks notebook
        2. **Auto-ML pipeline** detects everything automatically:
           - Target column
           - Problem type (classification/regression)
           - Data preprocessing needed
           - Best model selection
        3. **View results** here with model comparison
        """)
        
        st.warning("""
        ⚠️ **Important:**
        - Upload your CSV file in the Databricks notebook
        - The system will automatically detect the target column
        - No manual configuration needed!
        """)
    
    with col2:
        st.header("🚀 Start Auto-ML Pipeline")
        
        st.success("""
        **Current Status:**
        - ✅ Streamlit Dashboard: Ready
        - ✅ Databricks Auto-ML: Ready
        - ✅ Model Training: Automated
        - ✅ Results Display: Integrated
        """)
        
        if st.button("🎯 Start Fully Automated ML Pipeline", type="primary", use_container_width=True):
            run_auto_ml_pipeline()
        
        # Show status
        if st.session_state.job_status == 'running':
            st.info("🔄 Auto-ML Pipeline is running...")
            st.info("💡 The system is automatically:")
            st.info("   - Detecting target column")
            st.info("   - Preprocessing data")  
            st.info("   - Training multiple models")
            st.info("   - Selecting best model")
        elif st.session_state.job_status == 'completed':
            st.success("✅ Auto-ML Pipeline completed!")
        elif st.session_state.job_status == 'failed':
            st.error("❌ Pipeline failed.")

if __name__ == "__main__":
    main()

import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page configuration
st.set_page_config(
    page_title="Smart Predictor - Auto ML Platform", 
    layout="wide",
    page_icon="🚀"
)

def initialize_session_state():
    """Initialize session state variables"""
    if 'job_status' not in st.session_state:
        st.session_state.job_status = 'not_started'
    if 'run_id' not in st.session_state:
        st.session_state.run_id = None
    if 'auto_ml_results' not in st.session_state:
        st.session_state.auto_ml_results = None

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
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.info("🚀 Starting Auto-ML Pipeline on Databricks...")
        run_id = trigger_databricks_job(config)
        progress_bar.progress(20)
        
        if not run_id:
            st.error("❌ Failed to start Databricks job.")
            return
        
        st.session_state.run_id = run_id
        st.session_state.job_status = 'running'
        
        status_text.info("🔄 Auto-ML Pipeline running... This may take a few minutes.")
        
        max_attempts = 180
        for attempt in range(max_attempts):
            status_info = get_job_status(config, run_id)
            life_cycle_state = status_info["life_cycle_state"]
            
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
            load_and_display_results(config)
        else:
            status_text.error(f"❌ Pipeline ended with status: {life_cycle_state}")
            st.error(f"Error: {status_info['state_message']}")
            st.session_state.job_status = 'failed'
        
    except Exception as e:
        st.error(f"❌ Error in pipeline: {e}")
        st.session_state.job_status = 'failed'

def load_and_display_results(config):
    """Load and display Auto-ML results"""
    try:
        results_url = f"{config['host']}/api/2.0/dbfs/read"
        headers = {"Authorization": f"Bearer {config['token']}"}
        results_data = {"path": "/FileStore/auto_ml_results/results.json"}
        
        response = requests.get(results_url, headers=headers, json=results_data)
        if response.status_code == 200:
            import base64
            results_content = base64.b64decode(response.json()["data"]).decode('utf-8')
            results = json.loads(results_content)
            st.session_state.auto_ml_results = results
        else:
            st.error("❌ Could not load results from Databricks")
    except Exception as e:
        st.error(f"Error loading results: {e}")

def display_data_analyst_dashboard(results):
    """Display comprehensive data analyst dashboard"""
    st.header("📊 Data Analyst Dashboard")
    
    # Dataset Overview
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Samples", f"{results.get('dataset_info', {}).get('rows', 0):,}")
    with col2:
        st.metric("Features", results.get('dataset_info', {}).get('features', 0))
    with col3:
        st.metric("Target Variable", results.get('target_column', 'N/A'))
    with col4:
        st.metric("Problem Type", results.get('problem_type', 'N/A').replace('_', ' ').title())
    
    # First 5 rows preview
    st.subheader("🔍 Data Preview - First 5 Rows")
    try:
        config = get_databricks_config()
        if config:
            sql_url = f"{config['host']}/api/2.0/sql/statements"
            headers = {"Authorization": f"Bearer {config['token']}"}
            
            sql_data = {
                "statement": f"SELECT * FROM hive_metastore.default.auto_ml_dataset_{int(time.time())} LIMIT 5",
                "warehouse_id": "auto"
            }
            
            response = requests.post(sql_url, headers=headers, json=sql_data)
            if response.status_code == 200:
                result = response.json()
                if 'result' in result and 'data_array' in result['result']:
                    data = result['result']['data_array']
                    columns = [col['name'] for col in result['result']['manifest']['schema']['columns']]
                    preview_df = pd.DataFrame(data, columns=columns)
                    st.dataframe(preview_df, use_container_width=True)
    except:
        st.info("💡 Data preview available in Databricks notebook")
    
    # Model Performance Comparison
    st.subheader("🎯 Model Performance Comparison")
    if 'model_comparison' in results:
        model_metrics = results['model_comparison']
        
        # Create metrics comparison
        metrics_data = []
        for model_name, metrics in model_metrics.items():
            if 'error' not in metrics:
                if results.get('problem_type') != 'regression':
                    metrics_data.append({
                        'Model': model_name,
                        'Accuracy': metrics.get('accuracy', 0),
                        'Precision': metrics.get('precision', 0),
                        'Recall': metrics.get('recall', 0),
                        'F1 Score': metrics.get('f1_score', 0),
                        'ROC AUC': metrics.get('roc_auc', 0)
                    })
                else:
                    metrics_data.append({
                        'Model': model_name,
                        'R² Score': metrics.get('r2', 0),
                        'RMSE': metrics.get('rmse', 0),
                        'MAE': metrics.get('mae', 0)
                    })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            st.dataframe(metrics_df, use_container_width=True)
            
            # Best model highlight
            best_model = results.get('best_model', {})
            if best_model.get('name'):
                st.success(f"🏆 **Best Model**: {best_model['name']} (Score: {best_model.get('score', 0):.4f})")
    
    # Feature Importance
    if 'feature_importance' in results and results['feature_importance']:
        st.subheader("🔍 Top 10 Feature Importance")
        features = results['feature_importance']
        top_features = dict(sorted(features.items(), key=lambda x: x[1], reverse=True)[:10])
        
        fig = px.bar(x=list(top_features.values()), y=list(top_features.keys()),
                     orientation='h', title="Top 10 Most Important Features")
        fig.update_layout(xaxis_title="Importance", yaxis_title="Features")
        st.plotly_chart(fig, use_container_width=True)
    
    # Target Distribution
    st.subheader("📈 Target Distribution Analysis")
    col1, col2 = st.columns(2)
    
    with col1:
        if 'dataset_info' in results and 'target_distribution' in results['dataset_info']:
            dist_str = results['dataset_info']['target_distribution']
            try:
                # Try to parse the distribution string
                if dist_str != 'N/A':
                    st.write("**Class Distribution:**")
                    st.json(dist_str)
            except:
                st.write("**Target Stats:**")
                st.write(f"Distribution: {dist_str}")
    
    with col2:
        # Problem type insights
        problem_type = results.get('problem_type', '')
        if problem_type == 'binary_classification':
            st.info("**Binary Classification** - Predicting between two classes")
        elif problem_type == 'multiclass_classification':
            st.info("**Multi-class Classification** - Predicting between multiple classes")
        elif problem_type == 'regression':
            st.info("**Regression** - Predicting continuous values")
    
    # Execution Details
    st.subheader("⚙️ Execution Details")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write(f"**Dataset:** {results.get('dataset_name', 'N/A')}")
        st.write(f"**Target Column:** {results.get('target_column', 'N/A')}")
        st.write(f"**Total Models Trained:** {len(results.get('model_comparison', {}))}")
    
    with col2:
        st.write(f"**Execution Time:** {results.get('execution_time', 0):.2f} seconds")
        st.write(f"**Completed:** {results.get('timestamp', 'N/A')}")

def main():
    initialize_session_state()
    
    st.title("🚀 Smart Predictor - Auto ML Platform")
    st.markdown("### Fully Automated Machine Learning with Data Analyst Dashboard")
    
    databricks_config = get_databricks_config()
    if not databricks_config:
        st.error("❌ Databricks configuration missing.")
        return
    
    # Sidebar
    with st.sidebar:
        st.header("⚡ Auto-ML Features")
        st.info("""
        **Automated Process:**
        - 📁 File upload & table creation
        - 🎯 Smart target detection  
        - 🔧 Auto preprocessing
        - 🤖 Multiple model training
        - 📊 Model comparison
        - 🏆 Best model selection
        """)
        
        st.header("📊 Data Analyst Features")
        st.info("""
        **Interactive Dashboard:**
        - 🔍 Data preview & statistics
        - 📈 Model performance comparison
        - 🔍 Feature importance analysis
        - 📊 Target distribution
        - ⚙️ Execution insights
        """)
    
    # Main area
    tab1, tab2 = st.tabs(["🚀 Auto-ML Pipeline", "📊 Data Analyst Dashboard"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("🎯 How It Works")
            st.info("""
            1. **Upload your CSV** in Databricks notebook
            2. **Enter file path** in the input box
            3. **Auto-ML pipeline** detects everything automatically:
               - Target column (SMART detection)
               - Problem type (classification/regression)
               - Data preprocessing needed
               - Best model selection
            4. **View results** in the Data Analyst Dashboard
            """)
            
            st.warning("""
            ⚠️ **Smart Target Detection:**
            - Prefers binary classification targets
            - Skips high-unique columns like Income, Age
            - Uses intelligent keyword matching
            - First column fallback (not last!)
            """)
        
        with col2:
            st.header("🚀 Start Auto-ML Pipeline")
            
            st.success("""
            **Current Status:**
            - ✅ Streamlit Dashboard: Ready
            - ✅ Databricks Auto-ML: Ready  
            - ✅ Data Analyst Features: Ready
            - ✅ Smart Target Detection: Enabled
            """)
            
            if st.button("🎯 Start Fully Automated ML Pipeline", type="primary", use_container_width=True):
                run_auto_ml_pipeline()
            
            if st.session_state.job_status == 'running':
                st.info("🔄 Auto-ML Pipeline is running...")
                st.info("💡 The system is automatically:")
                st.info("   - Detecting target column (SMART)")
                st.info("   - Preprocessing data")  
                st.info("   - Training multiple models")
                st.info("   - Selecting best model")
            elif st.session_state.job_status == 'completed':
                st.success("✅ Auto-ML Pipeline completed!")
            elif st.session_state.job_status == 'failed':
                st.error("❌ Pipeline failed.")
    
    with tab2:
        if st.session_state.auto_ml_results:
            display_data_analyst_dashboard(st.session_state.auto_ml_results)
        else:
            st.info("👆 Run the Auto-ML pipeline first to see the Data Analyst Dashboard!")
            st.image("https://via.placeholder.com/600x300/4A90E2/FFFFFF?text=Data+Analyst+Dashboard+Ready", use_column_width=True)

if __name__ == "__main__":
    main()

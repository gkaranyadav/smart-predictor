import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
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
    
    # Data Quality & EDA Section
    st.subheader("🔍 Data Quality & EDA")
    col1, col2 = st.columns(2)
    
    with col1:
        st.info("**✅ Data Quality Check**")
        st.write("• Automatic missing value handling")
        st.write("• Categorical encoding applied")
        st.write("• Data type validation complete")
        st.write("• Feature scaling performed")
    
    with col2:
        st.info("**📈 EDA Insights**")
        problem_type = results.get('problem_type', '')
        if problem_type == 'binary_classification':
            st.write("• Binary classification problem")
            st.write("• Perfect for medical diagnosis")
            st.write("• Model interpretability high")
        elif problem_type == 'multiclass_classification':
            st.write("• Multi-class classification")
            st.write("• Multiple outcome categories")
            st.write("• Balanced accuracy important")
    
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
                        'Model': model_name.replace('_', ' ').title(),
                        'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                        'Precision': f"{metrics.get('precision', 0):.4f}",
                        'Recall': f"{metrics.get('recall', 0):.4f}",
                        'F1 Score': f"{metrics.get('f1_score', 0):.4f}",
                        'ROC AUC': f"{metrics.get('roc_auc', 0):.4f}" if 'roc_auc' in metrics else 'N/A'
                    })
                else:
                    metrics_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'R² Score': f"{metrics.get('r2', 0):.4f}",
                        'RMSE': f"{metrics.get('rmse', 0):.4f}",
                        'MAE': f"{metrics.get('mae', 0):.4f}"
                    })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            
            # Highlight best model
            def highlight_best_model(row):
                best_model = results.get('best_model', {}).get('name', '').replace('_', ' ').title()
                if row['Model'] == best_model:
                    return ['background-color: #90EE90'] * len(row)
                return [''] * len(row)
            
            st.dataframe(metrics_df.style.apply(highlight_best_model, axis=1), use_container_width=True)
            
            # Best model highlight
            best_model = results.get('best_model', {})
            if best_model.get('name'):
                score = best_model.get('score', 0)
                st.success(f"🏆 **Best Performing Model**: **{best_model['name'].replace('_', ' ').title()}** (Score: {score:.4f})")
                
                # Performance interpretation
                if score >= 0.9:
                    st.info("🔥 **Excellent Performance** - Model is highly accurate and reliable")
                elif score >= 0.8:
                    st.info("✅ **Good Performance** - Model performs well for practical use")
                elif score >= 0.7:
                    st.info("⚠️ **Moderate Performance** - Consider feature engineering or different algorithms")
                else:
                    st.warning("🔧 **Needs Improvement** - Review data quality and model selection")
    
    # Feature Importance with COLORFUL Chart
    if 'feature_importance' in results and results['feature_importance']:
        st.subheader("🔍 Feature Importance Analysis")
        
        features = results['feature_importance']
        if features:
            # Get top 10 features
            top_features = dict(sorted(features.items(), key=lambda x: x[1], reverse=True)[:10])
            
            # Create colorful bar chart
            fig = px.bar(
                x=list(top_features.values()), 
                y=list(top_features.keys()),
                orientation='h',
                title="Top 10 Most Important Features",
                color=list(top_features.values()),
                color_continuous_scale='viridis'
            )
            
            fig.update_layout(
                xaxis_title="Feature Importance Score",
                yaxis_title="Features",
                showlegend=False,
                height=400
            )
            
            fig.update_traces(
                marker_line_color='black',
                marker_line_width=1
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Feature importance insights
            col1, col2 = st.columns(2)
            with col1:
                most_important = list(top_features.keys())[0]
                st.info(f"**Most Important Feature:** `{most_important}`")
            
            with col2:
                st.info(f"**Total Features Analyzed:** {len(features)}")
    
    # Target Distribution with PROPER Chart
    st.subheader("📈 Target Distribution Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Create target distribution chart
        try:
            dist_str = results.get('dataset_info', {}).get('target_distribution', '{}')
            # Clean the distribution string
            dist_str = dist_str.replace("'", '"')
            target_dist = json.loads(dist_str)
            
            if target_dist:
                # Create pie chart
                fig_pie = px.pie(
                    values=list(target_dist.values()),
                    names=[f"Class {k}" for k in target_dist.keys()],
                    title="Target Class Distribution"
                )
                st.plotly_chart(fig_pie, use_container_width=True)
        except:
            st.info("📊 Target distribution visualization available")
    
    with col2:
        # Problem insights
        problem_type = results.get('problem_type', '')
        target_col = results.get('target_column', '')
        
        st.info("**🎯 Problem Insights**")
        if problem_type == 'binary_classification':
            st.write("• **Binary Classification**")
            st.write("• Predicting between two classes")
            st.write("• Ideal for yes/no predictions")
            st.write(f"• Target: `{target_col}`")
        elif problem_type == 'multiclass_classification':
            st.write("• **Multi-class Classification**")
            st.write("• Predicting multiple categories")
            st.write("• Requires balanced evaluation")
            st.write(f"• Target: `{target_col}`")
    
    # Execution Summary
    st.subheader("⚙️ Pipeline Execution Summary")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Execution Time", f"{results.get('execution_time', 0):.2f}s")
    
    with col2:
        st.metric("Models Trained", len(results.get('model_comparison', {})))
    
    with col3:
        st.metric("Status", "✅ Success" if results.get('status') == 'success' else "❌ Failed")
    
    # Additional Insights
    st.info("**💡 Auto-ML Insights**")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("• **Smart Target Detection** - Automatically identified the prediction target")
        st.write("• **Auto Preprocessing** - Handled missing values and encoding")
    
    with col2:
        st.write("• **Multi-Model Training** - Tested multiple algorithms")
        st.write("• **Best Model Selection** - Selected optimal model based on performance")

def main():
    initialize_session_state()
    
    st.title("🚀 Smart Predictor - Auto ML Platform")
    st.markdown("### Fully Automated Machine Learning with Professional Analytics Dashboard")
    
    databricks_config = get_databricks_config()
    if not databricks_config:
        st.error("❌ Databricks configuration missing.")
        return
    
    # Sidebar - CLEANED UP
    with st.sidebar:
        st.header("⚡ Features")
        st.success("""
        **Automated ML Pipeline:**
        - Smart Target Detection
        - Auto Preprocessing  
        - Multi-Model Training
        - Best Model Selection
        """)
        
        st.header("📊 Analytics")
        st.info("""
        **Professional Dashboard:**
        - Data Quality Reports
        - Model Performance
        - Feature Importance
        - Execution Insights
        """)
    
    # Main area with tabs
    tab1, tab2 = st.tabs(["🚀 Auto-ML Pipeline", "📊 Analytics Dashboard"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.header("🎯 How It Works")
            st.info("""
            1. **Upload CSV** in Databricks
            2. **Enter file path** 
            3. **Run Auto-ML Pipeline**
            4. **View Analytics Dashboard**
            
            **Smart Features:**
            - Automatic target detection
            - Data preprocessing
            - Model comparison
            - Best model selection
            """)
        
        with col2:
            st.header("🚀 Start Pipeline")
            
            if st.button("🎯 Start Auto-ML Pipeline", type="primary", use_container_width=True):
                run_auto_ml_pipeline()
            
            # Status display
            if st.session_state.job_status == 'running':
                st.info("🔄 Pipeline running...")
            elif st.session_state.job_status == 'completed':
                st.success("✅ Pipeline completed!")
            elif st.session_state.job_status == 'failed':
                st.error("❌ Pipeline failed.")
    
    with tab2:
        if st.session_state.auto_ml_results:
            display_data_analyst_dashboard(st.session_state.auto_ml_results)
        else:
            st.info("👆 Run the Auto-ML pipeline first to see the analytics dashboard!")
            st.image("https://images.unsplash.com/photo-1551288049-bebda4e38f71?w=600&h=300&fit=crop", 
                    use_column_width=True, caption="Ready to analyze your data!")

if __name__ == "__main__":
    main()

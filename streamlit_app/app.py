import streamlit as st
import requests
import json
import time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
from datetime import datetime

# Page configuration - UNIVERSAL PROFESSIONAL THEME
st.set_page_config(
    page_title="Smart Predictor - Universal Auto ML", 
    layout="wide",
    page_icon="🚀",
    initial_sidebar_state="expanded"
)

# Custom CSS for Professional Universal Theme
st.markdown("""
<style>
    /* Main Theme - Universal Professional */
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
        font-weight: 700;
    }
    
    .sub-header {
        font-size: 1.2rem;
        color: #2e86ab;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    /* Card Styling */
    .card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
    }
    
    /* Button Styling */
    .stButton button {
        background: linear-gradient(135deg, #1f77b4 0%, #2e86ab 100%);
        color: white;
        border: none;
        padding: 0.5rem 2rem;
        border-radius: 25px;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(31, 119, 180, 0.3);
    }
    
    /* Toggle Switch Styling */
    .stCheckbox label {
        font-weight: 600;
        color: #1f77b4;
    }
    
    /* Progress Bar */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #1f77b4 0%, #2e86ab 100%);
    }
    
    /* Sidebar Styling */
    .css-1d391kg {
        background: #f8f9fa;
    }
</style>
""", unsafe_allow_html=True)

def initialize_session_state():
    """Initialize session state variables"""
    if 'job_status' not in st.session_state:
        st.session_state.job_status = 'not_started'
    if 'run_id' not in st.session_state:
        st.session_state.run_id = None
    if 'auto_ml_results' not in st.session_state:
        st.session_state.auto_ml_results = None
    if 'pipeline_config' not in st.session_state:
        st.session_state.pipeline_config = {
            'enable_tuning': False,
            'use_ai_assist': True
        }

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

def trigger_databricks_job(config, pipeline_config):
    """Trigger Databricks Auto-ML job with configuration"""
    try:
        url = f"{config['host']}/api/2.0/jobs/run-now"
        
        headers = {
            "Authorization": f"Bearer {config['token']}",
            "Content-Type": "application/json"
        }
        
        data = {
            "job_id": int(config['job_id']),
            "notebook_params": {
                "enable_tuning": str(pipeline_config['enable_tuning']).lower(),
                "use_ai_assist": str(pipeline_config['use_ai_assist']).lower()
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

def run_auto_ml_pipeline():
    """Trigger the Auto-ML pipeline with user configuration"""
    try:
        config = get_databricks_config()
        if not config:
            st.error("❌ Cannot start pipeline - Databricks configuration missing")
            return
        
        # Show configuration summary
        st.info(f"⚙️ **Pipeline Configuration:**")
        col1, col2 = st.columns(2)
        with col1:
            st.write(f"🤖 AI Assistance: {'✅ Enabled' if st.session_state.pipeline_config['use_ai_assist'] else '❌ Disabled'}")
        with col2:
            st.write(f"🔧 Hyperparameter Tuning: {'✅ Enabled' if st.session_state.pipeline_config['enable_tuning'] else '❌ Disabled'}")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        status_text.info("🚀 Starting Enhanced Auto-ML Pipeline on Databricks...")
        run_id = trigger_databricks_job(config, st.session_state.pipeline_config)
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
                status_text.info("🤖 Auto-ML: Smart target detection, enhanced EDA, model training...")
            else:
                progress = 0.9
            
            progress_bar.progress(min(progress, 0.9))
            
            if life_cycle_state in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
                break
                
            time.sleep(5)
        
        progress_bar.progress(1.0)
        
        if life_cycle_state == "TERMINATED" and status_info["result_state"] == "SUCCESS":
            status_text.success("✅ Enhanced Auto-ML Pipeline completed successfully!")
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

def create_enhanced_eda_visualizations(results):
    """Create enhanced EDA visualizations"""
    try:
        # Target Distribution
        if 'dataset_info' in results and 'target_distribution' in results['dataset_info']:
            dist_data = results['dataset_info']['target_distribution']
            if isinstance(dist_data, str):
                dist_data = json.loads(dist_data.replace("'", '"'))
            
            if dist_data:
                # Create beautiful pie chart
                fig_pie = px.pie(
                    values=list(dist_data.values()),
                    names=[f"Class {k}" for k in dist_data.keys()],
                    title="🎯 Target Distribution",
                    color_discrete_sequence=px.colors.sequential.Blues_r
                )
                fig_pie.update_traces(
                    textposition='inside',
                    textinfo='percent+label',
                    marker=dict(line=dict(color='#000000', width=2))
                )
                fig_pie.update_layout(
                    height=400,
                    showlegend=True
                )
                return fig_pie
    except Exception as e:
        st.warning(f"Could not create EDA visualization: {e}")
    return None

def create_feature_importance_chart(feature_importance):
    """Create colorful feature importance chart"""
    if not feature_importance:
        return None
    
    try:
        # Get top 10 features
        features = list(feature_importance.keys())[:10]
        importance = list(feature_importance.values())[:10]
        
        # Create colorful horizontal bar chart
        fig = px.bar(
            x=importance,
            y=features,
            orientation='h',
            title="🔍 Top 10 Feature Importance",
            color=importance,
            color_continuous_scale='viridis'
        )
        
        fig.update_layout(
            xaxis_title="Importance Score",
            yaxis_title="Features",
            showlegend=False,
            height=400,
            yaxis={'categoryorder':'total ascending'}
        )
        
        fig.update_traces(
            marker_line_color='black',
            marker_line_width=1,
            hovertemplate='<b>%{y}</b><br>Importance: %{x:.4f}<extra></extra>'
        )
        
        return fig
    except Exception as e:
        st.warning(f"Could not create feature importance chart: {e}")
        return None

def display_enhanced_analytics_dashboard(results):
    """Display comprehensive enhanced analytics dashboard"""
    
    # Header with gradient
    st.markdown('<div class="main-header">📊 Universal Analytics Dashboard</div>', unsafe_allow_html=True)
    
    # Dataset Overview Cards
    st.subheader("📈 Dataset Overview")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Total Samples</h3>
            <h2>{results.get('dataset_info', {}).get('rows', 0):,}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Features</h3>
            <h2>{results.get('dataset_info', {}).get('features', 0)}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-card">
            <h3>Target</h3>
            <h4>{results.get('target_column', 'N/A')}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        problem_type = results.get('problem_type', 'N/A').replace('_', ' ').title()
        st.markdown(f"""
        <div class="metric-card">
            <h3>Problem Type</h3>
            <h4>{problem_type}</h4>
        </div>
        """, unsafe_allow_html=True)
    
    # AI Insights Section
    if results.get('ai_insights'):
        st.markdown("---")
        st.subheader("💬 AI Insights & Recommendations")
        st.info(f"**🤖 SMART ANALYSIS:** {results['ai_insights']}")
    
    # Enhanced EDA Section
    st.markdown("---")
    st.subheader("🔍 Enhanced Data Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("✅ Data Quality Report")
        
        # Enhanced EDA Insights
        problem_type = results.get('problem_type', '')
        if problem_type == 'binary_classification':
            st.write("• **Binary Classification** detected")
            st.write("• Perfect for yes/no predictions")
            st.write("• Model interpretability: High")
            st.write("• Business impact: Direct decision support")
        elif problem_type == 'multiclass_classification':
            st.write("• **Multi-class Classification** detected")  
            st.write("• Multiple category prediction")
            st.write("• Balanced accuracy important")
            st.write("• Use case: Categorization systems")
        elif problem_type == 'regression':
            st.write("• **Regression** problem detected")
            st.write("• Predicting continuous values")
            st.write("• R² score interpretation key")
            st.write("• Use case: Forecasting, pricing")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("📊 EDA Visualizations")
        
        # Interactive EDA Chart
        eda_chart = create_enhanced_eda_visualizations(results)
        if eda_chart:
            st.plotly_chart(eda_chart, use_container_width=True)
        else:
            st.info("🎯 Target distribution analysis available after pipeline execution")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Model Performance Comparison
    st.markdown("---")
    st.subheader("🎯 Model Performance Comparison")
    
    if 'model_comparison' in results:
        model_metrics = results['model_comparison']
        
        # Create interactive metrics comparison
        metrics_data = []
        for model_name, metrics in model_metrics.items():
            if 'error' not in metrics:
                model_display_name = model_name.replace('_', ' ').title()
                
                if results.get('problem_type') != 'regression':
                    metrics_data.append({
                        'Model': model_display_name,
                        'Accuracy': f"{metrics.get('accuracy', 0):.4f}",
                        'Precision': f"{metrics.get('precision', 0):.4f}",
                        'Recall': f"{metrics.get('recall', 0):.4f}",
                        'F1 Score': f"{metrics.get('f1_score', 0):.4f}",
                        'ROC AUC': f"{metrics.get('roc_auc', 'N/A')}"
                    })
                else:
                    metrics_data.append({
                        'Model': model_display_name,
                        'R² Score': f"{metrics.get('r2', 0):.4f}",
                        'RMSE': f"{metrics.get('rmse', 0):.4f}",
                        'MAE': f"{metrics.get('mae', 0):.4f}",
                        'MSE': f"{metrics.get('mse', 0):.4f}"
                    })
        
        if metrics_data:
            metrics_df = pd.DataFrame(metrics_data)
            
            # Highlight best model
            def highlight_best_model(row):
                best_model = results.get('best_model', {}).get('name', '').replace('_', ' ').title()
                if row['Model'] == best_model:
                    return ['background-color: #e6f3ff'] * len(row)
                return [''] * len(row)
            
            st.dataframe(metrics_df.style.apply(highlight_best_model, axis=1), use_container_width=True)
            
            # Best model performance with emoji indicators
            best_model = results.get('best_model', {})
            if best_model.get('name'):
                score = best_model.get('score', 0)
                model_name = best_model['name'].replace('_', ' ').title()
                
                # Performance interpretation with emojis
                if score >= 0.9:
                    performance_emoji = "🔥"
                    performance_text = "EXCELLENT"
                    color = "green"
                elif score >= 0.8:
                    performance_emoji = "✅"
                    performance_text = "GOOD"
                    color = "blue"
                elif score >= 0.7:
                    performance_emoji = "⚠️"
                    performance_text = "MODERATE"
                    color = "orange"
                else:
                    performance_emoji = "🔧"
                    performance_text = "NEEDS IMPROVEMENT"
                    color = "red"
                
                st.success(f"🏆 **Best Performing Model**: **{model_name}** {performance_emoji}")
                st.markdown(f"<h3 style='color: {color};'>Performance: {performance_text} (Score: {score:.4f})</h3>", unsafe_allow_html=True)
    
    # Feature Importance with Interactive Chart
    if 'feature_importance' in results and results['feature_importance']:
        st.markdown("---")
        st.subheader("🔍 Feature Importance Analysis")
        
        best_model = results.get('best_model', {}).get('name')
        if best_model and best_model in results['feature_importance']:
            feature_chart = create_feature_importance_chart(results['feature_importance'][best_model])
            if feature_chart:
                st.plotly_chart(feature_chart, use_container_width=True)
            
            # Feature insights
            features = results['feature_importance'][best_model]
            if features:
                top_features = sorted(features.items(), key=lambda x: x[1], reverse=True)[:5]
                
                col1, col2 = st.columns(2)
                with col1:
                    st.info(f"**🏆 Most Important Feature:** `{top_features[0][0]}`")
                with col2:
                    st.info(f"**📊 Total Features Analyzed:** {len(features)}")
    
    # Pipeline Execution Summary
    st.markdown("---")
    st.subheader("⚙️ Pipeline Execution Summary")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Execution Time", f"{results.get('execution_time', 0):.2f}s")
    
    with col2:
        st.metric("Models Trained", len(results.get('model_comparison', {})))
    
    with col3:
        tuning_status = "✅ Enabled" if results.get('hyperparameter_tuning') else "❌ Disabled"
        st.metric("Hyperparameter Tuning", tuning_status)
    
    with col4:
        ai_status = "✅ Enabled" if results.get('ai_assistance') else "❌ Disabled"
        st.metric("AI Assistance", ai_status)
    
    # Configuration Details
    with st.expander("🔧 View Detailed Configuration"):
        col1, col2 = st.columns(2)
        with col1:
            st.write("**Pipeline Settings:**")
            st.write(f"- Problem Type: {results.get('problem_type', 'N/A')}")
            st.write(f"- Target Column: {results.get('target_column', 'N/A')}")
            st.write(f"- Dataset: {results.get('dataset_name', 'N/A')}")
        
        with col2:
            st.write("**Advanced Features:**")
            st.write(f"- Hyperparameter Tuning: {results.get('hyperparameter_tuning', False)}")
            st.write(f"- AI Assistance: {results.get('ai_assistance', False)}")
            st.write(f"- Status: {results.get('status', 'N/A')}")

def main():
    initialize_session_state()
    
    # Main Header
    st.markdown('<div class="main-header">🚀 Smart Predictor - Universal Auto ML</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Fully Automated Machine Learning with Enhanced Analytics</div>', unsafe_allow_html=True)
    
    databricks_config = get_databricks_config()
    if not databricks_config:
        st.error("❌ Databricks configuration missing. Please check your secrets.toml file.")
        return
    
    # Sidebar - Enhanced Configuration
    with st.sidebar:
        st.markdown("## ⚡ Pipeline Configuration")
        
        st.markdown("### 🔧 Advanced Options")
        enable_tuning = st.checkbox(
            "Enable Hyperparameter Tuning", 
            value=st.session_state.pipeline_config['enable_tuning'],
            help="Better accuracy but slower execution"
        )
        
        use_ai_assist = st.checkbox(
            "Use AI for Target Detection & Insights", 
            value=st.session_state.pipeline_config['use_ai_assist'],
            help="Smart target identification and conversational insights"
        )
        
        # Update session state
        st.session_state.pipeline_config.update({
            'enable_tuning': enable_tuning,
            'use_ai_assist': use_ai_assist
        })
        
        st.markdown("---")
        st.markdown("## 📊 Features")
        
        st.markdown("""
        **🤖 Enhanced AutoML:**
        - Smart Target Detection
        - Enhanced EDA & Data Quality
        - Multi-Model Training
        - Optional Hyperparameter Tuning
        - AI-Powered Insights
        """)
        
        st.markdown("""
        **📈 Advanced Analytics:**
        - Interactive Visualizations
        - Feature Importance
        - Model Comparison
        - Business Insights
        - Performance Analysis
        """)
    
    # Main area with tabs
    tab1, tab2 = st.tabs(["🚀 Auto-ML Pipeline", "📊 Analytics Dashboard"])
    
    with tab1:
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("🎯 How It Works")
            st.info("""
            **1. Configure Pipeline**  
            Set your preferences using the sidebar options
            
            **2. Run Auto-ML**  
            Start the enhanced pipeline with one click
            
            **3. View Results**  
            Explore interactive analytics dashboard
            
            **4. Get Insights**  
            AI-powered explanations and recommendations
            """)
            
            st.markdown("**⚡ Smart Features:**")
            st.write("• Multi-layer target detection")
            st.write("• Enhanced data quality analysis") 
            st.write("• Optional hyperparameter tuning")
            st.write("• Conversational AI insights")
            st.markdown('</div>', unsafe_allow_html=True)
        
        with col2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.header("🚀 Start Enhanced Pipeline")
            
            if st.button("🎯 Run Auto-ML Pipeline", type="primary", use_container_width=True):
                run_auto_ml_pipeline()
            
            # Status display with emojis
            if st.session_state.job_status == 'running':
                st.info("🔄 Enhanced pipeline running...")
                st.write("• Smart target detection")
                st.write("• Enhanced EDA analysis") 
                st.write("• Model training with optional tuning")
                st.write("• AI insights generation")
            elif st.session_state.job_status == 'completed':
                st.success("✅ Pipeline completed successfully!")
                st.balloons()
            elif st.session_state.job_status == 'failed':
                st.error("❌ Pipeline execution failed.")
            st.markdown('</div>', unsafe_allow_html=True)
    
    with tab2:
        if st.session_state.auto_ml_results:
            display_enhanced_analytics_dashboard(st.session_state.auto_ml_results)
        else:
            st.info("👆 Run the Enhanced Auto-ML pipeline first to see the analytics dashboard!")
            
            # Show preview of what's coming
            col1, col2 = st.columns(2)
            with col1:
                st.markdown("### 📈 What to Expect:")
                st.write("• Interactive model performance charts")
                st.write("• Feature importance analysis")
                st.write("• Data quality insights")
                st.write("• AI-powered recommendations")
            
            with col2:
                st.markdown("### 🎯 Sample Output:")
                st.code("""
🤖 AI INSIGHTS:
"Your model achieved 89% accuracy - 
excellent for this problem type! 
Ready for production deployment."
                """)

if __name__ == "__main__":
    main()

import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration
st.set_page_config(
    page_title="Smart Predictor",
    page_icon="🤖",
    layout="wide"
)

# Main app
st.title("🤖 Smart Predictor - AI Assistant")
st.markdown("""
### Build Machine Learning Models in Minutes!

**Navigation Issue Detected:** Sidebar is hidden. Use the buttons below to navigate.
""")

# DIRECT NAVIGATION BUTTONS
st.markdown("---")
st.header("🚀 Quick Navigation")

col1, col2 = st.columns(2)

with col1:
    st.subheader("📊 Data Analysis")
    st.write("Upload and analyze your dataset with automatic insights")
    if st.button("📥 GO TO DATA ANALYSIS", use_container_width=True, type="primary"):
        # Switch to data analysis page
        st.session_state.current_page = "data_analysis"
        st.rerun()

with col2:
    st.subheader("🤖 Model Training")
    st.write("Train ML models with hyperparameter tuning")
    if st.button("⚡ GO TO MODEL TRAINING", use_container_width=True, type="primary"):
        # Switch to model training page  
        st.session_state.current_page = "model_training"
        st.rerun()

# Handle page switching
if st.session_state.get('current_page') == "data_analysis":
    # Import and run data analysis page
    from pages import Data_Analysis
    Data_Analysis.main()
elif st.session_state.get('current_page') == "model_training":
    # Import and run model training page
    from pages import Model_Training  
    Model_Training.main()

# File upload section (keep this for direct access)
st.markdown("---")
st.header("📁 Or Upload File Directly Here")

uploaded_file = st.file_uploader("Upload your CSV file:", type=["csv"])

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        st.session_state.current_dataset = df
        st.session_state.uploaded_file_name = uploaded_file.name
        
        st.success(f"✅ File uploaded successfully! {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Show quick preview
        st.subheader("📋 Data Preview")
        st.dataframe(df.head(10), use_container_width=True)
        
        # Show basic stats
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Total Rows", df.shape[0])
        with col2:
            st.metric("Total Columns", df.shape[1])
        with col3:
            st.metric("Missing Values", df.isnull().sum().sum())
            
        st.success("🎯 **Ready! Click 'GO TO DATA ANALYSIS' button above to explore your data!**")
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

# Show current status
st.markdown("---")
st.header("🔧 Current Status")

if 'current_dataset' in st.session_state:
    st.success("✅ Dataset loaded in memory and ready for analysis!")
    st.write(f"**File:** {st.session_state.uploaded_file_name}")
    st.write(f"**Shape:** {st.session_state.current_dataset.shape[0]} rows × {st.session_state.current_dataset.shape[1]} columns")
else:
    st.info("📝 No dataset loaded yet. Upload a CSV file above.")

st.markdown("---")
st.caption("Smart Predictor v1.0 | Streamlit + Databricks Integration")

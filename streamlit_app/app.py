import streamlit as st
import pandas as pd
import plotly.express as px

# Page configuration - THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Smart Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main app
st.title("🤖 Smart Predictor - AI Assistant")
st.markdown("""
### Build Machine Learning Models in Minutes!

This tool helps you:
- **Upload** your datasets (CSV files)
- **Analyze** data with automatic insights  
- **Train** multiple ML models
- **Compare** model performance
- **Tune** hyperparameters for better results

### 🚀 How to Use:
1. **Click "Data Analysis" in the sidebar on the left** ←
2. **Upload your CSV file** 
3. **Explore automatic data analysis**
4. **Go to Model Training** to build ML models
5. **View results** and predictions

---
**Supported Problems:** Binary Classification, Multi-class Classification, Regression
**Default Models:** Logistic Regression, Random Forest, XGBoost
""")

# Direct file upload as backup
st.markdown("---")
st.header("📁 Quick Start - Upload File Here")

uploaded_file = st.file_uploader("Upload your CSV file directly:", type="csv")

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
            
        st.info("🎯 **Next Step**: Go to **Data Analysis** page in sidebar for detailed analysis!")
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

# Debug information
with st.expander("🔧 Technical Information"):
    st.write("**App Status:** ✅ Running")
    st.write("**Pages Available:** 3 (Home, Data Analysis, Model Training)")
    st.write("**Navigation:** Use sidebar or upload file above")
    
    # Check session state
    if 'current_dataset' in st.session_state:
        st.success("✅ Dataset loaded in memory")
    else:
        st.info("📝 No dataset loaded yet")

st.markdown("---")
st.caption("Smart Predictor v1.0 | Streamlit + Databricks Integration")

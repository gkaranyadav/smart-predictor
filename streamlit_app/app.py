import streamlit as st
import pandas as pd

# THIS MUST BE THE FIRST STREAMLIT COMMAND
st.set_page_config(
    page_title="Smart Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"  # Force sidebar to open
)

# Main app
st.title("🤖 Smart Predictor - AI Assistant")
st.markdown("""
### Build Machine Learning Models in Minutes!

**If you don't see the sidebar on the left, please:**
1. **Refresh this page** (press F5 or Ctrl+R)
2. **Look for ☰ hamburger menu** in top-right corner
3. **Click it** to open the sidebar
4. **Click 'Data Analysis'** to upload and analyze your data

*The sidebar should automatically show: Home, Data Analysis, Model Training*
""")

# File upload as backup
st.markdown("---")
st.header("📁 Quick Start - Upload File Here")

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
            
        st.success("🎯 **Now go to 'Data Analysis' page in the sidebar to explore your data!**")
        
    except Exception as e:
        st.error(f"❌ Error reading file: {str(e)}")

# Show current status
st.markdown("---")
st.header("🔧 App Status")

if 'current_dataset' in st.session_state:
    st.success("✅ Dataset loaded in memory and ready for analysis!")
    df = st.session_state.current_dataset
    st.write(f"**File:** {st.session_state.uploaded_file_name}")
    st.write(f"**Shape:** {df.shape[0]} rows × {df.shape[1]} columns")
    st.write(f"**Columns:** {', '.join(df.columns.tolist()[:10])}{'...' if len(df.columns) > 10 else ''}")
else:
    st.info("📝 No dataset loaded yet. Upload a CSV file above.")

# Troubleshooting guide
with st.expander("🔧 Troubleshooting - If pages don't appear"):
    st.markdown("""
    **If sidebar doesn't show pages:**
    1. **Check your folder structure:**
       ```
       streamlit_app/
       ├── app.py
       ├── pages/
       │   ├── 1_🏠_Home.py
       │   ├── 2_📊_Data_Analysis.py
       │   └── 3_🤖_Model_Training.py
       └── requirements.txt
       ```
    
    2. **Make sure files are named correctly:**
       - `1_🏠_Home.py` (or `1_Home.py`)
       - `2_📊_Data_Analysis.py` (or `2_Data_Analysis.py`) 
       - `3_🤖_Model_Training.py` (or `3_Model_Training.py`)
    
    3. **Refresh the browser page completely** (Ctrl+F5)
    
    4. **Try a different browser** (Chrome, Firefox, Edge)
    """)

st.markdown("---")
st.caption("Smart Predictor v1.0 | If pages don't appear, refresh and check sidebar")

import streamlit as st

# Page configuration
st.set_page_config(
    page_title="Smart Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Main app
st.title("🤖 Smart Predictor - AI Assistant")
st.markdown("""
Welcome to **Smart Predictor**! This tool helps you build machine learning models 
for different datasets like Diabetes, Weather, Fraud Detection, and more.

### 🚀 Quick Start Guide:
1. **Go to Data Analysis page** (click in sidebar) 
2. **Upload your CSV file**
3. **Explore data insights**
4. **Train models** in Model Training page
5. **View results** and predictions

---
**Currently working with:** Diabetes Prediction, General Classification Problems
""")

# Direct navigation buttons since sidebar might not be working
st.markdown("---")
st.header("🎯 Get Started")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Data Analysis")
    st.write("Upload and analyze your dataset")
    if st.button("Start Data Analysis", type="primary", use_container_width=True):
        st.switch_page("pages/2_Data_Analysis.py")

with col2:
    st.subheader("🤖 Model Training") 
    st.write("Train ML models with hyperparameter tuning")
    if st.button("Train Models", type="secondary", use_container_width=True):
        st.switch_page("pages/3_Model_Training.py")

with col3:
    st.subheader("📈 Results")
    st.write("View model performance and predictions")
    if st.button("See Results", type="secondary", use_container_width=True):
        st.info("Train models first to see results!")

# Debug info
with st.expander("🔧 Debug Information"):
    st.write("Page files detected:")
    try:
        import os
        pages_dir = "pages"
        if os.path.exists(pages_dir):
            pages = os.listdir(pages_dir)
            st.write(pages)
        else:
            st.error("Pages directory not found!")
    except Exception as e:
        st.error(f"Error: {e}")

st.markdown("---")
st.info("💡 **Tip**: If sidebar navigation doesn't work, use the buttons above!")

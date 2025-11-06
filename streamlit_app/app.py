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
""")

# Sidebar
st.sidebar.success("Select a page above")

# Main content
col1, col2, col3 = st.columns(3)

with col1:
    st.header("📊 Data Analysis")
    st.write("Upload and analyze your dataset")
    st.button("Start Data Analysis", type="primary")

with col2:
    st.header("🤖 Model Training") 
    st.write("Train ML models with hyperparameter tuning")
    st.button("Train Models", type="secondary")

with col3:
    st.header("📈 Results")
    st.write("View model performance and predictions")
    st.button("See Results", type="secondary")

st.divider()
st.info("💡 **Tip**: Start with Data Analysis to understand your data first!")

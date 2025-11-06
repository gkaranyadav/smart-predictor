import streamlit as st

# Force sidebar to be visible
st.set_page_config(
    page_title="Smart Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"  # ← THIS FORCES SIDEBAR TO OPEN
)

st.title("🤖 Smart Predictor - AI Assistant")
st.markdown("""
### Build ML Models in Minutes!

Upload your data, train models, and get predictions - all in one tool.

**👇 Use the navigation buttons below (sidebar might be hidden)**
""")

# Direct navigation buttons as backup
st.markdown("---")
st.header("🚀 Quick Navigation")

col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📊 Data Analysis")
    st.write("Upload and analyze your dataset")
    if st.button("📥 Go to Data Analysis", use_container_width=True, type="primary"):
        st.switch_page("pages/2_Data_Analysis.py")

with col2:
    st.subheader("🤖 Model Training") 
    st.write("Train ML models with hyperparameter tuning")
    if st.button("⚡ Go to Model Training", use_container_width=True):
        st.switch_page("pages/3_Model_Training.py")

with col3:
    st.subheader("📈 Results")
    st.write("View model performance and predictions")
    if st.button("📊 Go to Results", use_container_width=True):
        st.info("Train models first to see results!")

# Sidebar check
with st.sidebar:
    st.title("🧭 Navigation")
    st.page_link("app.py", label="🏠 Home", icon="🏠")
    st.page_link("pages/2_Data_Analysis.py", label="📊 Data Analysis", icon="📊")
    st.page_link("pages/3_Model_Training.py", label="🤖 Model Training", icon="🤖")
    
    st.markdown("---")
    st.caption("Smart Predictor v1.0")

st.markdown("---")
st.info("💡 **Tip**: If you don't see the sidebar on the left, try refreshing the page!")

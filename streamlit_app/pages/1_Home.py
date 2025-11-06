import streamlit as st

st.set_page_config(
    page_title="Home - Smart Predictor",
    page_icon="🏠",
    layout="wide"
)

st.title("🏠 Smart Predictor - Home")
st.markdown("""
### Welcome to Your AI Assistant!

**Smart Predictor** helps you build machine learning models for various datasets.

### 🚀 Quick Start:
1. Go to **Data Analysis** page (in sidebar)
2. Upload your CSV file
3. Explore automatic data insights
4. Train models in **Model Training** page
5. View results and predictions

---
*Currently optimized for: Diabetes Prediction & General Classification*
""")

st.success("✅ Ready to start? Use the sidebar to navigate!")

import streamlit as st

st.set_page_config(
    page_title="Smart Predictor",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("🤖 Smart Predictor - AI Assistant")
st.markdown("""
### Build ML Models in Minutes!

**To get started:**

1. Look for the **sidebar on the left** ← 
2. Click **"Data Analysis"** in the sidebar
3. Upload your CSV file
4. Explore your data and train models

*If you don't see the sidebar, try refreshing the page or check the top-right corner for a ☰ menu icon.*
""")

# Simple instructions for sidebar
st.info("""
**💡 Can't see the sidebar?**
- Look for **☰ (hamburger menu)** in the top-right corner
- **Click it** to open the sidebar
- Then click **"Data Analysis"** to upload your file
""")

# Show what pages should be available
with st.expander("🔧 Debug Info"):
    st.write("Your app should have these pages in the sidebar:")
    st.write("- 🏠 Home (you are here)")
    st.write("- 📊 Data Analysis") 
    st.write("- 🤖 Model Training")
    st.write("If pages don't appear, try refreshing your browser.")

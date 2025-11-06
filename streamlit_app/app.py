import streamlit as st
import os

st.set_page_config(page_title="Debug", page_icon="🐛")

st.title("🔧 Debug - Check Your Files")

# Check what's in pages directory
st.header("📁 Files in pages folder:")
pages_dir = "pages"

if os.path.exists(pages_dir):
    files = os.listdir(pages_dir)
    st.write("Files found:", files)
    
    st.header("🚀 Try Navigation:")
    for file in files:
        if file.endswith('.py'):
            if st.button(f"Go to {file}"):
                st.switch_page(f"pages/{file}")
else:
    st.error("❌ 'pages' directory not found!")
    st.info("Make sure you have a 'pages' folder with your Python files")

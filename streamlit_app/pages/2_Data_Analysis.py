import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(
    page_title="Data Analysis - Smart Predictor", 
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Analysis")
st.markdown("Upload your dataset and get automated insights")

# File upload section
uploaded_file = st.file_uploader(
    "Choose a CSV file", 
    type="csv",
    help="Upload your dataset for analysis"
)

if uploaded_file is not None:
    try:
        # Load data
        df = pd.read_csv(uploaded_file)
        
        # Store in session state for other pages
        st.session_state.current_dataset = df
        st.session_state.uploaded_file_name = uploaded_file.name
        
        # Basic info
        st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Tabs for different analyses
        tab1, tab2, tab3 = st.tabs(["📋 Overview", "📈 Distributions", "🔍 Correlations"])
        
        with tab1:
            st.subheader("Dataset Overview")
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**First 10 Rows:**")
                st.dataframe(df.head(10), use_container_width=True)
            
            with col2:
                st.write("**Dataset Info:**")
                st.json({
                    "Total Rows": df.shape[0],
                    "Total Columns": df.shape[1],
                    "Missing Values": df.isnull().sum().sum(),
                    "Duplicate Rows": df.duplicated().sum()
                })
        
        with tab2:
            st.subheader("Feature Distributions")
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select a numeric column:", numeric_cols)
                fig_hist = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                st.plotly_chart(fig_hist, use_container_width=True)
            else:
                st.info("No numeric columns found")
        
        with tab3:
            st.subheader("Correlation Analysis")
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            if numeric_df.shape[1] > 1:
                corr_matrix = numeric_df.corr()
                fig_corr = px.imshow(corr_matrix, title="Feature Correlation Heatmap")
                st.plotly_chart(fig_corr, use_container_width=True)
            else:
                st.info("Need at least 2 numeric columns for correlation analysis")
                
        # Success message for next steps
        st.success("🎯 Dataset ready! Go to **Model Training** page to build ML models.")
        
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
else:
    st.info("👆 Please upload a CSV file to begin analysis")

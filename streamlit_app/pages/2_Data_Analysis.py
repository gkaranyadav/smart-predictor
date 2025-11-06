import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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
        
        # Basic info
        st.success(f"✅ Dataset loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # Create tabs for different analyses
        tab1, tab2, tab3, tab4 = st.tabs(["📋 Overview", "📈 Distributions", "🔍 Correlations", "🎯 Target Analysis"])
        
        with tab1:
            st.subheader("Dataset Overview")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("**First 10 Rows:**")
                st.dataframe(df.head(10), use_container_width=True)
            
            with col2:
                st.write("**Dataset Information:**")
                st.json({
                    "Total Rows": df.shape[0],
                    "Total Columns": df.shape[1],
                    "Missing Values": df.isnull().sum().sum(),
                    "Duplicate Rows": df.duplicated().sum()
                })
                
                st.write("**Column Types:**")
                dtype_counts = df.dtypes.value_counts()
                for dtype, count in dtype_counts.items():
                    st.write(f"- {dtype}: {count} columns")
        
        with tab2:
            st.subheader("Feature Distributions")
            
            # Select column to visualize
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            if numeric_cols:
                selected_col = st.selectbox("Select a numeric column:", numeric_cols)
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Histogram
                    fig_hist = px.histogram(df, x=selected_col, title=f"Distribution of {selected_col}")
                    st.plotly_chart(fig_hist, use_container_width=True)
                
                with col2:
                    # Box plot
                    fig_box = px.box(df, y=selected_col, title=f"Box Plot of {selected_col}")
                    st.plotly_chart(fig_box, use_container_width=True)
            else:
                st.info("No numeric columns found for distribution analysis")
        
        with tab3:
            st.subheader("Correlation Analysis")
            
            # Numeric correlations
            numeric_df = df.select_dtypes(include=['int64', 'float64'])
            if numeric_df.shape[1] > 1:
                corr_matrix = numeric_df.corr()
                
                fig_corr = px.imshow(
                    corr_matrix,
                    title="Feature Correlation Heatmap",
                    color_continuous_scale="RdBu_r",
                    aspect="auto"
                )
                st.plotly_chart(fig_corr, use_container_width=True)
                
                # Show top correlations
                st.write("**Top Correlations:**")
                corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
                # Remove self-correlations and duplicates
                corr_pairs = corr_pairs[corr_pairs < 0.99]
                top_corrs = corr_pairs.head(10)
                
                for (col1, col2), value in top_corrs.items():
                    st.write(f"`{col1}` ↔ `{col2}`: {value:.3f}")
            else:
                st.info("Need at least 2 numeric columns for correlation analysis")
        
        with tab4:
            st.subheader("Target Variable Analysis")
            
            # Auto-detect potential target columns
            potential_targets = []
            for col in df.columns:
                if df[col].nunique() <= 10:  # Low cardinality
                    potential_targets.append(col)
            
            if potential_targets:
                target_col = st.selectbox(
                    "Select target variable for analysis:",
                    potential_targets,
                    help="Choose the column you want to predict"
                )
                
                if target_col:
                    st.write(f"**Target Distribution:**")
                    target_counts = df[target_col].value_counts()
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        fig_pie = px.pie(
                            values=target_counts.values,
                            names=target_counts.index,
                            title=f"Distribution of {target_col}"
                        )
                        st.plotly_chart(fig_pie, use_container_width=True)
                    
                    with col2:
                        st.dataframe(target_counts, use_container_width=True)
                        
                        # Basic target stats
                        st.write("**Target Statistics:**")
                        st.json({
                            "Unique Values": df[target_col].nunique(),
                            "Most Frequent": target_counts.index[0],
                            "Balance Ratio": f"{(target_counts.max() / target_counts.sum() * 100):.1f}%"
                        })
            else:
                st.info("No obvious target columns detected (looking for low cardinality columns)")
    
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        
    # STORE DATASET IN SESSION STATE FOR MODEL TRAINING PAGE
    if uploaded_file is not None and 'df' in locals():
        st.session_state.current_dataset = df
        st.session_state.uploaded_file_name = uploaded_file.name
        
        # Auto-detect potential target
        potential_targets = []
        for col in df.columns:
            if df[col].nunique() <= 10:  # Low cardinality
                potential_targets.append(col)
        
        if potential_targets:
            st.session_state.target_column = potential_targets[0]
        
        st.success("✅ Dataset ready for model training! Go to **Model Training** page.")
        
else:
    st.info("👆 Please upload a CSV file to begin analysis")

# Footer
st.markdown("---")
st.caption("💡 **Tip**: Look for patterns, outliers, and potential target variables in your data!")

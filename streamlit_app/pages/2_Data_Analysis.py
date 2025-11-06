import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np

st.set_page_config(
    page_title="Data Analysis - Smart Predictor", 
    page_icon="📊",
    layout="wide"
)

st.title("📊 Data Analysis")
st.markdown("Upload your dataset and get automated insights and visualizations")

# Check if data already exists from main page
if 'current_dataset' in st.session_state and st.session_state.current_dataset is not None:
    df = st.session_state.current_dataset
    st.success(f"✅ Using existing dataset: {df.shape[0]} rows, {df.shape[1]} columns")
else:
    # File upload section
    uploaded_file = st.file_uploader(
        "Choose a CSV file", 
        type="csv",
        help="Upload your dataset for analysis"
    )
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.session_state.current_dataset = df
            st.session_state.uploaded_file_name = uploaded_file.name
            st.success(f"✅ Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        except Exception as e:
            st.error(f"Error loading file: {str(e)}")
            st.stop()
    else:
        st.info("👆 Please upload a CSV file to begin analysis")
        st.stop()

# MAIN ANALYSIS SECTION
try:
    # Create tabs for different analyses
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📋 Dataset Overview", "📈 Distributions", "🔍 Correlations", "🎯 Target Analysis", "⚙️ Data Quality"])
    
    with tab1:
        st.header("Dataset Overview")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.subheader("First 10 Rows")
            st.dataframe(df.head(10), use_container_width=True, height=400)
        
        with col2:
            st.subheader("Dataset Information")
            info_data = {
                "Shape": f"{df.shape[0]} rows × {df.shape[1]} columns",
                "Memory Usage": f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB",
                "Missing Values": f"{df.isnull().sum().sum()} total",
                "Duplicate Rows": f"{df.duplicated().sum()}",
                "Total Size": f"{(df.memory_usage(deep=True).sum() / 1024**2):.2f} MB"
            }
            
            for key, value in info_data.items():
                st.metric(key, value)
            
            st.subheader("Column Types")
            dtype_counts = df.dtypes.value_counts()
            for dtype, count in dtype_counts.items():
                st.write(f"• **{dtype}**: {count} columns")
    
    with tab2:
        st.header("Feature Distributions")
        
        # Select column to visualize
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
        
        if numeric_cols:
            selected_col = st.selectbox("Select a numeric column:", numeric_cols)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Histogram
                fig_hist = px.histogram(
                    df, 
                    x=selected_col, 
                    title=f"Distribution of {selected_col}",
                    nbins=50
                )
                fig_hist.update_layout(showlegend=False)
                st.plotly_chart(fig_hist, use_container_width=True)
            
            with col2:
                # Box plot
                fig_box = px.box(
                    df, 
                    y=selected_col, 
                    title=f"Box Plot of {selected_col}"
                )
                st.plotly_chart(fig_box, use_container_width=True)
                
                # Statistics
                st.subheader("Statistics")
                col_stats1, col_stats2 = st.columns(2)
                with col_stats1:
                    st.metric("Mean", f"{df[selected_col].mean():.2f}")
                    st.metric("Median", f"{df[selected_col].median():.2f}")
                with col_stats2:
                    st.metric("Std Dev", f"{df[selected_col].std():.2f}")
                    st.metric("Missing", f"{df[selected_col].isnull().sum()}")
        else:
            st.info("No numeric columns found for distribution analysis")
            
        # Categorical analysis
        if categorical_cols:
            st.subheader("Categorical Features")
            cat_col = st.selectbox("Select a categorical column:", categorical_cols)
            
            value_counts = df[cat_col].value_counts()
            fig_bar = px.bar(
                x=value_counts.index, 
                y=value_counts.values,
                title=f"Distribution of {cat_col}",
                labels={'x': cat_col, 'y': 'Count'}
            )
            st.plotly_chart(fig_bar, use_container_width=True)
    
    with tab3:
        st.header("Correlation Analysis")
        
        # Numeric correlations
        numeric_df = df.select_dtypes(include=[np.number])
        if numeric_df.shape[1] > 1:
            corr_matrix = numeric_df.corr()
            
            fig_corr = px.imshow(
                corr_matrix,
                title="Feature Correlation Heatmap",
                color_continuous_scale="RdBu_r",
                aspect="auto",
                text_auto=True
            )
            fig_corr.update_layout(height=600)
            st.plotly_chart(fig_corr, use_container_width=True)
            
            # Show top correlations
            st.subheader("Top Correlations")
            corr_pairs = corr_matrix.unstack().sort_values(ascending=False)
            # Remove self-correlations and duplicates
            corr_pairs = corr_pairs[corr_pairs < 0.99]
            top_corrs = corr_pairs.head(10)
            
            for (col1, col2), value in top_corrs.items():
                st.write(f"`{col1}` ↔ `{col2}`: **{value:.3f}**")
        else:
            st.info("Need at least 2 numeric columns for correlation analysis")
    
    with tab4:
        st.header("Target Variable Analysis")
        
        # Auto-detect potential target columns
        potential_targets = []
        for col in df.columns:
            if df[col].nunique() <= 20:  # Reasonable cardinality for target
                potential_targets.append(col)
        
        if potential_targets:
            target_col = st.selectbox(
                "Select target variable for analysis:",
                potential_targets,
                help="Choose the column you want to predict"
            )
            
            if target_col:
                st.subheader(f"Target: {target_col}")
                target_counts = df[target_col].value_counts()
                
                col1, col2 = st.columns(2)
                
                with col1:
                    # Pie chart
                    fig_pie = px.pie(
                        values=target_counts.values,
                        names=target_counts.index,
                        title=f"Distribution of {target_col}"
                    )
                    st.plotly_chart(fig_pie, use_container_width=True)
                
                with col2:
                    # Bar chart
                    fig_bar = px.bar(
                        x=target_counts.index,
                        y=target_counts.values,
                        title=f"Count of {target_col}",
                        labels={'x': target_col, 'y': 'Count'}
                    )
                    st.plotly_chart(fig_bar, use_container_width=True)
                    
                    # Target statistics
                    st.subheader("Target Statistics")
                    balance_ratio = (target_counts.max() / target_counts.sum() * 100)
                    
                    st.metric("Unique Values", df[target_col].nunique())
                    st.metric("Most Frequent", f"{target_counts.index[0]} ({target_counts.iloc[0]})")
                    st.metric("Balance", f"{balance_ratio:.1f}%")
                    
                    # Store target for model training
                    st.session_state.target_column = target_col
        else:
            st.info("No obvious target columns detected (looking for columns with ≤20 unique values)")
    
    with tab5:
        st.header("Data Quality Report")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Missing Values")
            missing_data = df.isnull().sum()
            missing_data = missing_data[missing_data > 0]
            
            if len(missing_data) > 0:
                fig_missing = px.bar(
                    x=missing_data.index,
                    y=missing_data.values,
                    title="Missing Values by Column",
                    labels={'x': 'Column', 'y': 'Missing Count'}
                )
                st.plotly_chart(fig_missing, use_container_width=True)
            else:
                st.success("✅ No missing values found!")
        
        with col2:
            st.subheader("Data Types")
            dtype_summary = df.dtypes.value_counts()
            fig_dtypes = px.pie(
                values=dtype_summary.values,
                names=[str(dtype) for dtype in dtype_summary.index],
                title="Data Type Distribution"
            )
            st.plotly_chart(fig_dtypes, use_container_width=True)
            
        # Data quality metrics
        st.subheader("Quality Metrics")
        quality_col1, quality_col2, quality_col3, quality_col4 = st.columns(4)
        
        with quality_col1:
            st.metric("Complete Columns", f"{df.shape[1] - (df.isnull().sum() > 0).sum()}/{df.shape[1]}")
        with quality_col2:
            st.metric("Duplicate Rows", df.duplicated().sum())
        with quality_col3:
            st.metric("Total Missing", df.isnull().sum().sum())
        with quality_col4:
            st.metric("Memory", f"{(df.memory_usage(deep=True).sum() / 1024**2):.2f} MB")

except Exception as e:
    st.error(f"Error in analysis: {str(e)}")

# Success message for next steps
st.success("🎯 **Dataset analysis complete! Go to 'Model Training' page to build machine learning models.**")

# Footer
st.markdown("---")
st.caption("💡 **Tip**: Look for patterns, outliers, and choose a good target variable for prediction!")

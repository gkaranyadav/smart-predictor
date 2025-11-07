# streamlit_app.py
import streamlit as st
import pandas as pd
import os
import tempfile
import json
import time
import matplotlib.pyplot as plt
import seaborn as sns

from dbx_utils import upload_file_to_dbfs, run_job_now, DATABRICKS_HOST, DATABRICKS_TOKEN

st.set_page_config(layout="wide", page_title="Smart Predict")

st.title("Smart Predict — Upload CSV, train on Databricks, get predictions")

st.sidebar.header("Databricks & ML settings")
st.sidebar.markdown("Make sure the following environment vars are set: `DATABRICKS_HOST`, `DATABRICKS_TOKEN`.")

upload = st.file_uploader("Upload CSV file", type=["csv"])
if not (DATABRICKS_HOST and DATABRICKS_TOKEN):
    st.sidebar.warning("Please set DATABRICKS_HOST and DATABRICKS_TOKEN in Streamlit Cloud secrets.")
    st.stop()

if upload is not None:
    # Save locally to temp file so we can upload to DBFS
    with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
        tmp.write(upload.getvalue())
        tmp_path = tmp.name

    df = pd.read_csv(tmp_path)
    st.subheader("Data preview")
    st.dataframe(df.head(200))

    st.subheader("Quick EDA")
    c1, c2 = st.columns([1,1])
    with c1:
        st.write("Shape:", df.shape)
        st.write("Missing values:")
        st.write(df.isnull().sum().sort_values(ascending=False).head(20))
        st.write("Dtypes:")
        st.write(df.dtypes)
    with c2:
        if df.select_dtypes("number").shape[1] > 0:
            st.write("Numeric describe:")
            st.write(df.describe().T)
        # Corr heatmap (sample to avoid very large)
        st.write("Correlation (sampled to 5000 rows)")
        sample = df.sample(min(len(df), 2000), random_state=42)
        corr = sample.select_dtypes("number").corr()
        fig, ax = plt.subplots(figsize=(6,6))
        sns.heatmap(corr, ax=ax)
        st.pyplot(fig)

    st.markdown("---")
    st.header("Upload to Databricks & run pipeline")

    dbfs_target = st.text_input("DBFS destination path (e.g. /FileStore/yourname/data/input.csv)", "/FileStore/shared_uploads/user@domain.com/input.csv")
    if st.button("Upload to DBFS & run ingest job"):
        try:
            st.info("Uploading file to DBFS...")
            # use upload helper
            upload_file_to_dbfs(tmp_path, dbfs_target)
            st.success(f"Uploaded to {dbfs_target}")

            # Trigger ingest_to_delta job (keeps same job name)
            st.info("Triggering Databricks job: ingest_to_delta")
            resp = run_job_now("ingest_to_delta", notebook_params={"dbfs_path": dbfs_target})
            st.write("Job response:", resp)
            st.success("ingest_to_delta triggered. You can check job run in Databricks UI.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.header("Train model on Databricks")

    target_override = st.text_input("Override target column (optional)", "")
    model_type = st.selectbox("Force model type (optional)", ["auto", "classification", "regression"])

    if st.button("Trigger train_model job"):
        try:
            params = {"delta_table_path": "", "target_col": target_override or "", "model_type": model_type}
            # If ingest created the delta path and returns it as job output, you'd normally read the run output.
            # For simplicity, pass the dbfs path if you saved Delta at a known location in ingest job.
            st.info("Triggering Databricks job: train_model")
            resp = run_job_now("train_model", notebook_params=params)
            st.write("train_model response:", resp)
            st.success("train_model started. Check MLflow UI for runs.")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.header("Score new data (batch_score on Databricks)")

    if st.button("Trigger batch_score job and fetch predictions"):
        try:
            # pass parameters as needed; use the same dbfs_target for scoring sample
            params = {"input_dbfs_path": dbfs_target}
            st.info("Triggering Databricks job: batch_score")
            resp = run_job_now("batch_score", notebook_params=params)
            st.write("batch_score response:", resp)
            st.success("batch_score started; predictions will be in job output (Databricks).")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("**Notes**: This Streamlit app triggers your Databricks jobs by name. Make sure jobs `ingest_to_delta`, `train_model`, and `batch_score` exist in your workspace, or create them with the attached scripts. If you want streaming back predictions into Streamlit, we can add a small polling logic to fetch run results after completion.")

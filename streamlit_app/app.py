# ===============================================================
# 🚀 Streamlit + Databricks AutoML Pipeline UI
# Upload CSV → DBFS → Run Unified Job → Show Insights
# ===============================================================

import streamlit as st
import pandas as pd
import requests
import io
import time
import json
import base64
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# ---------------------------------------------------------------
# 1️⃣ Databricks Config — Replace with your details
# ---------------------------------------------------------------

DATABRICKS_HOST = st.secrets["DATABRICKS_HOST"]
DATABRICKS_TOKEN = st.secrets["DATABRICKS_TOKEN"]
DATABRICKS_JOB_INGEST_ID = st.secrets["DATABRICKS_JOB_INGEST_ID"]
DATABRICKS_JOB_TRAIN_ID = st.secrets["DATABRICKS_JOB_TRAIN_ID"]
DATABRICKS_JOB_SCORE_ID = st.secrets["DATABRICKS_JOB_SCORE_ID"]

# ---------------------------------------------------------------
# 2️⃣ Helper Functions
# ---------------------------------------------------------------
def upload_to_dbfs(file, dbfs_path):
    """Upload file to Databricks DBFS"""
    url = f"{DATABRICKS_HOST}/api/2.0/dbfs/put"
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}
    data = {"path": dbfs_path, "overwrite": "true"}
    files = {"contents": file.getvalue()}
    response = requests.post(url, headers=headers, data=data, files=files)
    if response.status_code == 200:
        st.success(f"✅ Uploaded to DBFS: {dbfs_path}")
        return True
    else:
        st.error(f"❌ Upload failed: {response.text}")
        return False


def run_databricks_job(job_id):
    """Trigger Databricks job by job_id"""
    url = f"{DATABRICKS_HOST}/api/2.1/jobs/run-now"
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}
    data = {"job_id": job_id}
    response = requests.post(url, headers=headers, json=data)
    if response.status_code != 200:
        st.error(f"❌ Failed to start job: {response.text}")
        return None
    run_id = response.json().get("run_id")
    st.info(f"🚀 Job started (Run ID: {run_id})")
    return run_id


def get_job_status(run_id):
    """Poll job status"""
    url = f"{DATABRICKS_HOST}/api/2.1/jobs/runs/get?run_id={run_id}"
    headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}
    while True:
        time.sleep(8)
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            st.error("❌ Error checking job status")
            return
        state = response.json().get("state", {}).get("life_cycle_state", "")
        result = response.json().get("state", {}).get("result_state", "")
        st.write(f"⏳ Job Status: {state} ({result})")
        if state == "TERMINATED":
            st.success("✅ Job completed successfully!")
            break


def show_data_summary(uploaded_file):
    """Show summary and correlation matrix"""
    df = pd.read_csv(uploaded_file)
    st.subheader("📊 Dataset Summary")
    st.write(df.describe())
    st.write("🔢 Shape:", df.shape)
    st.write("📋 Columns:", list(df.columns))

    st.subheader("🧩 Correlation Matrix")
    corr = df.corr(numeric_only=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(corr, cmap="coolwarm", annot=False, ax=ax)
    st.pyplot(fig)


# ---------------------------------------------------------------
# 3️⃣ Streamlit UI
# ---------------------------------------------------------------
st.set_page_config(page_title="AutoML Pipeline", layout="wide")
st.title("🚀 AutoML Pipeline (Streamlit + Databricks + MLflow)")

uploaded_file = st.file_uploader("📤 Upload CSV", type=["csv"])

if uploaded_file is not None:
    st.write(f"File: {uploaded_file.name} ({round(len(uploaded_file.getvalue())/1e6,2)} MB)")

    dbfs_path = f"/FileStore/shared_uploads/{uploaded_file.name}"
    st.write(f"Uploading to {dbfs_path} ...")

    if upload_to_dbfs(uploaded_file, dbfs_path):
        # ✅ Step 1: Run Databricks Unified Job
        st.subheader("⚙️ Running Unified Job on Databricks...")
        run_id = run_databricks_job(DATABRICKS_JOB_INGEST_ID)

        if run_id:
            get_job_status(run_id)

        # ✅ Step 2: Show Data Summary
        show_data_summary(uploaded_file)

import streamlit as st
import pandas as pd
import base64
import requests
import json
import io
import time

# Read Databricks credentials
DATABRICKS_HOST = st.secrets["DATABRICKS_HOST"]
DATABRICKS_TOKEN = st.secrets["DATABRICKS_TOKEN"]

headers = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}


# ---------- Helper: Upload to DBFS ----------
def upload_to_dbfs(file, dbfs_path):
    content = file.read()
    b64_content = base64.b64encode(content).decode("utf-8")

    url = f"{DATABRICKS_HOST}/api/2.0/dbfs/put"
    data = {"path": dbfs_path, "contents": b64_content, "overwrite": True}
    resp = requests.post(url, headers=headers, json=data)
    if resp.status_code == 200:
        return True
    else:
        st.error(f"DBFS upload failed: {resp.text}")
        return False


# ---------- Helper: Run Databricks Job ----------
def run_job(job_name, params=None):
    url = f"{DATABRICKS_HOST}/api/2.1/jobs/list"
    jobs = requests.get(url, headers=headers).json()
    job_id = None
    for job in jobs.get("jobs", []):
        if job["settings"]["name"] == job_name:
            job_id = job["job_id"]
            break
    if not job_id:
        st.error(f"Job '{job_name}' not found!")
        return None

    run_url = f"{DATABRICKS_HOST}/api/2.1/jobs/run-now"
    payload = {"job_id": job_id, "notebook_params": params or {}}
    run = requests.post(run_url, headers=headers, json=payload).json()
    return run.get("run_id")


# ---------- Streamlit UI ----------
st.title("🚀 AutoML Pipeline (Streamlit + Databricks + MLflow)")

uploaded = st.file_uploader("Upload CSV", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    st.dataframe(df.head())

    dbfs_path = f"/FileStore/shared_uploads/{uploaded.name}"
    st.info(f"Uploading to {dbfs_path} ...")
    if upload_to_dbfs(uploaded, dbfs_path):
        st.success("✅ Uploaded to DBFS!")

        st.info("Running ingestion job...")
        run_id = run_job("ingest_to_delta", {"input_path": dbfs_path})
        if run_id:
            st.success(f"Ingestion started (Run ID: {run_id})")

        st.info("Running model training job...")
        run_id = run_job("train_model", {"input_table": "default.input_data"})
        if run_id:
            st.success(f"Training started (Run ID: {run_id})")

        st.info("Running batch scoring job...")
        run_id = run_job("batch_score", {"input_path": dbfs_path})
        if run_id:
            st.success(f"Prediction started (Run ID: {run_id})")

        # Show simple EDA
        st.subheader("📊 Dataset Summary")
        st.write(df.describe())
        st.write("🧩 Correlation Matrix:")
        st.dataframe(df.corr())


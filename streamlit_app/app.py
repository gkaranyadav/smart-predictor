import streamlit as st
import pandas as pd
from databricks_api import dbfs_put_single, dbfs_upload_chunked, run_job
from utils import gen_session_id, safe_dbfs_path

# Job IDs from secrets
INGEST_JOB_ID = st.secrets["DATABRICKS_JOB_INGEST_ID"]
TRAIN_JOB_ID = st.secrets["DATABRICKS_JOB_TRAIN_ID"]
SCORE_JOB_ID = st.secrets["DATABRICKS_JOB_SCORE_ID"]

st.title("Smart Predictor")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])
if uploaded_file:
    session_id = gen_session_id()
    dbfs_path = f"/FileStore/tmp/{session_id}/{uploaded_file.name}"

    # Decide chunked or single
    uploaded_file.seek(0)
    size = uploaded_file.size if hasattr(uploaded_file, "size") else len(uploaded_file.getvalue())
    if size > 2*1024*1024:
        dbfs_upload_chunked(dbfs_path, uploaded_file, overwrite=True)
    else:
        dbfs_put_single(dbfs_path, uploaded_file, overwrite=True)

    st.success(f"File uploaded to DBFS: {dbfs_path}")

    if st.button("Run Ingest Job"):
        res = run_job(INGEST_JOB_ID, {"dbfs_path": dbfs_path, "session_id": session_id})
        st.success(f"Ingest job finished: {res}")

    if st.button("Run Train Job"):
        res = run_job(TRAIN_JOB_ID, {"session_id": session_id})
        st.success(f"Training finished: {res}")

    if st.button("Run Score Job"):
        res = run_job(SCORE_JOB_ID, {"input_dbfs_path": dbfs_path, "session_id": session_id})
        st.success(f"Scoring finished: {res}")

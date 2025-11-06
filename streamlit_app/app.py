# app.py
import streamlit as st
import pandas as pd
import time
import io
from databricks_api import dbfs_put_single, dbfs_upload_chunked, run_job, get_run, download_dbfs_file
from utils import gen_session_id, safe_dbfs_path

st.set_page_config(page_title="Smart Predictor", layout="wide")
st.title("Smart Predictor — Streamlit → Databricks (Batch scoring)")

# --- Config ---
DATABRICKS_UPLOAD_CHUNK_THRESHOLD = 5 * 1024 * 1024  # 5 MB: use chunked above this

# --- Secrets (set these in Streamlit Secrets) ---
DATABRICKS_JOB_INGEST_ID = int(st.secrets["DATABRICKS_JOB_INGEST_ID"])
DATABRICKS_JOB_TRAIN_ID = int(st.secrets["DATABRICKS_JOB_TRAIN_ID"])
DATABRICKS_JOB_SCORE_ID = int(st.secrets["DATABRICKS_JOB_SCORE_ID"])

# --- Upload section ---
uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
if uploaded_file:
    session_id = gen_session_id()
    st.session_state["session_id"] = session_id
    filename = uploaded_file.name
    dbfs_path = safe_dbfs_path(session_id, filename)
    st.write("Uploading to Databricks DBFS...", dbfs_path)

    # Upload using chunk or single based on size
    uploaded_file.seek(0, io.SEEK_END)
    size = uploaded_file.tell()
    uploaded_file.seek(0)
    if size > DATABRICKS_UPLOAD_CHUNK_THRESHOLD:
        with st.spinner("Large file detected — uploading in chunks..."):
            dbfs_upload_chunked(dbfs_path, uploaded_file, overwrite=True)
    else:
        with st.spinner("Uploading..."):
            dbfs_put_single(dbfs_path, uploaded_file, overwrite=True)
    st.success("Upload complete. Triggering ingestion job to convert to Delta...")

    # Trigger the ingestion job (Databricks job should accept 'dbfs_path' and 'dataset_id')
    notebook_params = {"dbfs_path": dbfs_path, "session_id": session_id}
    run_id = run_job(DATABRICKS_JOB_INGEST_ID, notebook_params)
    st.info(f"Ingest job started (run_id: {run_id}). Polling status...")

    # Poll job status
    while True:
        run = get_run(run_id)
        state = run.get("state", {}).get("life_cycle_state")
        result_state = run.get("state", {}).get("result_state")
        st.write(f"Run state: {state} | Result: {result_state}")
        if state in ["TERMINATED", "INTERNAL_ERROR", "SKIPPED"]:
            break
        time.sleep(5)

    if run.get("state", {}).get("result_state") == "SUCCESS":
        st.success("Ingestion complete — Delta table created for this dataset.")
        st.session_state["delta_path"] = run.get("metadata", {}).get("spark_context", {}).get("dbfs_root", "") or ""
    else:
        st.error("Ingestion failed. Check Databricks job logs.")

# --- EDA & Train UI (only enabled if dataset ingested) ---
if "session_id" in st.session_state:
    st.markdown("---")
    st.subheader("Dataset actions")
    if st.button("Run Quick EDA (sampled)"):
        st.info("Triggering EDA notebook/job...")
        # For simplicity reuse the training job or a dedicated EDA job id (not in this example).
        # You can create a job that creates a sample parquet and writes a sample CSV to DBFS,
        # then Streamlit downloads the sample and displays charts.
        st.success("EDA triggered — once job completes a sample will be displayed from DBFS.")
    st.write("After training, you can click 'Save dataset' to persist this Delta for more than 4 hours.")
    if st.button("Save dataset (promote to /mnt/delta/)"):
        # call a small job or notebook to promote from tmp to permanent delta path
        st.info("Promote job triggered — dataset will be persistent.")
        promote_params = {"session_id": st.session_state["session_id"], "action": "promote"}
        run_id = run_job(DATABRICKS_JOB_INGEST_ID, promote_params)
        st.write("Promote job started:", run_id)

    st.markdown("---")
    st.subheader("Training")
    if st.button("Train model on this dataset"):
        st.info("Training job started...")
        train_params = {"session_id": st.session_state["session_id"]}
        run_id = run_job(DATABRICKS_JOB_TRAIN_ID, train_params)
        st.write("Train job run id:", run_id)
        # optionally poll or direct user to Databricks UI for logs

    st.markdown("---")
    st.subheader("Batch Scoring")
    new_file = st.file_uploader("Upload CSV to score (same schema)", type=["csv"], key="score_upload")
    if new_file and st.button("Run batch scoring"):
        score_session = gen_session_id()
        score_dbfs_path = safe_dbfs_path(score_session, new_file.name)
        # upload
        new_file.seek(0, io.SEEK_END)
        size = new_file.tell()
        new_file.seek(0)
        if size > DATABRICKS_UPLOAD_CHUNK_THRESHOLD:
            dbfs_upload_chunked(score_dbfs_path, new_file, overwrite=True)
        else:
            dbfs_put_single(score_dbfs_path, new_file, overwrite=True)

        # trigger scoring job (scoring job reads best model from Model Registry)
        job_params = {"input_dbfs_path": score_dbfs_path, "session_id": score_session}
        run_id = run_job(DATABRICKS_JOB_SCORE_ID, job_params)
        st.info(f"Scoring job started: {run_id}. Polling for result...")

        # simple poll
        while True:
            run = get_run(run_id)
            state = run.get("state", {}).get("life_cycle_state")
            result_state = run.get("state", {}).get("result_state")
            st.write(f"Run state: {state} | Result: {result_state}")
            if state in ["TERMINATED", "INTERNAL_ERROR", "SKIPPED"]:
                break
            time.sleep(5)

        if run.get("state", {}).get("result_state") == "SUCCESS":
            # expected output path (set in scoring notebook)
            output_path = f"/FileStore/results/{score_session}/preds.csv"
            local_path = f"/tmp/preds_{score_session}.csv"
            download_dbfs_file(output_path, local_path)
            df = pd.read_csv(local_path)
            st.subheader("Predictions")
            st.dataframe(df.head(200))
            st.download_button("Download predictions CSV", data=open(local_path,"rb"), file_name="preds.csv")
        else:
            st.error("Scoring job failed. Check Databricks logs.")

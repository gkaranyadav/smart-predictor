import streamlit as st
import requests
import base64
import time

# -------------------------------
# Read Databricks secrets
# -------------------------------
DATABRICKS_HOST = st.secrets["DATABRICKS_HOST"]
DATABRICKS_TOKEN = st.secrets["DATABRICKS_TOKEN"]

HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}

# -------------------------------
# 1️⃣ Upload small file to DBFS
# -------------------------------
def dbfs_put_single(path, file_obj, overwrite=False):
    """
    Upload small files (<2MB) to DBFS in a single request
    """
    content = file_obj.read()
    if isinstance(content, str):
        content = content.encode("utf-8")

    content_b64 = base64.b64encode(content).decode("utf-8")

    url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.0/dbfs/put"
    payload = {
        "path": path,
        "overwrite": overwrite,
        "contents": content_b64
    }

    r = requests.post(url, json=payload, headers=HEADERS)
    r.raise_for_status()
    return r.json()

# -------------------------------
# 2️⃣ Upload large file in chunks
# -------------------------------
def dbfs_upload_chunked(path, file_obj, overwrite=False, chunk_size=2*1024*1024):
    """
    Upload large files (>2MB) to DBFS using create -> add-block -> close
    chunk_size default: 2MB
    """
    base = f"{DATABRICKS_HOST.rstrip('/')}/api/2.0/dbfs"
    
    # 1) Create file
    create_url = base + "/create"
    create_payload = {"path": path, "overwrite": overwrite}
    r = requests.post(create_url, json=create_payload, headers=HEADERS)
    r.raise_for_status()

    # 2) Add blocks
    file_obj.seek(0)
    while True:
        chunk = file_obj.read(chunk_size)
        if not chunk:
            break
        if isinstance(chunk, str):
            chunk = chunk.encode("utf-8")
        chunk_b64 = base64.b64encode(chunk).decode("utf-8")
        add_block_url = base + "/add-block"
        add_payload = {"path": path, "contents": chunk_b64}
        r = requests.post(add_block_url, json=add_payload, headers=HEADERS)
        r.raise_for_status()

    # 3) Close file
    close_url = base + "/close"
    close_payload = {"path": path}
    r = requests.post(close_url, json=close_payload, headers=HEADERS)
    r.raise_for_status()
    return r.json()

# -------------------------------
# 3️⃣ Run Databricks Job
# -------------------------------
def run_job(job_id, notebook_params=None):
    """
    Trigger a Databricks Job and wait until it completes.
    Returns the job's result state (SUCCESS, FAILED, etc.)
    """
    url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.1/jobs/run-now"
    payload = {"job_id": job_id}
    if notebook_params:
        payload["notebook_params"] = notebook_params

    r = requests.post(url, json=payload, headers=HEADERS)
    r.raise_for_status()
    run_id = r.json()["run_id"]

    # Polling loop to wait until job finishes
    while True:
        status_url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.1/jobs/runs/get?run_id={run_id}"
        status_r = requests.get(status_url, headers=HEADERS)
        status_r.raise_for_status()
        state = status_r.json()["state"]
        life_cycle = state["life_cycle_state"]
        result_state = state.get("result_state", None)

        if life_cycle == "TERMINATED":
            return result_state
        time.sleep(10)  # wait 10 seconds before next check

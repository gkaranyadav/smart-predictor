# databricks_api.py
import streamlit as st
import requests
import base64
import time

# -------------------------------
# Read Databricks secrets from Streamlit Cloud with error handling
# -------------------------------
def get_secret(key, default=None):
    """Safely get secret from Streamlit secrets"""
    try:
        return st.secrets[key]
    except (KeyError, FileNotFoundError):
        if default is not None:
            return default
        st.error(f"❌ Missing secret: {key}. Please configure secrets in Streamlit Cloud.")
        st.stop()

# Get secrets with safe fallbacks
DATABRICKS_HOST = get_secret("DATABRICKS_HOST")
DATABRICKS_TOKEN = get_secret("DATABRICKS_TOKEN")

if DATABRICKS_HOST and DATABRICKS_TOKEN:
    HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}
else:
    HEADERS = {}

# -------------------------------
# 1️⃣ Upload small file to DBFS
# -------------------------------
def dbfs_put_single(path, file_obj, overwrite=False):
    """
    Upload small files (<2MB) to DBFS in a single request
    """
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
            
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
        return {"status": "success", "message": f"File uploaded to {path}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Upload failed: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

# -------------------------------
# 2️⃣ Upload large file in chunks
# -------------------------------
def dbfs_upload_chunked(path, file_obj, overwrite=False, chunk_size=2*1024*1024):
    """
    Upload large files (>2MB) to DBFS using create -> add-block -> close
    chunk_size default: 2MB
    """
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
            
        base = f"{DATABRICKS_HOST.rstrip('/')}/api/2.0/dbfs"
        
        # 1) Create file
        create_url = base + "/create"
        create_payload = {"path": path, "overwrite": overwrite}
        r = requests.post(create_url, json=create_payload, headers=HEADERS)
        r.raise_for_status()

        # 2) Add blocks
        file_obj.seek(0)
        block_count = 0
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
            block_count += 1

        # 3) Close file
        close_url = base + "/close"
        close_payload = {"path": path}
        r = requests.post(close_url, json=close_payload, headers=HEADERS)
        r.raise_for_status()
        return {"status": "success", "message": f"File uploaded in {block_count} chunks to {path}"}
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"Chunked upload failed: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

# -------------------------------
# 3️⃣ Run Databricks Job with Enhanced Error Handling
# -------------------------------
def run_job(job_id, notebook_params=None):
    """
    Trigger a Databricks Job and wait until it completes.
    Returns the job's result state (SUCCESS, FAILED, etc.)
    """
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
            
        url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.1/jobs/run-now"
        payload = {"job_id": job_id}
        if notebook_params:
            payload["notebook_params"] = notebook_params

        r = requests.post(url, json=payload, headers=HEADERS)
        r.raise_for_status()
        run_id = r.json()["run_id"]
        
        st.info(f"Job started with run_id: {run_id}")
        
        # Polling loop to wait until job finishes
        max_attempts = 60  # 10 minutes max wait
        attempt = 0
        
        while attempt < max_attempts:
            status_url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.1/jobs/runs/get?run_id={run_id}"
            status_r = requests.get(status_url, headers=HEADERS)
            status_r.raise_for_status()
            state = status_r.json()["state"]
            life_cycle = state["life_cycle_state"]
            result_state = state.get("result_state", None)

            if life_cycle == "TERMINATED":
                return {
                    "status": "success", 
                    "result_state": result_state,
                    "run_id": run_id
                }
            elif life_cycle == "INTERNAL_ERROR":
                return {
                    "status": "error",
                    "message": "Job internal error",
                    "run_id": run_id
                }
            elif life_cycle == "SKIPPED":
                return {
                    "status": "error", 
                    "message": "Job was skipped",
                    "run_id": run_id
                }
                
            attempt += 1
            time.sleep(10)  # wait 10 seconds before next check
            
        return {
            "status": "error",
            "message": f"Job timed out after {max_attempts} attempts",
            "run_id": run_id
        }
        
    except requests.exceptions.RequestException as e:
        return {"status": "error", "message": f"API call failed: {str(e)}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

# databricks_api.py
import streamlit as st
import requests
import base64
import time
import io

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
# 1️⃣ Upload small file to DBFS (SIMPLIFIED)
# -------------------------------
def dbfs_put_single(path, file_obj, overwrite=False):
    """
    Upload files to DBFS in a single request - works for files up to 10MB
    """
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
            
        # Read file content
        content = file_obj.read()
        if isinstance(content, str):
            content = content.encode("utf-8")

        # Check file size (DBFS single put has limits)
        if len(content) > 10 * 1024 * 1024:  # 10MB limit for single put
            return {"status": "error", "message": "File too large for single upload. Use chunked upload."}

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
# 2️⃣ Upload large file in chunks (FIXED VERSION)
# -------------------------------
def dbfs_upload_chunked(path, file_obj, overwrite=False, chunk_size=1*1024*1024):
    """
    Upload large files to DBFS using create -> add-block -> close
    Reduced chunk_size to 1MB for better reliability
    """
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
            
        base_url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.0/dbfs"
        
        # 1) Create handle
        create_url = f"{base_url}/create"
        create_payload = {"path": path, "overwrite": overwrite}
        create_response = requests.post(create_url, json=create_payload, headers=HEADERS)
        create_response.raise_for_status()
        handle = create_response.json()["handle"]
        
        # 2) Upload chunks
        file_obj.seek(0)
        chunk_count = 0
        total_size = 0
        
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
                
            if isinstance(chunk, str):
                chunk = chunk.encode("utf-8")
                
            chunk_b64 = base64.b64encode(chunk).decode("utf-8")
            
            add_block_url = f"{base_url}/add-block"
            add_block_payload = {
                "handle": handle,
                "data": chunk_b64
            }
            
            add_block_response = requests.post(add_block_url, json=add_block_payload, headers=HEADERS)
            add_block_response.raise_for_status()
            
            chunk_count += 1
            total_size += len(chunk)
        
        # 3) Close handle
        close_url = f"{base_url}/close"
        close_payload = {"handle": handle}
        close_response = requests.post(close_url, json=close_payload, headers=HEADERS)
        close_response.raise_for_status()
        
        return {
            "status": "success", 
            "message": f"File uploaded successfully in {chunk_count} chunks ({total_size/(1024*1024):.2f} MB)"
        }
        
    except requests.exceptions.RequestException as e:
        error_detail = ""
        try:
            if e.response is not None:
                error_detail = f" - {e.response.text}"
        except:
            pass
        return {"status": "error", "message": f"Chunked upload failed: {str(e)}{error_detail}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

# -------------------------------
# 3️⃣ Alternative: Use Files API for upload (Recommended)
# -------------------------------
def upload_to_dbfs_simple(file_obj, dbfs_path):
    """
    Simple upload using local file creation and dbfs cp command
    This is more reliable for larger files
    """
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
        
        # For now, let's use the single put method with smaller chunks
        # We'll implement a more robust solution later
        file_size = len(file_obj.getvalue()) if hasattr(file_obj, 'getvalue') else file_obj.size
        
        if file_size <= 10 * 1024 * 1024:  # 10MB
            return dbfs_put_single(dbfs_path, file_obj, overwrite=True)
        else:
            return dbfs_upload_chunked(dbfs_path, file_obj, overwrite=True)
            
    except Exception as e:
        return {"status": "error", "message": f"Upload failed: {str(e)}"}

# -------------------------------
# 4️⃣ Run Databricks Job with Enhanced Error Handling
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

        response = requests.post(url, json=payload, headers=HEADERS)
        response.raise_for_status()
        run_id = response.json()["run_id"]
        
        st.info(f"Job started with run_id: {run_id}")
        
        # Polling loop to wait until job finishes
        max_attempts = 60  # 10 minutes max wait
        attempt = 0
        
        while attempt < max_attempts:
            status_url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.1/jobs/runs/get?run_id={run_id}"
            status_response = requests.get(status_url, headers=HEADERS)
            status_response.raise_for_status()
            state = status_response.json()["state"]
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
        error_detail = ""
        try:
            if e.response is not None:
                error_detail = f" - {e.response.text}"
        except:
            pass
        return {"status": "error", "message": f"API call failed: {str(e)}{error_detail}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

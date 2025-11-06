# databricks_api.py
import streamlit as st
import requests
import base64
import time
import json

def get_secret(key, default=None):
    try:
        return st.secrets[key]
    except:
        if default is not None:
            return default
        st.error(f"❌ Missing secret: {key}")
        st.stop()

DATABRICKS_HOST = get_secret("DATABRICKS_HOST")
DATABRICKS_TOKEN = get_secret("DATABRICKS_TOKEN")

if DATABRICKS_HOST and DATABRICKS_TOKEN:
    HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}
else:
    HEADERS = {}

# File upload functions
def dbfs_put_single(path, file_obj, overwrite=False):
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
            
        content = file_obj.read()
        if isinstance(content, str):
            content = content.encode("utf-8")

        if len(content) > 10 * 1024 * 1024:
            return {"status": "error", "message": "File too large for single upload"}

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
    except Exception as e:
        return {"status": "error", "message": f"Upload failed: {str(e)}"}

def dbfs_upload_chunked(path, file_obj, overwrite=False, chunk_size=1*1024*1024):
    """
    Upload large files to DBFS using create -> add-block -> close
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
        
    except Exception as e:
        return {"status": "error", "message": f"Chunked upload failed: {str(e)}"}

def upload_to_dbfs_simple(file_obj, dbfs_path):
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
        
        # Get file size
        file_obj.seek(0, 2)
        file_size = file_obj.tell()
        file_obj.seek(0)
        
        if file_size <= 10 * 1024 * 1024:  # 10MB
            return dbfs_put_single(dbfs_path, file_obj, overwrite=True)
        else:
            return dbfs_upload_chunked(dbfs_path, file_obj, overwrite=True)
            
    except Exception as e:
        return {"status": "error", "message": f"Upload failed: {str(e)}"}

# Job execution
def run_job(job_id, job_params=None):
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
            
        url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.1/jobs/run-now"
        payload = {"job_id": job_id}
        
        if job_params:
            payload["job_parameters"] = job_params
        
        st.info(f"📤 Starting job with parameters: {json.dumps(job_params, indent=2)}")
        
        response = requests.post(url, json=payload, headers=HEADERS)
        
        if response.status_code != 200:
            error_detail = response.json()
            return {
                "status": "error", 
                "message": f"API Error {response.status_code}: {error_detail.get('message', 'Unknown error')}",
                "details": error_detail
            }
            
        response.raise_for_status()
        
        run_data = response.json()
        run_id = run_data["run_id"]
        
        st.info(f"🔄 Job started: {run_id}")
        
        # Wait for completion
        max_attempts = 60
        attempt = 0
        
        while attempt < max_attempts:
            status_url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.1/jobs/runs/get?run_id={run_id}"
            status_response = requests.get(status_url, headers=HEADERS)
            
            if status_response.status_code != 200:
                time.sleep(10)
                attempt += 1
                continue
                
            status_response.raise_for_status()
            
            run_info = status_response.json()
            state = run_info["state"]
            life_cycle = state["life_cycle_state"]
            result_state = state.get("result_state", None)

            if attempt % 3 == 0:
                st.info(f"⏳ Status: {life_cycle} ({attempt + 1}/{max_attempts})")

            if life_cycle == "TERMINATED":
                if result_state == "SUCCESS":
                    st.success("✅ Job completed successfully!")
                    return {
                        "status": "success", 
                        "result_state": result_state,
                        "run_id": run_id,
                        "message": "Job completed successfully"
                    }
                else:
                    st.error(f"❌ Job failed: {result_state}")
                    return {
                        "status": "error",
                        "result_state": result_state,
                        "run_id": run_id,
                        "message": f"Job failed: {result_state}"
                    }
            elif life_cycle in ["INTERNAL_ERROR", "SKIPPED"]:
                st.error(f"❌ Job ended: {life_cycle}")
                return {
                    "status": "error",
                    "message": f"Job ended: {life_cycle}",
                    "run_id": run_id
                }
                
            attempt += 1
            time.sleep(10)
            
        return {
            "status": "error",
            "message": f"Job timed out after {max_attempts} attempts",
            "run_id": run_id
        }
        
    except Exception as e:
        return {"status": "error", "message": f"API call failed: {str(e)}"}

# File reading functions
def dbfs_read_file(dbfs_path):
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
            
        url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.0/dbfs/read"
        payload = {
            "path": dbfs_path,
            "offset": 0,
            "length": 5000000  # 5MB max
        }
        
        response = requests.post(url, json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            data = response.json()
            content_b64 = data.get("data", "")
            if content_b64:
                content = base64.b64decode(content_b64).decode("utf-8")
                return {"status": "success", "content": content}
            else:
                return {"status": "error", "message": "File empty or doesn't exist"}
        else:
            return {"status": "error", "message": f"Failed to read file: {response.text}"}
    except Exception as e:
        return {"status": "error", "message": f"Error reading file: {str(e)}"}

def dbfs_file_exists(dbfs_path):
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return False
            
        url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.0/dbfs/get-status"
        payload = {"path": dbfs_path}
        
        response = requests.post(url, json=payload, headers=HEADERS)
        return response.status_code == 200
    except:
        return False

def dbfs_list_files(directory_path):
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
            
        url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.0/dbfs/list"
        payload = {"path": directory_path}
        
        response = requests.post(url, json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            data = response.json()
            return {"status": "success", "files": data.get("files", [])}
        else:
            return {"status": "error", "message": f"Failed to list directory: {response.text}"}
    except Exception as e:
        return {"status": "error", "message": f"Error listing files: {str(e)}"}

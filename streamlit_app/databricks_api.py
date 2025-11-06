# databricks_api.py
import streamlit as st
import requests
import base64
import time
import json

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
    Upload small files to DBFS in a single request
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
# 2️⃣ Upload large file in chunks
# -------------------------------
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
# 3️⃣ Unified upload function
# -------------------------------
def upload_to_dbfs_simple(file_obj, dbfs_path):
    """
    Simple upload that automatically chooses the best method
    """
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
        
        # Get file size
        file_obj.seek(0, 2)  # Seek to end to get size
        file_size = file_obj.tell()
        file_obj.seek(0)  # Reset to beginning
        
        if file_size <= 10 * 1024 * 1024:  # 10MB
            return dbfs_put_single(dbfs_path, file_obj, overwrite=True)
        else:
            return dbfs_upload_chunked(dbfs_path, file_obj, overwrite=True)
            
    except Exception as e:
        return {"status": "error", "message": f"Upload failed: {str(e)}"}

# -------------------------------
# 4️⃣ Run Databricks Job with JOB PARAMETERS
# -------------------------------
def run_job(job_id, job_params=None):
    """
    Trigger a Databricks Job with JOB PARAMETERS (not notebook parameters)
    """
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
            
        url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.1/jobs/run-now"
        
        # Build payload with JOB PARAMETERS (not notebook_params)
        payload = {"job_id": job_id}
        
        if job_params:
            # Use job_parameters for the new job parameters system
            payload["job_parameters"] = job_params
        
        st.info(f"📤 Sending job request with JOB parameters: {json.dumps(job_params, indent=2)}")
        
        response = requests.post(url, json=payload, headers=HEADERS)
        
        # Check for specific error details
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
        
        st.info(f"🔄 Job started with run_id: {run_id}")
        
        # Polling loop to wait until job finishes
        max_attempts = 60  # 10 minutes max wait
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

            # Show progress
            if attempt % 3 == 0:  # Update every 30 seconds
                st.info(f"⏳ Job status: {life_cycle} (attempt {attempt + 1}/{max_attempts})")

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
                    st.error(f"❌ Job failed with state: {result_state}")
                    return {
                        "status": "error",
                        "result_state": result_state,
                        "run_id": run_id,
                        "message": f"Job failed with state: {result_state}"
                    }
            elif life_cycle in ["INTERNAL_ERROR", "SKIPPED"]:
                st.error(f"❌ Job ended with state: {life_cycle}")
                return {
                    "status": "error",
                    "message": f"Job ended with state: {life_cycle}",
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
                error_detail = f" - Response: {e.response.text}"
        except:
            pass
        return {"status": "error", "message": f"API call failed: {str(e)}{error_detail}"}
    except Exception as e:
        return {"status": "error", "message": f"Unexpected error: {str(e)}"}

# -------------------------------
# 5️⃣ Get Job Output and Results
# -------------------------------
def get_job_output(run_id):
    """
    Get the output and logs from a completed job run
    """
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
            
        # Get run output
        output_url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.1/jobs/runs/get-output?run_id={run_id}"
        output_response = requests.get(output_url, headers=HEADERS)
        
        if output_response.status_code == 200:
            output_data = output_response.json()
            return {
                "status": "success", 
                "output": output_data,
                "notebook_output": output_data.get("notebook_output", {}),
                "logs": output_data.get("logs", ""),
                "metadata": output_data.get("metadata", {})
            }
        else:
            return {"status": "error", "message": f"Failed to get job output: {output_response.text}"}
            
    except Exception as e:
        return {"status": "error", "message": f"Error getting job output: {str(e)}"}

# -------------------------------
# 6️⃣ Download file from DBFS
# -------------------------------
def dbfs_read_file(dbfs_path):
    """
    Read a file from DBFS
    """
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
            
        url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.0/dbfs/read"
        payload = {
            "path": dbfs_path,
            "offset": 0,
            "length": 1000000  # Read up to 1MB
        }
        
        response = requests.post(url, json=payload, headers=HEADERS)
        
        if response.status_code == 200:
            data = response.json()
            content_b64 = data.get("data", "")
            if content_b64:
                content = base64.b64decode(content_b64).decode("utf-8")
                return {"status": "success", "content": content}
            else:
                return {"status": "error", "message": "File is empty or doesn't exist"}
        else:
            return {"status": "error", "message": f"Failed to read file: {response.text}"}
            
    except Exception as e:
        return {"status": "error", "message": f"Error reading file: {str(e)}"}

# -------------------------------
# 7️⃣ Check if file exists in DBFS
# -------------------------------
def dbfs_file_exists(dbfs_path):
    """
    Check if a file exists in DBFS
    """
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
            
        url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.0/dbfs/get-status"
        payload = {"path": dbfs_path}
        
        response = requests.post(url, json=payload, headers=HEADERS)
        return response.status_code == 200
            
    except:
        return False

# -------------------------------
# 8️⃣ List files in DBFS directory
# -------------------------------
def dbfs_list_files(directory_path):
    """
    List files in a DBFS directory
    """
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

# -------------------------------
# 9️⃣ Get Task Runs for Multi-Task Jobs
# -------------------------------
def get_task_runs(run_id):
    """
    Get all task runs for a multi-task job
    """
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
            
        # Get the run details first to see tasks
        status_url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.1/jobs/runs/get?run_id={run_id}"
        status_response = requests.get(status_url, headers=HEADERS)
        status_response.raise_for_status()
        
        run_info = status_response.json()
        tasks = run_info.get("tasks", [])
        
        if not tasks:
            return {"status": "error", "message": "No tasks found in this job run"}
        
        task_outputs = {}
        for task in tasks:
            task_run_id = task.get("run_id")
            task_key = task.get("task_key", "unknown")
            
            if task_run_id:
                # Get output for this specific task
                output_url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.1/jobs/runs/get-output?run_id={task_run_id}"
                output_response = requests.get(output_url, headers=HEADERS)
                
                if output_response.status_code == 200:
                    task_output = output_response.json()
                    task_outputs[task_key] = {
                        "run_id": task_run_id,
                        "output": task_output
                    }
                else:
                    task_outputs[task_key] = {
                        "run_id": task_run_id,
                        "error": f"Failed to get output: {output_response.text}"
                    }
        
        return {
            "status": "success", 
            "task_outputs": task_outputs,
            "run_info": run_info
        }
            
    except Exception as e:
        return {"status": "error", "message": f"Error getting task runs: {str(e)}"}

# -------------------------------
# 🔟 Get Specific Task Output
# -------------------------------
def get_task_output(task_run_id):
    """
    Get output for a specific task run
    """
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
            
        output_url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.1/jobs/runs/get-output?run_id={task_run_id}"
        output_response = requests.get(output_url, headers=HEADERS)
        
        if output_response.status_code == 200:
            output_data = output_response.json()
            return {
                "status": "success", 
                "output": output_data,
                "notebook_output": output_data.get("notebook_output", {}),
                "logs": output_data.get("logs", ""),
                "metadata": output_data.get("metadata", {})
            }
        else:
            return {"status": "error", "message": f"Failed to get task output: {output_response.text}"}
            
    except Exception as e:
        return {"status": "error", "message": f"Error getting task output: {str(e)}"}

# -------------------------------
# 1️⃣1️⃣ Get Run Details
# -------------------------------
def get_run_details(run_id):
    """
    Get detailed information about a job run
    """
    try:
        if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
            return {"status": "error", "message": "Databricks credentials not configured"}
            
        status_url = f"{DATABRICKS_HOST.rstrip('/')}/api/2.1/jobs/runs/get?run_id={run_id}"
        status_response = requests.get(status_url, headers=HEADERS)
        status_response.raise_for_status()
        
        run_info = status_response.json()
        return {
            "status": "success", 
            "run_info": run_info,
            "state": run_info.get("state", {}),
            "tasks": run_info.get("tasks", []),
            "start_time": run_info.get("start_time"),
            "end_time": run_info.get("end_time")
        }
            
    except Exception as e:
        return {"status": "error", "message": f"Error getting run details: {str(e)}"}

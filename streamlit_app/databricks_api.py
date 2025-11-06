import requests
import json
import base64
import streamlit as st
import os

# Get from Streamlit Cloud secrets - NO HARDCODED VALUES
DATABRICKS_INSTANCE = os.getenv("DATABRICKS_INSTANCE")
DATABRICKS_TOKEN = os.getenv("DATABRICKS_TOKEN")
DATABRICKS_JOB_INGEST_ID = os.getenv("DATABRICKS_JOB_INGEST_ID")
DATABRICKS_JOB_TRAIN_ID = os.getenv("DATABRICKS_JOB_TRAIN_ID")
DATABRICKS_JOB_SCORE_ID = os.getenv("DATABRICKS_JOB_SCORE_ID")

def validate_databricks_config():
    """Check if all required environment variables are set"""
    missing = []
    if not DATABRICKS_INSTANCE:
        missing.append("DATABRICKS_INSTANCE")
    if not DATABRICKS_TOKEN:
        missing.append("DATABRICKS_TOKEN")
    if not DATABRICKS_JOB_TRAIN_ID:
        missing.append("DATABRICKS_JOB_TRAIN_ID")
    
    if missing:
        st.error(f"❌ Missing Databricks configuration: {', '.join(missing)}")
        st.info("Please set these in Streamlit Cloud secrets under app settings")
        return False
    return True

def dbfs_upload_chunked(dbfs_path, file_obj, overwrite=True):
    """Upload file to DBFS using chunked upload"""
    if not validate_databricks_config():
        return False
        
    try:
        base = DATABRICKS_INSTANCE.rstrip("/") + "/api/2.0/dbfs"
        
        # Create handle
        create_url = f"{base}/create"
        create_data = {
            "path": dbfs_path,
            "overwrite": overwrite
        }
        
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(create_url, headers=headers, json=create_data)
        
        if response.status_code != 200:
            st.error(f"Failed to create DBFS handle: {response.text}")
            return False
            
        handle = response.json()["handle"]
        
        # Upload chunks
        chunk_size = 1024 * 1024  # 1MB chunks
        while True:
            chunk = file_obj.read(chunk_size)
            if not chunk:
                break
                
            add_block_url = f"{base}/add-block"
            add_block_data = {
                "handle": handle,
                "data": base64.b64encode(chunk).decode()
            }
            
            response = requests.post(add_block_url, headers=headers, json=add_block_data)
            if response.status_code != 200:
                st.error(f"Failed to upload chunk: {response.text}")
                return False
        
        # Close handle
        close_url = f"{base}/close"
        close_data = {"handle": handle}
        
        response = requests.post(close_url, headers=headers, json=close_data)
        if response.status_code != 200:
            st.error(f"Failed to close DBFS handle: {response.text}")
            return False
            
        st.success("✅ File uploaded to Databricks successfully!")
        return True
        
    except Exception as e:
        st.error(f"Error uploading to DBFS: {str(e)}")
        return False

def trigger_databricks_job(job_id, parameters=None):
    """Trigger a Databricks job"""
    if not validate_databricks_config():
        return None
        
    try:
        url = f"{DATABRICKS_INSTANCE}/api/2.1/jobs/run-now"
        
        payload = {
            "job_id": job_id
        }
        
        if parameters:
            payload["notebook_params"] = parameters
            
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        response = requests.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            run_id = response.json()["run_id"]
            st.success(f"✅ Job triggered successfully! Run ID: {run_id}")
            return run_id
        else:
            st.error(f"Failed to trigger job: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error triggering job: {str(e)}")
        return None

def get_job_status(run_id):
    """Get status of a Databricks job run"""
    if not validate_databricks_config():
        return None
        
    try:
        url = f"{DATABRICKS_INSTANCE}/api/2.1/jobs/runs/get"
        
        headers = {
            "Authorization": f"Bearer {DATABRICKS_TOKEN}",
            "Content-Type": "application/json"
        }
        
        params = {"run_id": run_id}
        response = requests.get(url, headers=headers, params=params)
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"Failed to get job status: {response.text}")
            return None
            
    except Exception as e:
        st.error(f"Error getting job status: {str(e)}")
        return None

def get_job_result(run_id):
    """Get the output/result of a completed job"""
    status_info = get_job_status(run_id)
    
    if not status_info:
        return None
        
    state = status_info["state"]
    
    if state["life_cycle_state"] == "TERMINATED":
        if state["result_state"] == "SUCCESS":
            # Get job output
            url = f"{DATABRICKS_INSTANCE}/api/2.1/jobs/runs/get-output"
            headers = {
                "Authorization": f"Bearer {DATABRICKS_TOKEN}",
                "Content-Type": "application/json"
            }
            params = {"run_id": run_id}
            
            response = requests.get(url, headers=headers, params=params)
            if response.status_code == 200:
                return response.json()
            else:
                st.error(f"Failed to get job output: {response.text}")
                return None
        else:
            st.error(f"Job failed: {state.get('state_message', 'Unknown error')}")
            return None
    else:
        # Job still running
        return {"status": "running", "state": state}

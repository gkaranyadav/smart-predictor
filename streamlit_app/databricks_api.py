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
# 3️⃣ Run Databricks Job with JOB PARAMETERS (FIXED)
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

# ... (keep the existing upload functions the same) ...

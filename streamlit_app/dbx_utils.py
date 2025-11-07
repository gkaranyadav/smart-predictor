# dbx_utils.py
import os
import requests
import json
from typing import Dict

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")  # e.g. "https://<your-workspace>.cloud.databricks.com"
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

if not DATABRICKS_HOST or not DATABRICKS_TOKEN:
    # don't raise here; streamlit will show instructions to set them
    pass

HEADERS = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}"
}

def upload_file_to_dbfs(local_path: str, dbfs_path: str):
    """
    Upload local file to DBFS using the 2-step create+put API (small files supported).
    Uses simple single PUT approach. For large files you can chunk.
    """
    # read bytes
    with open(local_path, "rb") as f:
        data = f.read()

    url = f"{DATABRICKS_HOST}/api/2.0/dbfs/put"
    resp = requests.post(url, headers=HEADERS, json={
        "path": dbfs_path,
        "contents": data.hex(),  # DBFS put expects base64, but hex won't work for every workspace; fallback to base64
    })
    # Some workspaces expect base64; if server returns 400 try base64:
    if resp.status_code != 200:
        import base64
        resp = requests.post(url, headers=HEADERS, json={
            "path": dbfs_path,
            "contents": base64.b64encode(data).decode("utf-8"),
        })
    resp.raise_for_status()
    return resp.json()

def run_job_now(job_name: str, notebook_params: Dict = None) -> Dict:
    """
    Starts a run of existing job by name (keeps job name the same).
    Returns run_id and response.
    """
    # First find job id by name
    search_url = f"{DATABRICKS_HOST}/api/2.1/jobs/list"
    r = requests.get(search_url, headers=HEADERS)
    r.raise_for_status()
    jobs = r.json().get("jobs", [])
    job_id = None
    for j in jobs:
        if j.get("settings", {}).get("name") == job_name:
            job_id = j["job_id"]
            break
    if job_id is None:
        raise ValueError(f"Job with name '{job_name}' not found in Databricks workspace.")

    run_url = f"{DATABRICKS_HOST}/api/2.1/jobs/run-now"
    payload = {"job_id": job_id}
    if notebook_params:
        payload["notebook_params"] = notebook_params
    r2 = requests.post(run_url, headers=HEADERS, json=payload)
    r2.raise_for_status()
    return r2.json()

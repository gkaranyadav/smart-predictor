# databricks_api.py
import base64
import json
import os
import requests
from tqdm import tqdm

DATABRICKS_HOST = os.environ.get("DATABRICKS_HOST")  # e.g. https://adb-xxxx.azuredatabricks.net
DATABRICKS_TOKEN = os.environ.get("DATABRICKS_TOKEN")

HEADERS = {"Authorization": f"Bearer {DATABRICKS_TOKEN}"}

def dbfs_put_single(path: str, fileobj, overwrite=True):
    """
    Upload small-to-medium file using the DBFS multipart upload endpoint.
    Uses /api/2.0/dbfs/put (multipart); Databricks supports streaming uploads.
    For very large files, use chunked create/add-block/close (example below).
    """
    url = DATABRICKS_HOST.rstrip("/") + "/api/2.0/dbfs/put"
    # The DBFS put endpoint accepts raw bytes in 'contents' base64 encoded, OR multipart upload.
    # We'll stream in chunks and base64-encode per chunk if needed.
    fileobj.seek(0)
    data = fileobj.read()
    b64 = base64.b64encode(data).decode("utf-8")
    payload = {"path": path, "contents": b64, "overwrite": overwrite}
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()

def dbfs_upload_chunked(path: str, fileobj, overwrite=True, chunk_size=2 * 1024 * 1024):
    """
    Chunked upload for large files using create -> add_block -> close pattern.
    Endpoints: /api/2.0/dbfs/create, /api/2.0/dbfs/add-block, /api/2.0/dbfs/close
    """
    base = DATABRICKS_HOST.rstrip("/") + "/api/2.0/dbfs"
    # 1) create
    create_url = base + "/create"
    create_payload = {"path": path, "overwrite": overwrite}
    r = requests.post(create_url, headers=HEADERS, json=create_payload)
    r.raise_for_status()
    handle = r.json().get("handle")

    # 2) chunk & add-block
    add_url = base + "/add-block"
    fileobj.seek(0)
    while True:
        chunk = fileobj.read(chunk_size)
        if not chunk:
            break
        chunk_b64 = base64.b64encode(chunk).decode("utf-8")
        r = requests.post(add_url, headers=HEADERS, json={"handle": handle, "data": chunk_b64})
        r.raise_for_status()
    # 3) close
    close_url = base + "/close"
    r = requests.post(close_url, headers=HEADERS, json={"handle": handle})
    r.raise_for_status()
    return {"path": path}

def run_job(job_id: int, notebook_params: dict = None):
    """
    Trigger a Databricks job by job_id (pre-created job in Databricks).
    Returns run_id.
    """
    url = DATABRICKS_HOST.rstrip("/") + "/api/2.2/jobs/run-now"
    payload = {"job_id": job_id}
    if notebook_params:
        payload["notebook_params"] = notebook_params
    r = requests.post(url, headers=HEADERS, json=payload)
    r.raise_for_status()
    return r.json()["run_id"]

def get_run(run_id: int):
    url = DATABRICKS_HOST.rstrip("/") + f"/api/2.2/jobs/runs/get?run_id={run_id}"
    r = requests.get(url, headers=HEADERS)
    r.raise_for_status()
    return r.json()

def download_dbfs_file(path: str, local_path: str):
    url = DATABRICKS_HOST.rstrip("/") + "/api/2.0/dbfs/read"
    payload = {"path": path}
    r = requests.get(url, headers=HEADERS, params=payload, stream=True)
    r.raise_for_status()
    data = r.json()
    # read returns base64 'data' string
    b64 = data.get("data")
    decoded = base64.b64decode(b64)
    with open(local_path, "wb") as f:
        f.write(decoded)
    return local_path

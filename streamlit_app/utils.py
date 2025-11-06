# utils.py
import uuid
import os
from datetime import datetime

def gen_session_id():
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]

def safe_dbfs_path(session_id, filename):
    # store in tmp folder
    name = filename.replace(" ", "_")
    return f"/FileStore/tmp/{session_id}/{name}"

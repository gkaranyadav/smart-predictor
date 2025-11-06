# utils.py
import uuid
import os
from datetime import datetime

def gen_session_id():
    """Generate unique session ID for each user session"""
    return datetime.utcnow().strftime("%Y%m%dT%H%M%SZ") + "_" + uuid.uuid4().hex[:8]

def safe_dbfs_path(session_id, filename):
    """Create safe DBFS path for file storage"""
    # Remove spaces and special characters from filename
    name = filename.replace(" ", "_").replace("(", "").replace(")", "")
    return f"/FileStore/tmp/{session_id}/{name}"

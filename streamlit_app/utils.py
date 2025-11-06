# utils.py
import uuid
import datetime

def gen_session_id():
    """Generate unique session ID"""
    return f"{datetime.datetime.now().strftime('%Y%m%dT%H%M%SZ')}_{str(uuid.uuid4())[:8]}"

def safe_dbfs_path(session_id, filename):
    """Create safe DBFS path"""
    import re
    safe_filename = re.sub(r'[^a-zA-Z0-9._-]', '_', filename)
    return f"/FileStore/tmp/{session_id}/{safe_filename}"

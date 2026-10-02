import hashlib
import json
from .db import connect

def _hash(event_type, status, details, previous_hash):
    payload = json.dumps({
        "event_type": event_type, "status": status,
        "details": details, "previous_hash": previous_hash
    }, sort_keys=True, separators=(",",":")).encode()
    return hashlib.sha256(payload).hexdigest()

def append_audit(db_path, event_type, status, details):
    con = connect(db_path)
    row = con.execute("SELECT record_hash FROM audit ORDER BY id DESC LIMIT 1").fetchone()
    previous = row[0] if row else "GENESIS"
    record_hash = _hash(event_type, status, details, previous)
    con.execute(
        "INSERT INTO audit(event_type,status,details,previous_hash,record_hash) VALUES(?,?,?,?,?)",
        (event_type, status, json.dumps(details, sort_keys=True), previous, record_hash)
    )
    con.commit()
    con.close()
    return record_hash

def get_audit_records(db_path, limit=None, event_type=None, status=None):
    con = connect(db_path)
    query = "SELECT id,event_type,status,details,previous_hash,record_hash,created_at FROM audit"
    params = []
    conditions = []
    if event_type:
        conditions.append("event_type = ?")
        params.append(event_type)
    if status:
        conditions.append("status = ?")
        params.append(status)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY id DESC"
    if limit:
        query += f" LIMIT {int(limit)}"
    
    rows = con.execute(query, params).fetchall()
    con.close()
    keys = ["id", "event_type", "status", "details", "previous_hash", "record_hash", "created_at"]
    records = []
    for r in rows:
        d = dict(zip(keys, r))
        try:
            d["details_parsed"] = json.loads(d["details"])
        except Exception:
            d["details_parsed"] = d["details"]
        records.append(d)
    return records

def verify_audit_chain(db_path):
    con = connect(db_path)
    rows = con.execute("SELECT id,event_type,status,details,previous_hash,record_hash FROM audit ORDER BY id").fetchall()
    con.close()
    if not rows:
        return True, "Audit log is empty (Genesis state)."
    previous = "GENESIS"
    for i, (rec_id, event_type, status, details, prev, stored) in enumerate(rows, 1):
        if prev != previous:
            return False, f"Broken chain at Block #{rec_id} (row {i}): Previous hash '{prev[:12]}...' does not match expected '{previous[:12]}...'."
        try:
            details_obj = json.loads(details)
        except Exception:
            details_obj = details
        expected = _hash(event_type, status, details_obj, prev)
        if expected != stored:
            return False, f"Tampered record at Block #{rec_id} (row {i}): Record hash mismatch. Content has been altered after registration."
        previous = stored
    return True, f"Cryptographic integrity verified across all {len(rows)} chained blocks."

def verify_audit_chain_detailed(db_path):
    con = connect(db_path)
    rows = con.execute("SELECT id,event_type,status,details,previous_hash,record_hash,created_at FROM audit ORDER BY id").fetchall()
    con.close()
    
    blocks = []
    previous = "GENESIS"
    chain_valid = True
    first_error = None
    
    for i, (rec_id, event_type, status, details, prev, stored, created_at) in enumerate(rows, 1):
        block_info = {
            "id": rec_id,
            "block_index": i,
            "event_type": event_type,
            "status": status,
            "previous_hash": prev,
            "record_hash": stored,
            "created_at": created_at,
            "link_valid": True,
            "content_valid": True,
            "error": None
        }
        if prev != previous:
            block_info["link_valid"] = False
            block_info["error"] = f"Previous hash mismatch: expected {previous[:10]}..., got {prev[:10]}..."
            if chain_valid:
                chain_valid = False
                first_error = block_info["error"]
        
        try:
            details_obj = json.loads(details)
        except Exception:
            details_obj = details
        expected = _hash(event_type, status, details_obj, prev)
        if expected != stored:
            block_info["content_valid"] = False
            block_info["error"] = "Content hash mismatch (record payload modified)"
            if chain_valid:
                chain_valid = False
                first_error = block_info["error"]
                
        previous = stored
        blocks.append(block_info)
        
    return chain_valid, first_error or f"Chain intact ({len(blocks)} blocks)", blocks

def tamper_audit_record_for_demo(db_path, target_id=None):
    """Directly alters an audit record in SQLite to demonstrate zero-trust chain verification."""
    con = connect(db_path)
    if target_id is None:
        row = con.execute("SELECT id, details FROM audit ORDER BY id ASC LIMIT 1").fetchone()
    else:
        row = con.execute("SELECT id, details FROM audit WHERE id=?", (target_id,)).fetchone()
    
    if not row:
        con.close()
        return None
    
    rec_id, details_str = row
    try:
        data = json.loads(details_str)
        data["tampered_by_attacker"] = True
        data["unauthorized_override"] = "Malicious weight substitution covered up"
        new_details = json.dumps(data)
    except Exception:
        new_details = details_str + " [TAMPERED]"
        
    con.execute("UPDATE audit SET details=? WHERE id=?", (new_details, rec_id))
    con.commit()
    con.close()
    return rec_id

def reset_db_audit_clean(db_path):
    """Resets audit table to clean verified genesis state."""
    con = connect(db_path)
    con.execute("DELETE FROM audit")
    con.commit()
    con.close()
    append_audit(db_path, "SYSTEM_INITIALIZATION", "PASS", {
        "engine": "Sentinel-CV Zero-Trust AI Framework",
        "crypto_version": "Ed25519 + SHA-256",
        "status": "Cryptographic Root Baseline Established"
    })
    return True

def seed_demo_audit(db_path):
    """Seeds rich illustrative audit records for demonstration."""
    reset_db_audit_clean(db_path)
    append_audit(db_path, "DATA_SANITIZATION", "PASS", {
        "batch_id": "BATCH-2026-09A",
        "inspected_count": 250,
        "exact_duplicates_removed": 4,
        "anomalies_flagged": 2
    })
    append_audit(db_path, "MODEL_REGISTERED", "PASS", {
        "name": "yolov8n_surveillance.onnx",
        "sha256": "4a7f01c9b3248e5d610c4fae8912cb4018239a04bd84918e9c2e0fa9501dc2ea",
        "version": "1.0.0"
    })
    append_audit(db_path, "MODEL_VERIFIED", "PASS", {
        "name": "yolov8n_surveillance.onnx",
        "runtime": "Triton-Inference-Engine",
        "status": "FINGERPRINT_VERIFIED"
    })
    append_audit(db_path, "DRIFT_ANALYSIS", "PASS", {
        "score": 0.0124,
        "status": "STABLE",
        "eval_window": "Last 24h Telemetry"
    })
    return True

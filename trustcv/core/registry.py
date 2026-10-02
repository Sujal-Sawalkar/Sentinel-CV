import hashlib
from .db import connect

def sha256_bytes(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def register_model(db_path, name, data, version="1.0", model_type="Computer Vision", description=""):
    digest = sha256_bytes(data)
    size = len(data) if isinstance(data, (bytes, bytearray)) else len(data.encode())
    con = connect(db_path)
    cur = con.cursor()
    # Check if exists
    row = cur.execute("SELECT id FROM models WHERE name=?", (name,)).fetchone()
    if row:
        cur.execute(
            "UPDATE models SET sha256=?, version=?, model_type=?, status='ACTIVE', file_size=?, registered_at=CURRENT_TIMESTAMP WHERE id=?",
            (digest, version, model_type, size, row[0])
        )
    else:
        cur.execute(
            "INSERT INTO models(name, sha256, version, model_type, status, file_size) VALUES(?,?,?,?,?,?)",
            (name, digest, version, model_type, 'ACTIVE', size)
        )
    con.commit()
    con.close()
    return {
        "name": name,
        "sha256": digest,
        "version": version,
        "model_type": model_type,
        "file_size": size,
        "status": "ACTIVE"
    }

def verify_model(db_path, name, data):
    observed = sha256_bytes(data)
    con = connect(db_path)
    row = con.execute("SELECT name, sha256, status, version, registered_at FROM models WHERE name=? ORDER BY id DESC LIMIT 1", (name,)).fetchone()
    con.close()
    
    if not row:
        return {
            "match": False,
            "status": "UNREGISTERED",
            "reason": f"No baseline model registered under '{name}'. Cannot verify artifact provenance.",
            "expected_sha256": None,
            "observed_sha256": observed,
            "name": name
        }
    
    name, expected_sha, status, version, reg_at = row
    is_match = (expected_sha == observed)
    
    return {
        "match": is_match,
        "status": "VERIFIED" if is_match else "TAMPERED_OR_SUBSTITUTED",
        "expected_sha256": expected_sha,
        "observed_sha256": observed,
        "registered_version": version,
        "registered_at": reg_at,
        "name": name,
        "model_status": status,
        "reason": "Cryptographic SHA-256 fingerprint matches registered baseline." if is_match else "Artifact fingerprint differs from registered baseline. Model file has been modified or swapped."
    }

def list_models(db_path):
    con = connect(db_path)
    rows = con.execute("SELECT id, name, sha256, version, model_type, status, file_size, registered_at FROM models ORDER BY id DESC").fetchall()
    con.close()
    keys = ["id", "name", "sha256", "version", "model_type", "status", "file_size", "registered_at"]
    return [dict(zip(keys, r)) for r in rows]

def quarantine_model(db_path, name, reason="Integrity check failed"):
    con = connect(db_path)
    con.execute("UPDATE models SET status='QUARANTINED' WHERE name=?", (name,))
    con.commit()
    con.close()
    return {"name": name, "status": "QUARANTINED", "reason": reason}

def delete_model(db_path, name):
    con = connect(db_path)
    con.execute("DELETE FROM models WHERE name=?", (name,))
    con.commit()
    con.close()
    return True

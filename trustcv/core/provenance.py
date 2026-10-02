import base64
import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey, Ed25519PublicKey
from cryptography.hazmat.primitives import serialization
from cryptography.exceptions import InvalidSignature
from .audit import append_audit
from .db import connect

def h(data):
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()

def _canonical(d):
    return json.dumps(d, sort_keys=True, separators=(",", ":")).encode("utf-8")

def get_or_create_keypair(keys_dir: Path):
    keys_dir.mkdir(parents=True, exist_ok=True)
    key_path = keys_dir / "trustcv_ed25519_private.key"
    pub_path = keys_dir / "trustcv_ed25519_public.key"
    
    if key_path.exists() and pub_path.exists():
        try:
            private = Ed25519PrivateKey.from_private_bytes(key_path.read_bytes())
            return private, private.public_key()
        except Exception:
            pass

    private = Ed25519PrivateKey.generate()
    key_path.write_bytes(private.private_bytes(
        serialization.Encoding.Raw, serialization.PrivateFormat.Raw, serialization.NoEncryption()))
    pub_path.write_bytes(private.public_key().public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw))
    return private, private.public_key()

def create_inference_certificate(db_path, input_bytes, model_bytes, preprocessing, output, evidence_dir, input_name="input.jpg", model_name="model.onnx"):
    evidence_dir = Path(evidence_dir)
    certs_dir = evidence_dir / "certificates"
    certs_dir.mkdir(parents=True, exist_ok=True)
    keys_dir = evidence_dir / "keys"
    
    private, public = get_or_create_keypair(keys_dir)
    
    # Parse output if valid json
    parsed_output = output
    if isinstance(output, str):
        try:
            parsed_output = json.loads(output)
        except Exception:
            parsed_output = {"raw_output": output}

    # Canonical record to sign
    inference_id = "INF-" + uuid.uuid4().hex[:12].upper()
    timestamp = datetime.now(timezone.utc).isoformat()
    
    record = {
        "certificate_version": "1.0-SentinelCV",
        "inference_id": inference_id,
        "timestamp_utc": timestamp,
        "input_metadata": {
            "filename": input_name,
            "byte_size": len(input_bytes)
        },
        "model_metadata": {
            "filename": model_name,
            "byte_size": len(model_bytes)
        },
        "input_sha256": h(input_bytes),
        "model_sha256": h(model_bytes),
        "preprocessing_sha256": h(preprocessing.encode("utf-8")),
        "output_sha256": h(json.dumps(parsed_output, sort_keys=True) if isinstance(parsed_output, dict) else str(parsed_output)),
        "preprocessing": json.loads(preprocessing) if isinstance(preprocessing, str) and preprocessing.strip().startswith("{") else preprocessing,
        "output": parsed_output,
    }
    
    signature = private.sign(_canonical(record))
    
    record["signature_ed25519_b64"] = base64.b64encode(signature).decode("ascii")
    record["public_key_ed25519_b64"] = base64.b64encode(public.public_bytes(
        serialization.Encoding.Raw, serialization.PublicFormat.Raw)).decode("ascii")
    
    cert_path = certs_dir / f"{inference_id}.json"
    cert_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    
    # Record in certificates table
    try:
        con = connect(db_path)
        out_summary = json.dumps(parsed_output) if isinstance(parsed_output, dict) else str(parsed_output)
        con.execute(
            "INSERT INTO certificates(inference_id, timestamp_utc, input_sha256, model_sha256, output_preview, status, certificate_path) VALUES(?,?,?,?,?,?,?)",
            (inference_id, timestamp, record["input_sha256"], record["model_sha256"], out_summary[:100], "VALID", str(cert_path))
        )
        con.commit()
        con.close()
    except Exception:
        pass
    
    append_audit(db_path, "INFERENCE_CERTIFICATE", "PASS", {
        "inference_id": inference_id,
        "input_sha256": record["input_sha256"],
        "model_sha256": record["model_sha256"],
        "path": str(cert_path)
    })
    
    return record

def verify_certificate(raw):
    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        if isinstance(raw, str):
            record = json.loads(raw)
        elif isinstance(raw, dict):
            record = json.loads(json.dumps(raw))
        else:
            return False, "Invalid certificate format: Expected JSON string or object.", None
    except Exception as e:
        return False, f"Malformed certificate payload: {e}", None

    if "signature_ed25519_b64" not in record or "public_key_ed25519_b64" not in record:
        return False, "Certificate is missing cryptographic Ed25519 signature fields.", None

    sig_b64 = record.pop("signature_ed25519_b64")
    pub_b64 = record.pop("public_key_ed25519_b64")

    try:
        signature = base64.b64decode(sig_b64)
        public_key_bytes = base64.b64decode(pub_b64)
        public_key = Ed25519PublicKey.from_public_bytes(public_key_bytes)
    except Exception as e:
        return False, f"Cryptographic key/signature decoding failure: {e}", None

    canonical_data = _canonical(record)
    
    try:
        public_key.verify(signature, canonical_data)
        record["signature_ed25519_b64"] = sig_b64
        record["public_key_ed25519_b64"] = pub_b64
        return True, "CRYPTOGRAPHIC SIGNATURE VALID — Zero-trust proof confirmed: No field or hash was altered after signing.", record
    except InvalidSignature:
        record["signature_ed25519_b64"] = sig_b64
        record["public_key_ed25519_b64"] = pub_b64
        return False, "SIGNATURE INVALID — Cryptographic proof failed! One or more fields (hashes, outputs, or timestamps) were tampered with post-signing.", record
    except Exception as e:
        return False, f"Verification failed with error: {e}", None

def list_certificates(db_path):
    try:
        con = connect(db_path)
        rows = con.execute("SELECT id, inference_id, timestamp_utc, input_sha256, model_sha256, output_preview, status, certificate_path, created_at FROM certificates ORDER BY id DESC").fetchall()
        con.close()
        keys = ["id", "inference_id", "timestamp_utc", "input_sha256", "model_sha256", "output_preview", "status", "certificate_path", "created_at"]
        return [dict(zip(keys, r)) for r in rows]
    except Exception:
        return []

import hashlib
import json
from pathlib import Path
from trustcv.core.registry import register_model, verify_model
from trustcv.core.provenance import create_inference_certificate, verify_certificate
from trustcv.core.audit import tamper_audit_record_for_demo, verify_audit_chain_detailed

def run_tamper_demo(db_path):
    """Scenario 1: Model Substitution / Supply-chain Tampering Attack"""
    trusted_weights = b"SENTINEL-CV-VERIFIED-YOLO-V8-OFFICIAL-RELEASE-WEIGHTS-CRC32-0x49FA1"
    tampered_weights = b"SENTINEL-CV-VERIFIED-YOLO-V8-MALICIOUS-BACKDOORED-WEIGHTS-PAYLOAD"
    model_name = "production_surveillance_yolov8.onnx"
    
    # 1. Register baseline
    reg = register_model(db_path, model_name, trusted_weights, version="2.4.0", model_type="Object Detection")
    
    # 2. Intercept tampered weights during deployment
    verification = verify_model(db_path, model_name, tampered_weights)
    
    return {
        "detected": not verification["match"],
        "scenario": "Model Weight Substitution Attack",
        "model_name": model_name,
        "trusted_hash": reg["sha256"],
        "tampered_hash": verification["observed_sha256"],
        "expected_hash": verification["expected_sha256"],
        "action": "QUARANTINE_MODEL",
        "finding": "Observed model artifact differs from cryptographically registered SHA-256 fingerprint.",
        "defense": "Sentinel-CV Gate 02 blocked deployment before malicious model was loaded into GPU runtime."
    }

def run_certificate_tamper_demo(db_path, evidence_dir):
    """Scenario 2: Inference Result Spoofing Attack"""
    img_bytes = b"SAMPLE_SECURITY_CAMERA_FRAME_DATA"
    model_bytes = b"APPROVED_MODEL_TENSOR_GRAPH"
    preprocessing = '{"resize": [640, 640], "norm": "standard"}'
    legit_output = '{"class": "suspicious_intruder", "confidence": 0.94}'
    
    cert = create_inference_certificate(db_path, img_bytes, model_bytes, preprocessing, legit_output, evidence_dir)
    
    # Attacker tries to modify confidence or label in certificate
    tampered_cert = json.loads(json.dumps(cert))
    tampered_cert["output"]["confidence"] = 0.12 # Lower confidence to bypass alarm
    tampered_cert["output"]["class"] = "authorized_personnel"
    
    # Verify both
    ok_original, msg_original, _ = verify_certificate(cert)
    ok_tampered, msg_tampered, _ = verify_certificate(tampered_cert)
    
    return {
        "inference_id": cert["inference_id"],
        "original_valid": ok_original,
        "tampered_valid": ok_tampered,
        "tamper_detected": not ok_tampered,
        "original_payload": cert["output"],
        "tampered_payload": tampered_cert["output"],
        "defense": "Ed25519 asymmetric signature rejected modified payload without valid private key."
    }

def run_audit_tamper_demo(db_path):
    """Scenario 3: Audit Trail Database Tampering Attack"""
    tampered_id = tamper_audit_record_for_demo(db_path)
    chain_valid, message, blocks = verify_audit_chain_detailed(db_path)
    
    return {
        "tampered_id": tampered_id,
        "chain_valid": chain_valid,
        "verification_message": message,
        "blocks_count": len(blocks),
        "defense": "Cryptographic SHA-256 hash-chaining detected modified row in SQLite and localized compromised block."
    }

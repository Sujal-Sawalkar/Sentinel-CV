# 🛡️ Sentinel-CV — Zero-Trust AI & Computer Vision Integrity Platform

**Sentinel-CV** is an end-to-end Zero-Trust AI Assurance and Cryptographic Provenance Platform designed specifically for Computer Vision (CV) pipelines. It intercepts data tampering, model weight substitution, silent backdoor injection, inference spoofing, and statistical distribution drift by binding every stage of the computer vision lifecycle into immutable, mathematically verifiable cryptographic proofs.

---

## 🌟 Key Capabilities & Architecture

```
                    ┌────────────────────────────────────────────────────────┐
                    │            SENTINEL-CV VERIFICATION GATES              │
                    └────────────────────────────────────────────────────────┘
                                               │
       ┌───────────────────────────────┬───────┴───────────────────────┬───────────────────────────────┐
       ▼                               ▼                               ▼                               ▼
┌──────────────┐               ┌──────────────┐                ┌──────────────┐                ┌──────────────┐
│   GATE 01    │               │   GATE 02    │                │   GATE 03    │                │WATCHDOG & LAB│
│Data Sanitizer│               │Model Finger- │                │  Inference   │                │Drift Monitor │
│              │               │    print     │                │    Binder    │                │ & Tamper Lab │
└──────┬───────┘               └──────┬───────┘                └──────┬───────┘                └──────┬───────┘
       │                              │                               │                               │
  • SHA-256 Dupes                • SHA-256 Root Registry         • Ed25519 Asymmetric Signatures • Multivariate RBF MMD
  • pHash Near-Duplicates        • Model Weight Substitution     • Input + Model + Preproc + Out • Cohen's d Feature Shift
  • Laplacian Blur Detection       Defense                       • Non-repudiation Certificates  • 3-Vector Attack Simulator
  • Luminance Outliers           • Automatic Quarantine Engine   • Tamper-detection Verifier     • Hash-Chain Block Explorer
```

### 1. 🛡️ Gate 01: Data Tier Sanitizer
- **Exact Duplicate Detection:** Intercepts byte-identical images via cryptographic SHA-256 hash collision matching.
- **Perceptual Near-Duplicates:** Computes grayscale median Perceptual Hashes (pHash) with 64-bit Hamming distance thresholding (`dist <= 8`) to detect cropped, re-compressed, or watermarked duplicates.
- **Statistical & Optical Anomaly Outliers:** Evaluates multivariate feature distributions (RGB luminance, standard deviation, discrete 3x3 Laplacian blur variance, and over/underexposure).
- **Interactive Visual Frame Gallery:** Displays thumbnail previews with real-time status chips and dynamic Dataset Trust Scores (0–100%).

### 2. 🔍 Gate 02: Model Fingerprint Console
- **Authoritative Cryptographic Registry:** Maintains registered approved model weights with semantic versioning, architecture categories, file size, and SHA-256 fingerprints.
- **Pre-Execution Runtime Interceptor:** Computes the SHA-256 digest of candidate model artifacts and blocks runtime deployment if a hash mismatch occurs.
- **Quarantine Enforcer:** Automatically isolates suspicious or modified neural network weights in the database to prevent GPU execution.

### 3. 📜 Gate 03: Inference Verification Ledger
- **Cryptographic 4-Way Binding:** Locks together:
  1. Input image payload
  2. Model weights artifact
  3. Preprocessing parameters (resize, normalization, color space)
  4. Model inference output (bounding boxes, class labels, confidence scores)
- **Ed25519 Asymmetric Signatures:** Seals the canonical JSON record with private key digital signatures.
- **Tamper Verifier & Live Attack Test:** Validates mathematical proof of integrity and non-repudiation; highlights any modified confidence scores or forged bounding boxes.

### 4. 📈 Drift Watchdog (Distribution Shift Monitor)
- **Multivariate RBF Maximum Mean Discrepancy (MMD):** Calculates kernel-based distribution shift across multi-dimensional computer vision features.
- **Standardized Cohen's d Divergence:** Evaluates individual feature shifts with 3-tier severity classification (`STABLE`, `MONITOR`, `CRITICAL_DRIFT`).
- **Interactive Altair Visualizations:** Renders feature shift magnitude bar charts with threshold boundaries and overlaid empirical density distributions.
- **Automated Mitigation Guidance:** Generates root-cause diagnostic recommendations (sensor recalibration, lighting adjustments).

### 5. ⚡ Tamper Demonstration Lab
Interactive Red Team vs. Blue Team simulation proving zero-trust interception:
- **Scenario A:** Supply-chain model weight substitution (YOLOv8 Trojan swap).
- **Scenario B:** Inference certificate spoofing (adulterated intruder classification).
- **Scenario C:** Malicious insider database audit trail manipulation (SQLite record alteration).

### 6. 📑 Cryptographic Audit Trail & Compliance Reports
- **Hash-Chained Block Explorer:** Chronological linked ledger where each block's SHA-256 incorporates the previous block's hash (`Genesis -> Block 1 -> Block 2...`).
- **Block Integrity Validator:** Localizes the exact index of any compromised or tampered block.
- **Multi-Format Compliance Export:** 1-click export of complete certified audit logs in JSON and CSV formats.

---

## 🚀 Quick Start & Installation

### Prerequisites
- Python 3.10+ (tested with Python 3.11/3.12/3.13/3.14)
- Windows PowerShell, macOS, or Linux terminal

### Setup Virtual Environment

```powershell
# 1. Clone or navigate to the repository
cd b:\Sentinel-CV

# 2. Activate virtual environment
.venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch the Sentinel-CV interface
python -m streamlit run app.py
```

Open your browser at `http://localhost:8501`.

---

## ⚡ 1-Click Interactive Testing

Sentinel-CV includes pre-generated operational demo datasets so you can test all verification gates immediately without needing external files:

| Page | 1-Click Demo Feature | What It Demonstrates |
| :--- | :--- | :--- |
| **🛡️ Gate 1: Data Assurance** | `⚡ Load Demo Dataset` | Analyzes 7 synthesized camera frames, identifying exact duplicates, pHash near-duplicates, sensor blowout, and heavy noise outliers. |
| **🔍 Gate 2: Model Fingerprint** | Select `yolov8n_surveillance.onnx` vs `yolov8n_surveillance_tampered.onnx` | Tests baseline verification (Green PASS) vs. injected trojan weights (Red INTEGRITY VIOLATION). |
| **📜 Gate 3: Inference Ledger** | Check `Use Demo Camera Frame & YOLOv8 Artifact` | Generates signed Ed25519 certificate; test live payload tampering to observe cryptographic signature failure. |
| **📈 Drift Watchdog** | `⚡ Scenario A (Sensor Degradation)` vs `⚡ Scenario B (Stable Operations)` | Real-time RBF MMD calculation and Altair distribution comparison between daylight and degraded night camera feeds. |
| **⚡ Tamper Demo Lab** | `🚀 Execute Model Tamper Attack` / `Execute Certificate Spoofing` | Live deterministic Red vs. Blue attack simulation and quarantine enforcement. |
| **📑 Reports & Audit** | `Visual Block Explorer` | Inspects cryptographic hash-chain blocks; verifies mathematical continuity. |

---

## 🔒 Cryptographic Specifications

- **Digital Signatures:** Ed25519 (RFC 8032) asymmetric cryptography via Python `cryptography` primitives.
- **Artifact Hashing:** SHA-256 (FIPS 180-4) with canonical JSON serialization (`sort_keys=True, separators=(",", ":")`).
- **Perceptual Image Hashing:** 64-bit median perceptual hashing with Hamming distance matching.
- **Distribution Divergence:** Multivariate Radial Basis Function (RBF) Maximum Mean Discrepancy (MMD) with median heuristic gamma estimation.
- **Audit Persistence:** SQLite with continuous `previous_hash` chaining.

import json
import hashlib
import io
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd
import numpy as np
from PIL import Image
import streamlit as st

from trustcv.core.db import init_db
from trustcv.core.registry import register_model, verify_model, list_models, quarantine_model, delete_model
from trustcv.core.provenance import create_inference_certificate, verify_certificate, list_certificates
from trustcv.core.audit import (
    get_audit_records, verify_audit_chain, verify_audit_chain_detailed,
    append_audit, reset_db_audit_clean, seed_demo_audit
)
from trustcv.engines.data_engine import analyze_images, load_demo_image_files
from trustcv.engines.drift_engine import analyze_drift
from trustcv.engines.demo_engine import run_tamper_demo, run_certificate_tamper_demo, run_audit_tamper_demo
from trustcv.ui_charts import (
    plot_drift_shifts, plot_feature_distribution,
    plot_audit_event_distribution, plot_image_scatter
)

# ----------------- Core Setup & DB Init -----------------
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "trustcv.db"
EVIDENCE_DIR = BASE_DIR / "evidence"
EVIDENCE_DIR.mkdir(exist_ok=True)
(EVIDENCE_DIR / "certificates").mkdir(exist_ok=True)
(EVIDENCE_DIR / "keys").mkdir(exist_ok=True)
init_db(DB_PATH)

st.set_page_config(
    page_title="Sentinel-CV | Zero-Trust AI Integrity",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------- UI / Theme Custom CSS -----------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-base: #060d17;
    --bg-card: #0b1828;
    --bg-card-hover: #0f2238;
    --border-color: rgba(0, 194, 255, 0.18);
    --border-highlight: rgba(0, 194, 255, 0.45);
    --cyan: #00d2ff;
    --emerald: #00f2a1;
    --amber: #ffb800;
    --crimson: #ff4757;
    --text-primary: #f0f6fc;
    --text-muted: #8b9eb0;
}

html, body, [class*="css"] {
    font-family: 'Outfit', -apple-system, BlinkMacSystemFont, sans-serif;
    color: var(--text-primary);
}

code, pre, .mono {
    font-family: 'JetBrains Mono', monospace !important;
}

.block-container {
    padding: 1.2rem 2.2rem 3.5rem;
    max-width: 1550px;
}

/* Sidebar styling */
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050b13 0%, #081320 100%);
    border-right: 1px solid rgba(0, 194, 255, 0.15);
}

section[data-testid="stSidebar"] * {
    color: #d8e5f2;
}

/* Cards & Containers */
.sentinel-hero {
    background: linear-gradient(135deg, rgba(8, 24, 42, 0.95) 0%, rgba(13, 36, 60, 0.85) 100%);
    border: 1px solid var(--border-color);
    border-radius: 16px;
    padding: 22px 26px;
    margin-bottom: 20px;
    box-shadow: 0 8px 32px rgba(0, 0, 0, 0.35);
    position: relative;
    overflow: hidden;
}

.sentinel-hero::after {
    content: "";
    position: absolute;
    top: 0; right: 0; width: 280px; height: 100%;
    background: radial-gradient(circle at center, rgba(0, 210, 255, 0.08) 0%, transparent 70%);
    pointer-events: none;
}

.hero-title {
    font-size: 1.85rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    color: #ffffff;
    margin: 0;
    display: flex;
    align-items: center;
    gap: 12px;
}

.hero-sub {
    color: var(--text-muted);
    font-size: 0.92rem;
    margin-top: 6px;
    max-width: 880px;
    line-height: 1.5;
}

/* KPI metric cards */
.kpi-card {
    background: linear-gradient(145deg, #091726, #0d1e30);
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 16px 18px;
    position: relative;
    transition: all 0.2s ease-in-out;
    box-shadow: 0 4px 16px rgba(0, 0, 0, 0.25);
}

.kpi-card:hover {
    border-color: var(--border-highlight);
    transform: translateY(-2px);
    box-shadow: 0 6px 20px rgba(0, 194, 255, 0.12);
}

.kpi-label {
    font-size: 0.76rem;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-muted);
    font-weight: 600;
}

.kpi-val {
    font-size: 1.85rem;
    font-weight: 800;
    color: #ffffff;
    margin: 6px 0 2px;
    font-family: 'Outfit', sans-serif;
}

.kpi-sub {
    font-size: 0.74rem;
    color: #7b94a8;
}

/* Gate cards */
.gate-card {
    background: linear-gradient(145deg, #0a1727, #0d2136);
    border: 1px solid var(--border-color);
    border-radius: 15px;
    padding: 18px 20px;
    height: 100%;
    display: flex;
    flex-direction: column;
    justify-content: space-between;
    transition: all 0.2s ease;
}

.gate-card:hover {
    border-color: var(--cyan);
    box-shadow: 0 6px 24px rgba(0, 210, 255, 0.15);
}

.gate-badge {
    display: inline-block;
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    background: rgba(0, 210, 255, 0.12);
    color: var(--cyan);
    border: 1px solid rgba(0, 210, 255, 0.25);
    margin-bottom: 8px;
}

.gate-title {
    font-size: 1.12rem;
    font-weight: 700;
    color: #ffffff;
    margin: 0 0 6px;
}

.gate-desc {
    font-size: 0.82rem;
    color: #8fa7bc;
    line-height: 1.45;
    margin-bottom: 14px;
}

/* Status pills */
.badge-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    padding: 4px 10px;
    border-radius: 999px;
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.04em;
}

.badge-pass {
    background: rgba(0, 242, 161, 0.12);
    color: var(--emerald);
    border: 1px solid rgba(0, 242, 161, 0.35);
}

.badge-warn {
    background: rgba(255, 184, 0, 0.12);
    color: var(--amber);
    border: 1px solid rgba(255, 184, 0, 0.35);
}

.badge-fail {
    background: rgba(255, 71, 87, 0.12);
    color: var(--crimson);
    border: 1px solid rgba(255, 71, 87, 0.35);
}

.badge-info {
    background: rgba(0, 210, 255, 0.12);
    color: var(--cyan);
    border: 1px solid rgba(0, 210, 255, 0.35);
}

/* Glass panel */
.sentinel-panel {
    background: #091726;
    border: 1px solid var(--border-color);
    border-radius: 14px;
    padding: 18px 20px;
    margin-bottom: 16px;
}

.panel-header {
    font-size: 1.05rem;
    font-weight: 700;
    color: #ffffff;
    margin-bottom: 12px;
    display: flex;
    justify-content: space-between;
    align-items: center;
}

/* Action Banner */
.action-box {
    background: rgba(0, 210, 255, 0.05);
    border: 1px dashed rgba(0, 210, 255, 0.3);
    border-radius: 12px;
    padding: 14px 16px;
    margin-bottom: 14px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 12px;
}

/* Hash block visualization */
.chain-block {
    background: #081523;
    border: 1px solid #16324c;
    border-radius: 10px;
    padding: 12px 14px;
    margin-bottom: 8px;
    transition: border-color 0.2s;
}

.chain-block:hover {
    border-color: var(--cyan);
}

.block-tampered {
    border: 1px solid var(--crimson) !important;
    background: rgba(255, 71, 87, 0.08) !important;
}

/* Streamlit button enhancement */
button[kind="primary"] {
    background: linear-gradient(135deg, #0091ff 0%, #00c2ff 100%) !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border: none !important;
    border-radius: 8px !important;
    box-shadow: 0 4px 12px rgba(0, 194, 255, 0.3) !important;
}

button[kind="secondary"] {
    border: 1px solid rgba(0, 194, 255, 0.25) !important;
    border-radius: 8px !important;
    background: rgba(9, 23, 38, 0.8) !important;
}

/* Dataframe styling */
[data-testid="stDataFrame"] {
    border: 1px solid var(--border-color);
    border-radius: 10px;
    overflow: hidden;
}
</style>
""", unsafe_allow_html=True)

# ----------------- Navigation State Handling -----------------
PAGES = [
    "📊 Dashboard",
    "🛡️ Gate 1: Data Assurance",
    "🔍 Gate 2: Model Fingerprint",
    "📜 Gate 3: Inference Ledger",
    "📈 Drift Watchdog",
    "⚡ Tamper Demo Lab",
    "📑 Reports & Audit"
]

if "current_page" not in st.session_state:
    st.session_state.current_page = PAGES[0]

def switch_page(target_page_name):
    for p in PAGES:
        if target_page_name.lower() in p.lower():
            st.session_state.current_page = p
            st.rerun()

# ----------------- Sidebar Navigation -----------------
with st.sidebar:
    st.markdown("""
    <div style="padding: 6px 0 14px;">
        <div style="display:flex; align-items:center; gap:10px;">
            <span style="font-size:1.8rem;">🛡️</span>
            <div>
                <div style="font-size:1.25rem; font-weight:800; letter-spacing:0.04em; color:#ffffff;">SENTINEL-CV</div>
                <div style="font-size:0.72rem; color:#00d2ff; font-weight:600; text-transform:uppercase; letter-spacing:0.06em;">Zero-Trust AI Integrity</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    selected_page = st.radio(
        "Navigation Menu",
        PAGES,
        index=PAGES.index(st.session_state.current_page) if st.session_state.current_page in PAGES else 0,
        label_visibility="collapsed",
        key="sidebar_nav_radio"
    )

    if selected_page != st.session_state.current_page:
        st.session_state.current_page = selected_page
        st.rerun()

    st.divider()

    # System Status Telemetry
    chain_ok, chain_msg = verify_audit_chain(DB_PATH)
    total_records = len(get_audit_records(DB_PATH))

    st.markdown("**Platform Status**")
    if chain_ok:
        st.markdown('<span class="badge-pill badge-pass">● CHAIN INTACT / PROTECTED</span>', unsafe_allow_html=True)
    else:
        st.markdown('<span class="badge-pill badge-fail">● AUDIT CHAIN BROKEN</span>', unsafe_allow_html=True)

    st.caption(f"Audit Height: **{total_records} chained blocks**")
    st.caption("Crypto Engine: **Ed25519 · SHA-256**")

    st.divider()

    st.markdown("**Quick Actions**")
    col_sb1, col_sb2 = st.columns(2)
    with col_sb1:
        if st.button("🔄 Clean Chain", use_container_width=True, help="Re-initializes a clean verified audit baseline."):
            reset_db_audit_clean(DB_PATH)
            st.toast("Audit chain reset to verified genesis state!")
            st.rerun()
    with col_sb2:
        if st.button("🌱 Seed Demo", use_container_width=True, help="Populate sample audit records for demonstration."):
            seed_demo_audit(DB_PATH)
            st.toast("Demo audit records seeded!")
            st.rerun()

    st.divider()
    st.markdown("""
    <div style="font-size:0.7rem; color:#678299; line-height:1.4;">
        <b>Sentinel-CV v2.0-Production</b><br>
        Zero-trust verification binds datasets, neural network weights, preprocessing, and inference decisions into tamper-evident cryptographic proofs.
    </div>
    """, unsafe_allow_html=True)

# ----------------- Helper UI Components -----------------
def render_hero(title, subtitle, tag="ACTIVE MONITORING"):
    st.markdown(f"""
    <div class="sentinel-hero">
        <div style="display:flex; justify-content:space-between; align-items:flex-start; margin-bottom:4px;">
            <div class="hero-title">🛡️ {title}</div>
            <span class="badge-pill badge-info">{tag}</span>
        </div>
        <div class="hero-sub">{subtitle}</div>
    </div>
    """, unsafe_allow_html=True)

def render_kpi(label, value, sub="", status="info"):
    color_map = {
        "pass": "var(--emerald)",
        "warn": "var(--amber)",
        "fail": "var(--crimson)",
        "info": "var(--cyan)"
    }
    col = color_map.get(status, "var(--cyan)")
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">{label}</div>
        <div class="kpi-val" style="color:{col};">{value}</div>
        <div class="kpi-sub">{sub}</div>
    </div>
    """, unsafe_allow_html=True)

page = st.session_state.current_page

# =========================================================================
# PAGE 1: 📊 DASHBOARD
# =========================================================================
if page == "📊 Dashboard":
    render_hero(
        "AI Integrity Assurance Dashboard",
        "Continuous zero-trust verification across the computer vision lifecycle: Data Sanitization, Model Fingerprinting, and Signed Cryptographic Inferences.",
        "GATEWAYS OPERATIONAL"
    )

    records = get_audit_records(DB_PATH)
    chain_ok, chain_reason = verify_audit_chain(DB_PATH)
    signed_certs = len(list_certificates(DB_PATH))
    registered_models = len(list_models(DB_PATH))
    
    # Calculate posture score
    posture = 100
    if not chain_ok:
        posture -= 40
    if registered_models == 0:
        posture -= 15
    if len(records) == 0:
        posture -= 10
    posture = max(20, posture)

    # Metric Row
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi("Security Posture", f"{posture}%", "Composite Trust Indicator", "pass" if posture >= 80 else "warn")
    with c2:
        render_kpi("Audit Chain Integrity", "VALID" if chain_ok else "BROKEN", f"{len(records)} Cryptographic Blocks", "pass" if chain_ok else "fail")
    with c3:
        render_kpi("Trusted Models", f"{registered_models}", "SHA-256 Registered Baselines", "info")
    with c4:
        render_kpi("Signed Inferences", f"{signed_certs}", "Ed25519 Sealed Certificates", "pass" if signed_certs > 0 else "info")

    st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

    # 3 Verification Gates Overview
    st.markdown("### 🛡️ Three-Tier Verification Gates")
    g1, g2, g3 = st.columns(3)

    with g1:
        st.markdown("""
        <div class="gate-card">
            <div>
                <span class="gate-badge">GATE 01</span>
                <div class="gate-title">Data Sanitizer</div>
                <div class="gate-desc">
                    Sanitizes vision datasets by intercepting exact SHA-256 duplicates, perceptual near-duplicates (pHash), and sensor exposure/focus anomalies.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Inspect Data Assurance →", key="btn_g1", use_container_width=True):
            switch_page("Gate 1")

    with g2:
        st.markdown("""
        <div class="gate-card">
            <div>
                <span class="gate-badge">GATE 02</span>
                <div class="gate-title">Model Fingerprint</div>
                <div class="gate-desc">
                    Cryptographic hash registry preventing silent weight tampering, backdoor injection, and unauthorized artifact substitution before runtime deployment.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Open Model Console →", key="btn_g2", use_container_width=True):
            switch_page("Gate 2")

    with g3:
        st.markdown("""
        <div class="gate-card">
            <div>
                <span class="gate-badge">GATE 03</span>
                <div class="gate-title">Inference Binder</div>
                <div class="gate-desc">
                    Mathematically seals input images, model weights, preprocessing parameters, and output predictions into non-repudiable Ed25519 signed certificates.
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Review Inference Ledger →", key="btn_g3", use_container_width=True):
            switch_page("Gate 3")

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Lower Section: Audit Activity & Telemetry
    left_col, right_col = st.columns([1.6, 1])

    with left_col:
        st.markdown("### 📋 Recent Hash-Chained Audit Trail")
        if records:
            df_recs = pd.DataFrame(records)
            display_df = df_recs[["id", "event_type", "status", "created_at", "record_hash"]].copy()
            display_df["record_hash"] = display_df["record_hash"].apply(lambda h: f"{h[:10]}...{h[-6:]}")
            display_df.columns = ["Block #", "Event Type", "Status", "Timestamp (UTC)", "SHA-256 Hash"]
            st.dataframe(display_df.head(10), use_container_width=True, hide_index=True)
            
            if st.button("View Full Block Chain Explorer →"):
                switch_page("Reports")
        else:
            st.info("Audit log is currently empty. Run the Tamper Demo or create an inference certificate to generate events.")

    with right_col:
        st.markdown("### 📊 Event Breakdown")
        if records:
            chart = plot_audit_event_distribution(records)
            if chart:
                st.altair_chart(chart, use_container_width=True)
        else:
            st.info("No audit telemetry records yet.")

        # Interactive Tamper Demo Callout
        st.markdown("""
        <div class="sentinel-panel" style="margin-top:10px; background:linear-gradient(135deg, #102135, #0a1726);">
            <div style="font-weight:700; color:#00d2ff; margin-bottom:4px;">⚡ Interactive Threat Lab</div>
            <div style="font-size:0.8rem; color:#8fa7bc; margin-bottom:12px;">
                Simulate supply-chain model substitution, signature forgery, and database tampering in real time.
            </div>
        </div>
        """, unsafe_allow_html=True)
        if st.button("Launch Tamper Demo Lab ⚡", type="primary", use_container_width=True):
            switch_page("Tamper")

# =========================================================================
# PAGE 2: 🛡️ GATE 1: DATA ASSURANCE
# =========================================================================
elif page == "🛡️ Gate 1: Data Assurance":
    render_hero(
        "Gate 01: Data Tier Sanitizer",
        "Inspect image batches for exact duplicates, perceptual near-duplicates (pHash), and sensor exposure/blur anomalies before model ingestion.",
        "GATE 01 ENFORCEMENT"
    )

    # Demo Dataset Loader vs Custom Upload
    st.markdown("""
    <div class="action-box">
        <div>
            <b>🚀 Need quick sample data?</b>
            <span style="color:#8fa7bc; font-size:0.85rem; margin-left:8px;">
                Load 7 synthesized camera frames with intentional duplicates, perceptual compression, and blowout anomalies.
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        load_demo_clicked = st.button("⚡ Load Demo Dataset", type="primary", use_container_width=True)
    with col_btn2:
        if st.session_state.get("data_engine_loaded_demo", False):
            if st.button("Clear Loaded Dataset", use_container_width=False):
                st.session_state.data_engine_loaded_demo = False
                st.rerun()

    uploaded_files = st.file_uploader(
        "Or upload your own image files",
        type=["png", "jpg", "jpeg", "webp", "bmp"],
        accept_multiple_files=True,
        key="data_uploader"
    )

    # Determine files to process
    files_to_process = []
    if load_demo_clicked:
        st.session_state.data_engine_loaded_demo = True

    if uploaded_files:
        files_to_process = uploaded_files
    elif st.session_state.get("data_engine_loaded_demo", False):
        files_to_process = load_demo_image_files()

    if files_to_process:
        with st.spinner("Executing cryptographic SHA-256, perceptual hash, and statistical outlier engines..."):
            results = analyze_images(files_to_process)

        # Record assessment button in audit log
        c_head1, c_head2 = st.columns([3, 1])
        with c_head1:
            st.markdown(f"### Analysis Report — {results['summary']}")
        with c_head2:
            if st.button("📝 Log to Audit Trail", use_container_width=True):
                append_audit(DB_PATH, "DATA_SANITIZATION", "PASS" if results["trust_score"] >= 75 else "REVIEW", {
                    "images_count": results["count"],
                    "exact_duplicates": len(results["exact_duplicates"]),
                    "near_duplicates": len(results["near_duplicates"]),
                    "outliers": len(results["outliers"]),
                    "trust_score": results["trust_score"]
                })
                st.toast("Data Sanitization assessment logged to audit trail!")

        # KPI Metrics
        m1, m2, m3, m4, m5 = st.columns(5)
        with m1: render_kpi("Inspected", str(results["count"]), "Image Frames", "info")
        with m2: render_kpi("Exact Duplicates", str(len(results["exact_duplicates"])), "SHA-256 Collisions", "fail" if results["exact_duplicates"] else "pass")
        with m3: render_kpi("Near Duplicates", str(len(results["near_duplicates"])), "pHash Hamming ≤ 8", "warn" if results["near_duplicates"] else "pass")
        with m4: render_kpi("Anomalies", str(len(results["outliers"])), "Exposure / Noise Outliers", "warn" if results["outliers"] else "pass")
        with m5: render_kpi("Data Trust Score", f"{results['trust_score']}%", results["status"], "pass" if results["trust_score"] >= 80 else ("warn" if results["trust_score"] >= 60 else "fail"))

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Tabs for categorized findings
        tab_gal, tab_dup, tab_out, tab_raw = st.tabs([
            "🖼️ Visual Frame Gallery",
            "👯 Duplicate Interceptor",
            "⚠️ Statistical Outliers & Quality",
            "📋 Raw Integrity JSON"
        ])

        with tab_gal:
            st.markdown("#### Inspected Frames & Integrity Badges")
            items = results.get("items", [])
            cols_per_row = 4
            for row_idx in range(0, len(items), cols_per_row):
                row_items = items[row_idx: row_idx + cols_per_row]
                g_cols = st.columns(cols_per_row)
                for col, item in zip(g_cols, row_items):
                    with col:
                        # Determine badge
                        is_outlier = len(item.get("anomaly_flags", [])) > 0
                        status_chip = '<span class="badge-pill badge-fail">ANOMALY</span>' if is_outlier else '<span class="badge-pill badge-pass">CLEAN</span>'
                        st.markdown(f"""
                        <div class="sentinel-panel" style="padding:10px; margin-bottom:12px; text-align:center;">
                            <img src="{item['thumbnail']}" style="width:100%; max-height:130px; object-fit:contain; border-radius:8px; margin-bottom:8px; background:#050c15;">
                            <div style="font-weight:700; font-size:0.82rem; white-space:nowrap; overflow:hidden; text-overflow:ellipsis;">{item['name']}</div>
                            <div style="font-size:0.72rem; color:#8fa7bc; margin:2px 0 6px;">{item['resolution']} · {item['mean_brightness']} Brt · {item['contrast']} Ctr</div>
                            <div>{status_chip}</div>
                        </div>
                        """, unsafe_allow_html=True)

        with tab_dup:
            col_d1, col_d2 = st.columns(2)
            with col_d1:
                st.markdown("#### 🔒 Exact SHA-256 Duplicates")
                if results["exact_duplicates"]:
                    for group in results["exact_duplicates"]:
                        st.error(f"Duplicate Collision ({group['count']} identical files):")
                        st.write(group["files"])
                        st.caption(f"SHA-256: `{group['sha256']}`")
                else:
                    st.success("No exact byte-level duplicates found.")

            with col_d2:
                st.markdown("#### 👁️ Perceptual Hash (pHash) Near-Duplicates")
                if results["near_duplicates"]:
                    df_near = pd.DataFrame(results["near_duplicates"])
                    st.dataframe(df_near, use_container_width=True, hide_index=True)
                else:
                    st.success("No perceptual near-duplicates detected (all Hamming distances > 8).")

        with tab_out:
            st.markdown("#### 🔬 Multivariate Statistical & Quality Outliers")
            scatter = plot_image_scatter(results.get("items", []))
            if scatter:
                st.altair_chart(scatter, use_container_width=True)

            if results["outliers"]:
                df_out = pd.DataFrame(results["outliers"])
                st.dataframe(df_out, use_container_width=True, hide_index=True)
            else:
                st.success("All images conform to expected optical luminance, contrast, and feature boundaries.")

        with tab_raw:
            st.json({k: v for k, v in results.items() if k != "items"})
            st.download_button(
                "📥 Export Sanitizer Report (JSON)",
                json.dumps({k: v for k, v in results.items() if k != "items"}, indent=2),
                "data_assurance_report.json",
                "application/json"
            )
    else:
        st.info("Upload image files or click **'⚡ Load Demo Dataset'** to launch the Data Sanitizer engine.")

# =========================================================================
# PAGE 3: 🔍 GATE 2: MODEL FINGERPRINT
# =========================================================================
elif page == "🔍 Gate 2: Model Fingerprint":
    render_hero(
        "Gate 02: Model Fingerprint Console",
        "Cryptographic SHA-256 verification of neural network model artifacts. Intercepts unauthorized weight modifications, backdoor injection, and supply-chain substitution.",
        "GATE 02 ENFORCEMENT"
    )

    tab_verify, tab_reg = st.tabs(["🛡️ Verify Observed Model Artifact", "➕ Register New Baseline Model"])

    with tab_verify:
        st.markdown("### Runtime Integrity Verification")
        st.caption("Verify that a candidate model file matches the registered cryptographic baseline before deploying into inference execution.")

        # Preset demo selection
        c_m1, c_m2 = st.columns([1.5, 1])
        with c_m1:
            demo_models_dir = BASE_DIR / "data" / "demo_models"
            demo_files = list(demo_models_dir.glob("*.onnx")) if demo_models_dir.exists() else []
            demo_names = [f.name for f in demo_files]
            
            selected_demo = st.selectbox(
                "Quick Test: Select from Demo Models",
                ["-- Upload Custom File Below --"] + demo_names,
                index=1 if len(demo_names) > 0 else 0
            )

        model_file = st.file_uploader(
            "Or upload candidate model file (.onnx, .pt, .pth, .bin, .zip)",
            type=["onnx", "pt", "pth", "bin", "zip", "json", "txt"],
            key="verify_model_upload"
        )

        candidate_name = None
        candidate_bytes = None

        if selected_demo != "-- Upload Custom File Below --":
            target_path = demo_models_dir / selected_demo
            candidate_name = selected_demo.replace("_tampered", "") # map back to target baseline
            candidate_bytes = target_path.read_bytes()
            st.caption(f"Using sample model: `{selected_demo}` ({len(candidate_bytes)} bytes)")
        elif model_file:
            candidate_name = model_file.name
            candidate_bytes = model_file.getvalue()

        if candidate_bytes:
            sha_observed = hashlib.sha256(candidate_bytes).hexdigest()
            st.markdown(f"**Observed Artifact SHA-256:** `{sha_observed}`")

            if st.button("🔍 Verify Artifact Against Trusted Registry", type="primary"):
                result = verify_model(DB_PATH, candidate_name, candidate_bytes)
                
                if result["match"]:
                    st.success("✅ **MODEL INTEGRITY VERIFIED** — Cryptographic SHA-256 fingerprint matches approved baseline.")
                    append_audit(DB_PATH, "MODEL_VERIFIED", "PASS", result)
                else:
                    st.error("🚨 **INTEGRITY VIOLATION DETECTED** — Candidate artifact differs from approved baseline. Deployment blocked!")
                    append_audit(DB_PATH, "MODEL_VERIFICATION_FAILED", "FAIL", result)
                    
                    if st.button("🛡️ Quarantine Model in Registry", type="secondary"):
                        quarantine_model(DB_PATH, candidate_name, "Integrity check failed: Observed SHA-256 mismatch")
                        st.warning(f"Model '{candidate_name}' quarantined.")
                        st.rerun()

                st.json(result)

    with tab_reg:
        st.markdown("### Register New Approved Baseline")
        st.caption("Establish the authoritative cryptographic root of trust for an approved model artifact.")
        
        reg_file = st.file_uploader("Upload trusted model weights", type=["onnx", "pt", "pth", "bin", "zip"], key="reg_upload")
        col_r1, col_r2 = st.columns(2)
        with col_r1:
            reg_name = st.text_input("Model Identifier", value=reg_file.name if reg_file else "yolov8n_surveillance.onnx")
            reg_ver = st.text_input("Semantic Version", value="1.0.0")
        with col_r2:
            reg_type = st.selectbox("Architecture Category", ["Object Detection", "Image Classification", "Instance Segmentation", "Feature Extractor", "Pose Estimation"])
            reg_desc = st.text_area("Audit Notes", value="Officially validated baseline weights.")

        if reg_file and st.button("Register as Trusted Model Baseline", type="primary"):
            res = register_model(DB_PATH, reg_name, reg_file.getvalue(), version=reg_ver, model_type=reg_type, description=reg_desc)
            append_audit(DB_PATH, "MODEL_REGISTERED", "PASS", res)
            st.success(f"Model '{reg_name}' registered successfully with SHA-256 fingerprint: `{res['sha256']}`")
            st.rerun()

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    # Registered Models Inventory
    st.markdown("### 📋 Approved Model Security Registry")
    models_list = list_models(DB_PATH)
    if models_list:
        df_models = pd.DataFrame(models_list)
        df_models["sha256"] = df_models["sha256"].apply(lambda h: f"{h[:12]}...{h[-8:]}")
        display_m = df_models[["name", "version", "model_type", "status", "file_size", "sha256", "registered_at"]]
        display_m.columns = ["Model Name", "Version", "Architecture", "Status", "Size (Bytes)", "SHA-256 Fingerprint", "Registered UTC"]
        st.dataframe(display_m, use_container_width=True, hide_index=True)
    else:
        st.info("No models registered yet. Register a baseline or seed demo models from the sidebar.")

# =========================================================================
# PAGE 4: 📜 GATE 3: INFERENCE LEDGER
# =========================================================================
elif page == "📜 Gate 3: Inference Ledger":
    render_hero(
        "Gate 03: Inference Verification Ledger",
        "Cryptographically bind input images, model weights, preprocessing configurations, and model predictions into signed, verifiable Ed25519 certificates.",
        "GATE 03 ENFORCEMENT"
    )

    tab_issue, tab_check, tab_history = st.tabs([
        "✍️ Seal & Sign Inference",
        "🔍 Verify Certificate Integrity",
        "📜 Sealed Inferences Ledger"
    ])

    with tab_issue:
        st.markdown("### Issue Tamper-Evident Inference Certificate")
        st.caption("Locks the four critical dimensions of an AI inference decision with an Ed25519 asymmetric cryptographic signature.")

        # Preset demo button
        st.markdown("""
        <div class="action-box">
            <div>
                <b>⚡ Quick Inference Generator</b>
                <span style="color:#8fa7bc; font-size:0.85rem; margin-left:8px;">
                    Populate preset security camera frame and YOLOv8 model artifact.
                </span>
            </div>
        </div>
        """, unsafe_allow_html=True)
        use_demo_inf = st.checkbox("Use Demo Camera Frame & YOLOv8 Artifact", value=True)

        col_in1, col_in2 = st.columns(2)
        with col_in1:
            if use_demo_inf:
                demo_img_path = BASE_DIR / "data" / "demo_images" / "camera_01_baseline.png"
                img_bytes = demo_img_path.read_bytes() if demo_img_path.exists() else b"DEMO_IMAGE_BYTES"
                img_name = "camera_01_baseline.png"
                st.success("Loaded preset: camera_01_baseline.png")
            else:
                img_file = st.file_uploader("Input image", type=["png", "jpg", "jpeg", "webp"], key="inf_img_up")
                img_bytes = img_file.getvalue() if img_file else None
                img_name = img_file.name if img_file else "input.jpg"

        with col_in2:
            if use_demo_inf:
                demo_m_path = BASE_DIR / "data" / "demo_models" / "yolov8n_surveillance.onnx"
                model_bytes = demo_m_path.read_bytes() if demo_m_path.exists() else b"DEMO_MODEL_BYTES"
                model_name = "yolov8n_surveillance.onnx"
                st.success("Loaded preset: yolov8n_surveillance.onnx")
            else:
                m_file = st.file_uploader("Model weights", type=["onnx", "pt", "pth", "bin"], key="inf_m_up")
                model_bytes = m_file.getvalue() if m_file else None
                model_name = m_file.name if m_file else "model.onnx"

        # Parameters & Output
        c_txt1, c_txt2 = st.columns(2)
        with c_txt1:
            preproc_text = st.text_area(
                "Preprocessing Configuration (JSON)",
                value='{\n  "resize": [640, 640],\n  "normalization": "imagenet_standard",\n  "color_space": "RGB"\n}',
                height=130
            )
        with c_txt2:
            output_text = st.text_area(
                "Model Inference Output (JSON)",
                value='{\n  "detections": [\n    {"label": "person", "confidence": 0.962, "bbox": [45, 60, 180, 210]},\n    {"label": "backpack", "confidence": 0.884, "bbox": [110, 95, 160, 150]}\n  ]\n}',
                height=130
            )

        if img_bytes and model_bytes:
            # SHA preview bar
            c_s1, c_s2, c_s3, c_s4 = st.columns(4)
            with c_s1: st.caption(f"Input SHA: `{hashlib.sha256(img_bytes).hexdigest()[:12]}...`")
            with c_s2: st.caption(f"Model SHA: `{hashlib.sha256(model_bytes).hexdigest()[:12]}...`")
            with c_s3: st.caption(f"Preproc SHA: `{hashlib.sha256(preproc_text.encode()).hexdigest()[:12]}...`")
            with c_s4: st.caption(f"Output SHA: `{hashlib.sha256(output_text.encode()).hexdigest()[:12]}...`")

        if st.button("🔏 Sign & Seal Inference with Ed25519", type="primary", use_container_width=True):
            if not img_bytes or not model_bytes:
                st.error("Both input image and model artifact are required to seal an inference.")
            else:
                try:
                    cert = create_inference_certificate(
                        DB_PATH, img_bytes, model_bytes, preproc_text, output_text, EVIDENCE_DIR,
                        input_name=img_name, model_name=model_name
                    )
                    st.success(f"✅ **Inference {cert['inference_id']} cryptographically signed and stored in ledger!**")
                    
                    st.json({k: v for k, v in cert.items() if k not in ["private_key"]})
                    
                    st.download_button(
                        "📥 Download Evidence Certificate (JSON)",
                        json.dumps(cert, indent=2),
                        f"{cert['inference_id']}_certificate.json",
                        "application/json"
                    )
                except Exception as exc:
                    st.error(f"Certificate issuance failed: {exc}")

    with tab_check:
        st.markdown("### Cryptographic Certificate Verification")
        st.caption("Upload or test an existing certificate to verify mathematical non-repudiation and detect data tampering.")

        # Preset Demo Certificate Selection
        cert_demo_dir = BASE_DIR / "data" / "demo_certificates"
        valid_cert_path = cert_demo_dir / "valid_inference_certificate.json"
        tampered_cert_path = cert_demo_dir / "tampered_inference_certificate.json"

        col_cd1, col_cd2 = st.columns(2)
        cert_data_to_verify = None

        with col_cd1:
            if valid_cert_path.exists() and st.button("⚡ Load Authentic Valid Certificate", use_container_width=True):
                st.session_state.inspect_cert_content = valid_cert_path.read_text(encoding="utf-8")
        with col_cd2:
            if tampered_cert_path.exists() and st.button("🚨 Load Tampered / Forged Certificate", use_container_width=True):
                st.session_state.inspect_cert_content = tampered_cert_path.read_text(encoding="utf-8")

        cert_file_up = st.file_uploader("Or upload certificate JSON", type=["json"], key="cert_check_upload")
        if cert_file_up:
            st.session_state.inspect_cert_content = cert_file_up.getvalue().decode("utf-8")

        if "inspect_cert_content" in st.session_state:
            cert_raw = st.session_state.inspect_cert_content
            st.text_area("Certificate Payload Preview", value=cert_raw, height=180)

            col_v1, col_v2 = st.columns(2)
            with col_v1:
                if st.button("🛡️ Execute Cryptographic Verification", type="primary", use_container_width=True):
                    is_valid, msg, rec = verify_certificate(cert_raw)
                    if is_valid:
                        st.success(f"✅ {msg}")
                    else:
                        st.error(f"🚨 {msg}")
            with col_v2:
                if st.button("💥 Simulate Live Malicious Tampering", use_container_width=True):
                    try:
                        parsed = json.loads(cert_raw)
                        # Modify confidence in detections
                        if isinstance(parsed.get("output"), dict) and "detections" in parsed["output"]:
                            parsed["output"]["detections"][0]["confidence"] = 0.05
                            parsed["output"]["detections"][0]["label"] = "tampered_ghost"
                        else:
                            parsed["output"] = "TAMPERED_OUTPUT"
                        st.session_state.inspect_cert_content = json.dumps(parsed, indent=2)
                        st.warning("Payload modified! Now click 'Execute Cryptographic Verification' to watch Sentinel-CV catch it.")
                        st.rerun()
                    except Exception as e:
                        st.error(f"Cannot tamper: {e}")

    with tab_history:
        st.markdown("### 📜 Recorded Inferences")
        certs = list_certificates(DB_PATH)
        if certs:
            df_c = pd.DataFrame(certs)
            df_c["input_sha256"] = df_c["input_sha256"].apply(lambda h: f"{h[:10]}...{h[-6:]}")
            df_c["model_sha256"] = df_c["model_sha256"].apply(lambda h: f"{h[:10]}...{h[-6:]}")
            st.dataframe(df_c[["inference_id", "timestamp_utc", "status", "input_sha256", "model_sha256", "output_preview"]], use_container_width=True, hide_index=True)
        else:
            st.info("No certificates sealed in the ledger yet.")

# =========================================================================
# PAGE 5: 📈 DRIFT WATCHDOG
# =========================================================================
elif page == "📈 Drift Watchdog":
    render_hero(
        "Drift Watchdog: Distribution Shift Analyzer",
        "Evaluates multivariate Maximum Mean Discrepancy (RBF MMD) and feature-by-feature divergence between baseline reference and live deployment data.",
        "MONITORING"
    )

    st.markdown("""
    <div class="action-box">
        <div>
            <b>🚀 Test with Built-in Operational Scenarios:</b>
            <span style="color:#8fa7bc; font-size:0.85rem; margin-left:8px;">
                Compare daylight baseline vs degraded night/rain conditions (Drifted) or continuous daylight (Stable).
            </span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    col_s1, col_s2, col_s3 = st.columns(3)
    ref_path = BASE_DIR / "data" / "demo_reference" / "reference.csv"
    dep_drift_path = BASE_DIR / "data" / "demo_deployment" / "deployment.csv"
    dep_stable_path = BASE_DIR / "data" / "demo_deployment" / "deployment_stable.csv"

    ref_df = None
    dep_df = None

    with col_s1:
        if ref_path.exists() and dep_drift_path.exists() and st.button("⚡ Scenario A: Sensor Degradation (Critical Drift)", use_container_width=True):
            st.session_state.drift_ref = pd.read_csv(ref_path)
            st.session_state.drift_dep = pd.read_csv(dep_drift_path)
            st.session_state.drift_scenario_label = "Night-time / Sensor Degradation (Simulated)"

    with col_s2:
        if ref_path.exists() and dep_stable_path.exists() and st.button("⚡ Scenario B: Stable Operations (No Drift)", use_container_width=True):
            st.session_state.drift_ref = pd.read_csv(ref_path)
            st.session_state.drift_dep = pd.read_csv(dep_stable_path)
            st.session_state.drift_scenario_label = "Standard Operational Baseline (Conforming)"

    with col_s3:
        if st.button("Upload Custom CSVs Below", use_container_width=True):
            st.session_state.pop("drift_ref", None)
            st.session_state.pop("drift_dep", None)
            st.session_state.pop("drift_scenario_label", None)

    # File Uploaders
    c_u1, c_u2 = st.columns(2)
    with c_u1:
        ref_up = st.file_uploader("Reference Dataset (Baseline CSV)", type=["csv"], key="ref_up")
        if ref_up:
            st.session_state.drift_ref = pd.read_csv(ref_up)
            st.session_state.drift_scenario_label = "Custom User Reference"

    with c_u2:
        dep_up = st.file_uploader("Deployment Dataset (Production CSV)", type=["csv"], key="dep_up")
        if dep_up:
            st.session_state.drift_dep = pd.read_csv(dep_up)
            st.session_state.drift_scenario_label = "Custom User Deployment"

    if "drift_ref" in st.session_state and "drift_dep" in st.session_state:
        ref_df = st.session_state.drift_ref
        dep_df = st.session_state.drift_dep

        result = analyze_drift(ref_df, dep_df)

        st.markdown(f"### Assessment: **{st.session_state.get('drift_scenario_label', 'Active Analysis')}**")

        # Top KPIs
        k1, k2, k3, k4 = st.columns(4)
        with k1: render_kpi("MMD-like Metric", f"{result['score']:.4f}", "Multivariate RBF Kernel", "pass" if result["score"] < 0.03 else ("warn" if result["score"] < 0.08 else "fail"))
        with k2: render_kpi("System Verdict", result["status"], result["verdict"], "pass" if result["status"] == "STABLE" else ("warn" if result["status"] == "MODERATE_SHIFT" else "fail"))
        with k3: render_kpi("Shifted Features", f"{result['drifted_features']} / {result['total_features']}", "Critical Divergence", "fail" if result["drifted_features"] > 0 else "pass")
        with k4: render_kpi("Evaluated Samples", f"{result['dep_samples']} obs", f"Baseline: {result['ref_samples']} obs", "info")

        st.markdown("<div style='height: 14px;'></div>", unsafe_allow_html=True)

        # Visualizations
        col_c1, col_c2 = st.columns([1.3, 1])
        with col_c1:
            st.markdown("#### Feature Divergence Magnitude (Standardized σ)")
            shift_chart = plot_drift_shifts(result["per_feature"])
            if shift_chart:
                st.altair_chart(shift_chart, use_container_width=True)

        with col_c2:
            st.markdown("#### Feature Density Overlay")
            common_cols = [p["feature"] for p in result["per_feature"]]
            if common_cols:
                feat_choice = st.selectbox("Inspect Feature Distribution", common_cols)
                dens_chart = plot_feature_distribution(ref_df[feat_choice], dep_df[feat_choice], feat_choice)
                st.altair_chart(dens_chart, use_container_width=True)

        # Tabular breakdown & recommendations
        st.markdown("#### 📋 Detailed Feature Metrics & Investigation Guidance")
        st.dataframe(pd.DataFrame(result["per_feature"]), use_container_width=True, hide_index=True)

        st.markdown("##### 💡 Automated Mitigation Recommendations")
        for rec in result["recommendations"]:
            st.markdown(f"• {rec}")

        if st.button("📝 Record Drift Assessment to Audit Log", type="primary"):
            append_audit(DB_PATH, "DRIFT_ANALYSIS", result["status"], {
                "score": result["score"],
                "drifted_features": result["drifted_features"],
                "total_features": result["total_features"],
                "verdict": result["verdict"]
            })
            st.success("Drift assessment appended to the cryptographic audit trail.")
    else:
        st.info("Select a scenario above or upload reference and deployment CSV files to run the drift engine.")

# =========================================================================
# PAGE 6: ⚡ TAMPER DEMO LAB
# =========================================================================
elif page == "⚡ Tamper Demo Lab":
    render_hero(
        "Zero-Trust Tamper Demonstration Lab",
        "Deterministic Red Team vs Blue Team attack simulation proving how Sentinel-CV detects and halts model weight substitution, inference spoofing, and database tampering.",
        "LIVE SIMULATION"
    )

    demo_scenario = st.radio(
        "Select Attack Vector Simulation",
        [
            "Scenario A: Supply-Chain Model Substitution",
            "Scenario B: Inference Certificate Signature Spoofing",
            "Scenario C: Audit Trail Database Tampering"
        ],
        horizontal=True
    )

    if "Scenario A" in demo_scenario:
        st.markdown("""
        <div class="sentinel-panel">
            <div style="font-weight:700; font-size:1.1rem; color:#ffffff; margin-bottom:6px;">
                🎯 Attack Scenario A: Unauthorized Model Weight Swap
            </div>
            <div style="font-size:0.86rem; color:#8fa7bc; line-height:1.5;">
                An adversary compromises the model delivery pipeline or storage bucket, silently substituting the approved <code>yolov8n_surveillance.onnx</code> weights with a trojaned payload. Sentinel-CV Gate 02 intercepts the model prior to GPU runtime execution.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Execute Model Tamper Attack & Defense", type="primary"):
            with st.spinner("Simulating supply-chain injection and Sentinel Gate 02 verification..."):
                res = run_tamper_demo(DB_PATH)
                append_audit(DB_PATH, "TAMPER_DEMO_MODEL", "FAIL" if res["detected"] else "PASS", res)

            if res["detected"]:
                st.error("🚨 **ATTACK DETECTED & NEUTRALIZED BY GATE 02**")
            else:
                st.success("Integrity verified.")

            c1, c2, c3 = st.columns(3)
            with c1: render_kpi("Approved Fingerprint", f"{res['trusted_hash'][:12]}...", "Baseline SHA-256", "pass")
            with c2: render_kpi("Observed Fingerprint", f"{res['tampered_hash'][:12]}...", "Injected SHA-256", "fail")
            with c3: render_kpi("Enforced Action", res["action"], res["defense"], "warn")

            st.json(res)

    elif "Scenario B" in demo_scenario:
        st.markdown("""
        <div class="sentinel-panel">
            <div style="font-weight:700; font-size:1.1rem; color:#ffffff; margin-bottom:6px;">
                🎯 Attack Scenario B: Inference Certificate Spoofing
            </div>
            <div style="font-size:0.86rem; color:#8fa7bc; line-height:1.5;">
                An adversary intercepts a signed security camera inference certificate identifying a <code>suspicious_intruder</code> (0.94 confidence) and rewrites the classification to <code>authorized_personnel</code> (0.12 confidence) to evade detection. Sentinel-CV Gate 03 verifies the Ed25519 digital signature.
            </div>
        </div>
        """, unsafe_allow_html=True)

        if st.button("🚀 Execute Certificate Spoofing Attack & Defense", type="primary"):
            with st.spinner("Issuing genuine certificate, forging payload, and verifying Ed25519 signature..."):
                res = run_certificate_tamper_demo(DB_PATH, EVIDENCE_DIR)
                append_audit(DB_PATH, "TAMPER_DEMO_CERTIFICATE", "FAIL" if res["tamper_detected"] else "PASS", res)

            if res["tamper_detected"]:
                st.error("🚨 **CRYPTOGRAPHIC SIGNATURE FORGERY DETECTED**")
            else:
                st.success("Valid.")

            col_p1, col_p2 = st.columns(2)
            with col_p1:
                st.markdown("##### Original Signed Payload (Approved)")
                st.json(res["original_payload"])
                st.markdown('<span class="badge-pill badge-pass">VALID ED25519 SIGNATURE</span>', unsafe_allow_html=True)

            with col_p2:
                st.markdown("##### Adversary Modified Payload (Spoofed)")
                st.json(res["tampered_payload"])
                st.markdown('<span class="badge-pill badge-fail">SIGNATURE REJECTED - MATH PROOF FAILED</span>', unsafe_allow_html=True)

            st.info(f"🛡️ **Zero-Trust Defense Mechanism:** {res['defense']}")

    elif "Scenario C" in demo_scenario:
        st.markdown("""
        <div class="sentinel-panel">
            <div style="font-weight:700; font-size:1.1rem; color:#ffffff; margin-bottom:6px;">
                🎯 Attack Scenario C: Insider Audit Trail Tampering
            </div>
            <div style="font-size:0.86rem; color:#8fa7bc; line-height:1.5;">
                A malicious insider directly alters a historical audit log entry in the SQLite database to cover up an operational failure. Sentinel-CV's cryptographic SHA-256 hash-chain detector scans the ledger to locate the altered block.
            </div>
        </div>
        """, unsafe_allow_html=True)

        col_t1, col_t2 = st.columns(2)
        with col_t1:
            if st.button("💥 Simulate Unauthorized Database Tampering", type="primary", use_container_width=True):
                res = run_audit_tamper_demo(DB_PATH)
                st.session_state.tamper_c_result = res
                st.error(f"🚨 Row in SQLite mutated! Verifier result: {res['verification_message']}")
        with col_t2:
            if st.button("🔄 Repair & Re-establish Clean Audit Chain", use_container_width=True):
                reset_db_audit_clean(DB_PATH)
                seed_demo_audit(DB_PATH)
                st.session_state.pop("tamper_c_result", None)
                st.success("Audit chain repaired with clean verified genesis baseline!")
                st.rerun()

        if "tamper_c_result" in st.session_state:
            res = st.session_state.tamper_c_result
            st.json(res)
            st.warning("Navigate to **'📑 Reports & Audit'** to see the exact localized broken block in the visual explorer.")

# =========================================================================
# PAGE 7: 📑 REPORTS & AUDIT
# =========================================================================
elif page == "📑 Reports & Audit":
    render_hero(
        "Cryptographic Audit Trail & Compliance Reports",
        "Inspect the immutable SQLite hash-chain ledger, verify block-by-block cryptographic continuity, and export certified compliance documentation.",
        "GOVERNANCE"
    )

    # Run verification scan
    chain_valid, first_err, blocks = verify_audit_chain_detailed(DB_PATH)

    # Top Chain Banner
    if chain_valid:
        st.markdown(f"""
        <div class="sentinel-panel" style="border-left: 4px solid var(--emerald); background: rgba(0, 242, 161, 0.05);">
            <div style="font-size:1.1rem; font-weight:800; color:var(--emerald);">
                ✅ CRYPTOGRAPHIC HASH CHAIN VALID
            </div>
            <div style="font-size:0.84rem; color:#8fa7bc; margin-top:4px;">
                All {len(blocks)} chained blocks cryptographically linked. Every block's SHA-256 hash correctly includes the preceding block's hash.
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        st.markdown(f"""
        <div class="sentinel-panel" style="border-left: 4px solid var(--crimson); background: rgba(255, 71, 87, 0.08);">
            <div style="font-size:1.1rem; font-weight:800; color:var(--crimson);">
                🚨 AUDIT CHAIN BROKEN / COMPROMISED
            </div>
            <div style="font-size:0.84rem; color:#ff8a96; margin-top:4px;">
                {first_err}
            </div>
        </div>
        """, unsafe_allow_html=True)

    tab_explorer, tab_export, tab_maint = st.tabs([
        "🔗 Visual Hash-Chain Block Explorer",
        "📥 Compliance Export & Certification",
        "⚙️ Ledger Maintenance"
    ])

    with tab_explorer:
        st.markdown("### Block-by-Block Integrity Chain")
        st.caption("Each block is cryptographically bound to its predecessor via SHA-256 hash chaining.")

        if blocks:
            for b in blocks:
                is_err = not b["link_valid"] or not b["content_valid"]
                card_class = "chain-block block-tampered" if is_err else "chain-block"
                stat_badge = f'<span class="badge-pill badge-fail">{b["status"]}</span>' if b["status"] in ["FAIL", "DRIFT"] else f'<span class="badge-pill badge-pass">{b["status"]}</span>'
                
                err_html = f"<div style='color:var(--crimson); font-size:0.75rem; margin-top:4px; font-weight:700;'>⚠️ {b['error']}</div>" if is_err else ""
                
                st.markdown(f"""
                <div class="{card_class}">
                    <div style="display:flex; justify-content:space-between; align-items:center;">
                        <div>
                            <b>Block #{b['id']}</b> · <span class="mono" style="color:var(--cyan);">{b['event_type']}</span>
                            {err_html}
                        </div>
                        <div>
                            {stat_badge}
                            <span style="font-size:0.72rem; color:#7893a7; margin-left:8px;">{b['created_at']}</span>
                        </div>
                    </div>
                    <div style="font-size:0.72rem; color:#68869c; margin-top:6px; display:flex; gap:16px;">
                        <span>Prev Hash: <code class="mono">{b['previous_hash'][:12]}...</code></span>
                        <span>Record Hash: <code class="mono">{b['record_hash'][:12]}...</code></span>
                    </div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("No audit blocks in the database. Perform an assurance action or seed demo records.")

    with tab_export:
        st.markdown("### Executive Compliance Reports")
        records = get_audit_records(DB_PATH)
        
        col_ex1, col_ex2 = st.columns(2)
        with col_ex1:
            report_data = {
                "system": "Sentinel-CV Zero-Trust AI Integrity Platform",
                "export_timestamp_utc": datetime.now(timezone.utc).isoformat(),
                "chain_validity": "VALID" if chain_valid else "COMPROMISED",
                "total_blocks_recorded": len(blocks),
                "records": records
            }
            st.download_button(
                "📥 Download Certified Audit Trail (JSON)",
                json.dumps(report_data, indent=2),
                f"sentinel_cv_audit_trail_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                "application/json",
                use_container_width=True
            )

        with col_ex2:
            if records:
                df_export = pd.DataFrame(records)
                csv_bytes = df_export.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "📥 Export Audit Records (CSV)",
                    csv_bytes,
                    "sentinel_cv_audit_records.csv",
                    "text/csv",
                    use_container_width=True
                )

    with tab_maint:
        st.markdown("### Ledger Maintenance & Diagnostics")
        col_m1, col_m2 = st.columns(2)
        with col_m1:
            if st.button("🔄 Reset Ledger to Clean Genesis State", use_container_width=True):
                reset_db_audit_clean(DB_PATH)
                st.success("Ledger reset to clean verified genesis state.")
                st.rerun()
        with col_m2:
            if st.button("🌱 Populate Comprehensive Demo Records", use_container_width=True):
                seed_demo_audit(DB_PATH)
                st.success("Demo audit records populated.")
                st.rerun()

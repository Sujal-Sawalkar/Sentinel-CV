import hashlib
import io
import math
import base64
from pathlib import Path
from PIL import Image, ImageFilter
import numpy as np

def _phash(img, hash_size=8):
    g = img.convert("L").resize((hash_size, hash_size), Image.Resampling.BILINEAR)
    a = np.asarray(g, dtype=np.float32)
    med = np.median(a)
    bits = (a > med).flatten()
    return "".join("1" if x else "0" for x in bits)

def _hamming(a, b):
    return sum(x != y for x, y in zip(a, b))

def _laplacian_variance(img_gray):
    """Estimate image focus / blur using 3x3 discrete Laplacian operator."""
    arr = np.asarray(img_gray, dtype=np.float32)
    h, w = arr.shape
    if h < 3 or w < 3:
        return 0.0
    # Fast vectorized discrete Laplacian using numpy roll
    lap = (
        np.roll(arr, 1, axis=0) + np.roll(arr, -1, axis=0) +
        np.roll(arr, 1, axis=1) + np.roll(arr, -1, axis=1) - 4.0 * arr
    )
    return float(lap[1:-1, 1:-1].var())

def _make_thumbnail_b64(img, size=(120, 120)):
    thumb = img.copy()
    thumb.thumbnail(size)
    buf = io.BytesIO()
    thumb.save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")

def analyze_images(uploaded_or_paths):
    items = []
    
    for item in uploaded_or_paths:
        try:
            if hasattr(item, "getvalue"): # Streamlit UploadedFile
                name = item.name
                data = item.getvalue()
            elif isinstance(item, (str, Path)):
                p = Path(item)
                name = p.name
                data = p.read_bytes()
            elif isinstance(item, dict) and "data" in item:
                name = item.get("name", "image.png")
                data = item["data"]
            else:
                continue

            sha256 = hashlib.sha256(data).hexdigest()
            img = Image.open(io.BytesIO(data))
            rgb = img.convert("RGB")
            gray = img.convert("L")
            w, h = img.size
            
            # Numeric feature vector
            arr_rgb = np.asarray(rgb.resize((32, 32)), dtype=np.float32) / 255.0
            mean_bright = float(np.asarray(gray).mean())
            contrast = float(np.asarray(gray).std())
            blur_var = _laplacian_variance(gray.resize((128, 128)))
            
            feat = np.array([
                arr_rgb.mean(),
                arr_rgb.std(),
                arr_rgb[:, :, 0].mean(),
                arr_rgb[:, :, 1].mean(),
                arr_rgb[:, :, 2].mean(),
                contrast / 100.0,
                mean_bright / 255.0
            ])
            
            thumb_b64 = _make_thumbnail_b64(rgb)

            items.append({
                "name": name,
                "size_bytes": len(data),
                "resolution": f"{w}x{h}",
                "aspect_ratio": round(w / max(1, h), 2),
                "sha256": sha256,
                "phash": _phash(rgb),
                "mean_brightness": round(mean_bright, 1),
                "contrast": round(contrast, 1),
                "blur_score": round(blur_var, 1),
                "feat": feat,
                "thumbnail": thumb_b64,
                "anomaly_flags": []
            })
        except Exception:
            continue

    if not items:
        return {
            "count": 0,
            "exact_duplicates": [],
            "near_duplicates": [],
            "outliers": [],
            "items": [],
            "trust_score": 100,
            "status": "NO_DATA",
            "summary": "No valid image files processed."
        }

    # 1. Exact duplicates (SHA-256 hash collision)
    exact_map = {}
    for r in items:
        exact_map.setdefault(r["sha256"], []).append(r["name"])
    
    exact_dups = []
    exact_dup_names = set()
    for sha, group in exact_map.items():
        if len(group) > 1:
            exact_dups.append({"sha256": sha, "files": group, "count": len(group)})
            for n in group[1:]:
                exact_dup_names.add(n)

    # 2. Near duplicates (Hamming distance on perceptual hash <= 8)
    near_dups = []
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            dist = _hamming(items[i]["phash"], items[j]["phash"])
            if dist <= 8:
                sim_pct = round((1.0 - (dist / 64.0)) * 100, 1)
                near_dups.append({
                    "file_a": items[i]["name"],
                    "file_b": items[j]["name"],
                    "hamming_dist": dist,
                    "similarity_pct": sim_pct
                })

    # 3. Statistical outlier detection (Mahalanobis / Z-score)
    outliers = []
    if len(items) >= 4:
        X = np.vstack([r["feat"] for r in items])
        mu = X.mean(axis=0)
        sd = X.std(axis=0) + 1e-6
        z_scores = np.sqrt(((X - mu) / sd) ** 2).mean(axis=1)
        
        for r, z in zip(items, z_scores):
            z_val = round(float(z), 3)
            flags = []
            if r["mean_brightness"] > 235:
                flags.append("SENSOR_BLOWOUT_OVEREXPOSED")
            elif r["mean_brightness"] < 25:
                flags.append("UNDEREXPOSED_PITCH_BLACK")
            if r["contrast"] < 8:
                flags.append("FLAT_CONTRAST_LOW_INFORMATION")
            if r["blur_score"] < 10 and len(items) > 3:
                flags.append("HEAVY_BLUR_OUT_OF_FOCUS")
            if z_val > 1.8:
                flags.append("STATISTICAL_FEATURE_OUTLIER")
                
            r["anomaly_score"] = z_val
            r["anomaly_flags"] = flags
            
            if flags or z_val > 1.8:
                outliers.append({
                    "name": r["name"],
                    "anomaly_score": z_val,
                    "reasons": ", ".join(flags) if flags else "Multivariate feature shift",
                    "brightness": r["mean_brightness"],
                    "contrast": r["contrast"]
                })
    else:
        for r in items:
            r["anomaly_score"] = 0.0

    # 4. Composite Trust Score calculation
    # Base: 100.
    penalty = 0
    if items:
        dup_ratio = len(exact_dup_names) / len(items)
        near_ratio = len(near_dups) / max(1, len(items))
        outlier_ratio = len(outliers) / len(items)
        penalty = (dup_ratio * 40.0) + (near_ratio * 25.0) + (outlier_ratio * 35.0)
    
    trust_score = max(10, min(100, int(round(100.0 - penalty))))
    
    status = "CLEAN / APPROVED"
    if trust_score < 60:
        status = "CRITICAL / SANITIZATION REQUIRED"
    elif trust_score < 85:
        status = "CAUTION / REVIEW RECOMMENDED"

    return {
        "count": len(items),
        "exact_duplicates": exact_dups,
        "near_duplicates": near_dups,
        "outliers": outliers,
        "items": items,
        "trust_score": trust_score,
        "status": status,
        "summary": f"Analyzed {len(items)} images. Found {len(exact_dups)} duplicate group(s), {len(near_dups)} near-duplicate pair(s), and {len(outliers)} anomaly outlier(s)."
    }

def load_demo_image_files():
    p = Path("data/demo_images")
    if not p.exists():
        return []
    files = sorted(list(p.glob("*.*")))
    res = []
    for f in files:
        if f.suffix.lower() in [".png", ".jpg", ".jpeg", ".webp", ".bmp"]:
            res.append(f)
    return res

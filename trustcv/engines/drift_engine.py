import numpy as np
import pandas as pd

def _rbf_mmd(x, y):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    z = np.vstack([x, y])
    d = ((z[:, None, :] - z[None, :, :]) ** 2).mean(axis=2)
    med = np.median(d[d > 0]) if np.any(d > 0) else 1.0
    gamma = 1.0 / (2.0 * med + 1e-9)
    K = np.exp(-gamma * d)
    n = len(x)
    m = len(y)
    kxx = K[:n, :n]
    kyy = K[n:, n:]
    kxy = K[:n, n:]
    score = float(kxx.sum() / (n * n) + kyy.sum() / (m * m) - 2.0 * kxy.sum() / (n * m))
    return max(0.0, score)

def analyze_drift(ref: pd.DataFrame, dep: pd.DataFrame):
    common_cols = [c for c in ref.columns if c in dep.columns and pd.api.types.is_numeric_dtype(ref[c]) and pd.api.types.is_numeric_dtype(dep[c])]
    if not common_cols:
        return {
            "score": 0.0,
            "status": "ERROR",
            "message": "No common numeric features found between datasets.",
            "per_feature": [],
            "drifted_features": 0,
            "total_features": 0
        }
    
    per_feature = []
    drifted_count = 0
    warning_count = 0
    recommendations = []

    for c in common_cols:
        a = ref[c].dropna().to_numpy(dtype=float)
        b = dep[c].dropna().to_numpy(dtype=float)
        
        if len(a) == 0 or len(b) == 0:
            continue
            
        a_mean, a_std = float(a.mean()), float(a.std())
        b_mean, b_std = float(b.mean()), float(b.std())
        
        pooled_std = np.sqrt((a_std ** 2 + b_std ** 2) / 2.0) + 1e-9
        standardized_shift = abs(a_mean - b_mean) / pooled_std
        
        # Percentage shift
        pct_shift = ((b_mean - a_mean) / (abs(a_mean) + 1e-6)) * 100.0
        
        if standardized_shift > 1.2:
            feat_status = "CRITICAL_DRIFT"
            drifted_count += 1
            recommendations.append(f"Feature '{c}' shifted by {pct_shift:+.1f}% (standardized shift: {standardized_shift:.2f}). Recalibrate preprocessing or inspect camera sensor.")
        elif standardized_shift > 0.5:
            feat_status = "MONITOR"
            warning_count += 1
            recommendations.append(f"Feature '{c}' exhibits moderate divergence ({standardized_shift:.2f} σ). Monitor in production.")
        else:
            feat_status = "STABLE"

        per_feature.append({
            "feature": c,
            "status": feat_status,
            "reference_mean": round(a_mean, 3),
            "reference_std": round(a_std, 3),
            "deployment_mean": round(b_mean, 3),
            "deployment_std": round(b_std, 3),
            "pct_shift": round(pct_shift, 1),
            "standardized_shift": round(standardized_shift, 3)
        })

    # Multivariate MMD calculation
    x_mat = ref[common_cols].dropna().to_numpy(dtype=float)
    y_mat = dep[common_cols].dropna().to_numpy(dtype=float)
    
    # Scale matrices jointly for fair MMD
    combined = np.vstack([x_mat, y_mat])
    mean_vec = combined.mean(axis=0)
    std_vec = combined.std(axis=0) + 1e-6
    x_scaled = (x_mat - mean_vec) / std_vec
    y_scaled = (y_mat - mean_vec) / std_vec

    mmd_score = _rbf_mmd(x_scaled, y_scaled)
    
    if mmd_score > 0.05 or drifted_count >= 2:
        overall_status = "CRITICAL_DRIFT"
        verdict = "Significant statistical drift detected across features. Model predictions may be degraded."
    elif mmd_score > 0.02 or warning_count >= 2 or drifted_count == 1:
        overall_status = "MODERATE_SHIFT"
        verdict = "Moderate distribution divergence observed. Caution advised."
    else:
        overall_status = "STABLE"
        verdict = "Deployment distribution conforms to reference baseline. Model operating within expected parameters."

    if not recommendations:
        recommendations.append("All features conform to baseline operational parameters. No mitigation required.")

    return {
        "score": round(mmd_score, 5),
        "status": overall_status,
        "verdict": verdict,
        "per_feature": per_feature,
        "drifted_features": drifted_count,
        "warning_features": warning_count,
        "total_features": len(common_cols),
        "recommendations": recommendations,
        "ref_samples": len(ref),
        "dep_samples": len(dep)
    }

import altair as alt
import pandas as pd
import numpy as np

def plot_drift_shifts(per_feature_list):
    """Horizontal bar chart of feature drift shifts with color indicators."""
    if not per_feature_list:
        return None
    
    df = pd.DataFrame(per_feature_list)
    chart = alt.Chart(df).mark_bar(cornerRadius=4, height=18).encode(
        x=alt.X("standardized_shift:Q", title="Standardized Divergence (σ)"),
        y=alt.Y("feature:N", sort="-x", title="Feature"),
        color=alt.Color("status:N", scale=alt.Scale(
            domain=["STABLE", "MONITOR", "CRITICAL_DRIFT"],
            range=["#00e699", "#ffb800", "#ff4757"]
        ), legend=alt.Legend(title="Drift Classification", orient="top")),
        tooltip=["feature", "standardized_shift", "pct_shift", "reference_mean", "deployment_mean", "status"]
    ).properties(height=max(160, len(df) * 32))
    
    # Threshold rule line at 1.0 (drift cutoff)
    rule = alt.Chart(pd.DataFrame({'threshold': [1.0]})).mark_rule(
        color='#ff4757', strokeDash=[4, 4], size=1.5
    ).encode(x='threshold:Q')
    
    return (chart + rule).configure_view(strokeWidth=0).configure_axis(
        labelColor="#9bb2c9", titleColor="#c0d7ea", gridColor="rgba(255,255,255,0.06)"
    ).configure_legend(
        labelColor="#dbe8f1", titleColor="#9bb2c9"
    )

def plot_feature_distribution(ref_series, dep_series, feature_name):
    """Overlaid distribution comparison between reference and deployment."""
    df_ref = pd.DataFrame({"value": ref_series, "source": "Baseline Reference"})
    df_dep = pd.DataFrame({"value": dep_series, "source": "Production Deployment"})
    combined = pd.concat([df_ref, df_dep], ignore_index=True)
    
    chart = alt.Chart(combined).mark_area(
        opacity=0.45,
        interpolate='monotone'
    ).transform_density(
        'value',
        groupby=['source'],
        as_=['value', 'density']
    ).encode(
        x=alt.X('value:Q', title=f"{feature_name} Value Distribution"),
        y=alt.Y('density:Q', title="Empirical Density"),
        color=alt.Color('source:N', scale=alt.Scale(
            domain=["Baseline Reference", "Production Deployment"],
            range=["#00d2ff", "#ff8000"]
        ), legend=alt.Legend(orient="top", title=""))
    ).properties(height=220)
    
    return chart.configure_view(strokeWidth=0).configure_axis(
        labelColor="#9bb2c9", titleColor="#c0d7ea", gridColor="rgba(255,255,255,0.06)"
    ).configure_legend(
        labelColor="#dbe8f1"
    )

def plot_audit_event_distribution(records):
    """Donut/bar chart of audit event breakdown."""
    if not records:
        return None
    df = pd.DataFrame(records)
    counts = df.groupby(["event_type", "status"]).size().reset_index(name="count")
    
    chart = alt.Chart(counts).mark_bar(cornerRadius=4).encode(
        x=alt.X("count:Q", title="Event Count"),
        y=alt.Y("event_type:N", sort="-x", title="Event Category"),
        color=alt.Color("status:N", scale=alt.Scale(
            domain=["PASS", "FAIL", "REVIEW", "STABLE", "DRIFT"],
            range=["#00e699", "#ff4757", "#ffb800", "#00d2ff", "#ff7844"]
        ), legend=alt.Legend(title="Status", orient="top")),
        tooltip=["event_type", "status", "count"]
    ).properties(height=180)
    
    return chart.configure_view(strokeWidth=0).configure_axis(
        labelColor="#9bb2c9", titleColor="#c0d7ea", gridColor="rgba(255,255,255,0.06)"
    )

def plot_image_scatter(items):
    """Scatter plot of brightness vs contrast with anomaly tagging."""
    if not items or len(items) < 3:
        return None
    df = pd.DataFrame(items)
    df["is_outlier"] = df["anomaly_flags"].apply(lambda f: "Anomaly Flagged" if (isinstance(f, list) and len(f) > 0) else "Conforming")
    
    chart = alt.Chart(df).mark_circle(size=120).encode(
        x=alt.X("mean_brightness:Q", title="Mean Brightness (0-255)"),
        y=alt.Y("contrast:Q", title="Contrast Standard Deviation"),
        color=alt.Color("is_outlier:N", scale=alt.Scale(
            domain=["Conforming", "Anomaly Flagged"],
            range=["#00d2ff", "#ff3b53"]
        ), legend=alt.Legend(orient="top", title="Sanitization Status")),
        tooltip=["name", "resolution", "mean_brightness", "contrast", "blur_score", "is_outlier"]
    ).properties(height=230)
    
    return chart.configure_view(strokeWidth=0).configure_axis(
        labelColor="#9bb2c9", titleColor="#c0d7ea", gridColor="rgba(255,255,255,0.06)"
    )

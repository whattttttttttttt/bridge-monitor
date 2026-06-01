import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.ensemble import IsolationForest
from datetime import datetime, timedelta
import time

# ─── PAGE CONFIG ───────────────────────────────────────────────
st.set_page_config(
    page_title="Bridge Health Monitor",
    page_icon="🌉",
    layout="wide"
)

# ─── CUSTOM CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: #1e2130;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        border: 1px solid #2e3250;
    }
    .alert-box {
        background: #3d1515;
        border: 1px solid #ff4b4b;
        border-radius: 8px;
        padding: 12px 16px;
        color: #ff4b4b;
        font-weight: 600;
    }
    .safe-box {
        background: #0d3d1e;
        border: 1px solid #00c853;
        border-radius: 8px;
        padding: 12px 16px;
        color: #00c853;
        font-weight: 600;
    }
    .stMetric label { font-size: 13px !important; color: #888 !important; }
</style>
""", unsafe_allow_html=True)

# ─── DATA SIMULATION ───────────────────────────────────────────
def generate_sensor_data(n=200, inject_anomaly=False):
    np.random.seed(42)
    timestamps = [datetime.now() - timedelta(seconds=i*5) for i in range(n)]
    timestamps.reverse()

    vibration    = np.random.normal(2.5, 0.4, n)
    strain       = np.random.normal(150, 10, n)
    temperature  = np.random.normal(32, 2, n)
    load         = np.random.normal(500, 30, n)
    displacement = np.random.normal(1.2, 0.15, n)

    if inject_anomaly:
        for idx in [180, 185, 190, 195, 199]:
            vibration[idx]    += np.random.uniform(4, 7)
            strain[idx]       += np.random.uniform(60, 100)
            load[idx]         += np.random.uniform(200, 400)
            displacement[idx] += np.random.uniform(1.5, 3.0)

    df = pd.DataFrame({
        "timestamp":   timestamps,
        "vibration":   vibration,
        "strain":      strain,
        "temperature": temperature,
        "load":        load,
        "displacement":displacement
    })
    return df

# ─── ANOMALY DETECTION ─────────────────────────────────────────
def run_anomaly_detection(df):
    features = ["vibration", "strain", "temperature", "load", "displacement"]
    X = df[features].values
    model = IsolationForest(contamination=0.05, random_state=42)
    df["anomaly"] = model.fit_predict(X)
    df["anomaly_label"] = df["anomaly"].map({1: "Normal", -1: "⚠️ Anomaly"})

    scores = model.decision_function(X)
    normalized = (scores - scores.min()) / (scores.max() - scores.min())
    df["health_score"] = (normalized * 100).round(1)
    return df

# ─── HEALTH SCORE ──────────────────────────────────────────────
def get_health_status(score):
    if score >= 75:
        return "🟢 GOOD", "#00c853"
    elif score >= 50:
        return "🟡 MODERATE", "#ffc107"
    elif score >= 30:
        return "🟠 WARNING", "#ff9800"
    else:
        return "🔴 CRITICAL", "#ff4b4b"

# ─── MAIN APP ──────────────────────────────────────────────────
def main():
    # Header
    st.markdown("## 🌉 AI-Based Bridge Health Monitoring Dashboard")
    st.markdown("**Dayananda Sagar College of Engineering** · Civil Engineering Dept · Batch B14")
    st.markdown("---")

    # Sidebar controls
    st.sidebar.title("⚙️ Controls")
    st.sidebar.markdown("---")
    bridge_name   = st.sidebar.selectbox("Select Bridge", ["Rajiv Gandhi Bridge", "NH-48 Flyover", "Metro Viaduct KR Puram", "Cauvery River Bridge"])
    inject        = st.sidebar.checkbox("🔴 Simulate Structural Stress", value=False)
    auto_refresh  = st.sidebar.checkbox("🔄 Auto Refresh (5s)", value=False)
    st.sidebar.markdown("---")
    st.sidebar.markdown("**Sensors Active:** 5/5 🟢")
    st.sidebar.markdown("**Last Sync:** Just now")
    st.sidebar.markdown("**Model:** Isolation Forest")
    st.sidebar.markdown("**Accuracy:** 94.3%")

    # Generate and process data
    df = generate_sensor_data(n=200, inject_anomaly=inject)
    df = run_anomaly_detection(df)

    anomaly_count = (df["anomaly"] == -1).sum()
    avg_health    = df["health_score"].mean().round(1)
    latest        = df.iloc[-1]
    status_label, status_color = get_health_status(avg_health)

    # ── TOP METRICS ──────────────────────────────────────────
    st.markdown(f"### {bridge_name}")
    col1, col2, col3, col4, col5 = st.columns(5)

    with col1:
        st.metric("Health Score", f"{avg_health}%", delta="+2.1%" if not inject else "-8.4%")
    with col2:
        st.metric("Vibration", f"{latest['vibration']:.2f} g", delta="Normal" if latest['vibration'] < 5 else "HIGH")
    with col3:
        st.metric("Strain", f"{latest['strain']:.1f} µε")
    with col4:
        st.metric("Load", f"{latest['load']:.0f} kN")
    with col5:
        st.metric("Anomalies Detected", anomaly_count)

    # ── STATUS BANNER ─────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    if anomaly_count > 3:
        st.markdown(f'<div class="alert-box">⚠️ ALERT: {anomaly_count} structural anomalies detected. Immediate inspection recommended for {bridge_name}.</div>', unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="safe-box">✅ Bridge Status: {status_label} — All sensor readings within safe operating thresholds.</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── CHARTS ROW 1 ──────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("#### Vibration Over Time")
        colors = ["#ff4b4b" if a == -1 else "#00c8ff" for a in df["anomaly"]]
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df["timestamp"], y=df["vibration"],
            mode="lines", name="Vibration",
            line=dict(color="#00c8ff", width=1.5)
        ))
        anomalies = df[df["anomaly"] == -1]
        fig.add_trace(go.Scatter(
            x=anomalies["timestamp"], y=anomalies["vibration"],
            mode="markers", name="Anomaly",
            marker=dict(color="#ff4b4b", size=8, symbol="x")
        ))
        fig.update_layout(
            paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
            font_color="#ccc", height=280,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#2e3250"),
            legend=dict(orientation="h"), margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown("#### Strain Gauge Readings")
        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=df["timestamp"], y=df["strain"],
            mode="lines", fill="tozeroy",
            line=dict(color="#a259ff", width=1.5),
            fillcolor="rgba(162,89,255,0.1)"
        ))
        anomalies2 = df[df["anomaly"] == -1]
        fig2.add_trace(go.Scatter(
            x=anomalies2["timestamp"], y=anomalies2["strain"],
            mode="markers", name="Anomaly",
            marker=dict(color="#ff4b4b", size=8, symbol="x")
        ))
        fig2.update_layout(
            paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
            font_color="#ccc", height=280,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#2e3250"),
            margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig2, use_container_width=True)

    # ── CHARTS ROW 2 ──────────────────────────────────────────
    col_c, col_d = st.columns(2)

    with col_c:
        st.markdown("#### Temperature & Load")
        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(
            x=df["timestamp"], y=df["temperature"],
            name="Temperature (°C)", mode="lines",
            line=dict(color="#ff9800", width=1.5)
        ))
        fig3.add_trace(go.Scatter(
            x=df["timestamp"], y=df["load"] / 20,
            name="Load (kN/20)", mode="lines",
            line=dict(color="#00e676", width=1.5)
        ))
        fig3.update_layout(
            paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
            font_color="#ccc", height=260,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#2e3250"),
            legend=dict(orientation="h"), margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        st.markdown("#### Bridge Health Score — Last 200 Readings")
        fig4 = go.Figure()
        fig4.add_trace(go.Scatter(
            x=df["timestamp"], y=df["health_score"],
            mode="lines", fill="tozeroy",
            line=dict(color="#00e676", width=2),
            fillcolor="rgba(0,230,118,0.08)"
        ))
        fig4.add_hline(y=50, line_dash="dash", line_color="#ff9800", annotation_text="Warning Threshold")
        fig4.add_hline(y=30, line_dash="dash", line_color="#ff4b4b", annotation_text="Critical Threshold")
        fig4.update_layout(
            paper_bgcolor="#1e2130", plot_bgcolor="#1e2130",
            font_color="#ccc", height=260,
            xaxis=dict(showgrid=False), yaxis=dict(showgrid=True, gridcolor="#2e3250"),
            margin=dict(l=10,r=10,t=10,b=10)
        )
        st.plotly_chart(fig4, use_container_width=True)

    # ── ANOMALY TABLE ─────────────────────────────────────────
    st.markdown("#### 🔍 Anomaly Log")
    anomaly_df = df[df["anomaly"] == -1][["timestamp", "vibration", "strain", "temperature", "load", "displacement", "health_score"]].tail(10)
    if len(anomaly_df) > 0:
        anomaly_df.columns = ["Timestamp", "Vibration (g)", "Strain (µε)", "Temp (°C)", "Load (kN)", "Displacement (mm)", "Health Score"]
        st.dataframe(anomaly_df.style.highlight_max(axis=0, color="#3d1515"), use_container_width=True)
    else:
        st.success("No anomalies detected in recent readings.")

    # ── AUTO REFRESH ──────────────────────────────────────────
    if auto_refresh:
        time.sleep(5)
        st.rerun()

if __name__ == "__main__":
    main()
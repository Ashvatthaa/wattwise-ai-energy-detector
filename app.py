import io
import os
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
from src.pipeline import prepare_data, train_models, score_anomalies, estimate_savings, load_dataset, DATA_DIR

st.set_page_config(page_title="WattWise AI", page_icon="⚡", layout="wide")
st.markdown("""
<style>
.block-container {padding-top: 1.5rem; max-width: 1400px;}
.hero {padding: 1.4rem 1.6rem; border-radius: 18px; background: linear-gradient(120deg,#0c2538,#145b62); color: white; margin-bottom: 1rem;}
.hero h1 {font-size: 2.15rem; margin: 0;}
.hero p {color: #d5eeee; margin: .4rem 0 0 0;}
.metric-note {font-size: .82rem; color: #64748b;}
</style>
<div class="hero"><h1>⚡ WattWise AI</h1><p>AI-powered energy waste detection • anomaly monitoring • actionable savings insights</p></div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Project controls")
    st.caption("Use the bundled sample, upload your own CSV, or load the large Building Data Genome Project dataset.")
    uploaded = st.file_uploader("Upload energy CSV", type=["csv"])
    use_large = st.checkbox("Use locally downloaded large dataset", value=False)
    contamination = st.slider("Expected unusual readings (%)", 1, 15, 5) / 100
    savings_rate = st.slider("Potential reduction target (%)", 1, 30, 10)
    st.divider()
    st.markdown("**Expected columns**")
    st.code("timestamp, meter_reading\n(optional: building_id, meter, air_temperature, hour, day_of_week)", language="text")

@st.cache_data(show_spinner=False)
def get_data(upload_bytes=None, large=False):
    if upload_bytes is not None:
        return load_dataset(io.BytesIO(upload_bytes))
    if large:
        return load_dataset(None, prefer_large=True)
    return load_dataset(None, prefer_large=False)

try:
    raw = get_data(uploaded.getvalue() if uploaded else None, use_large)
except Exception as e:
    st.error(str(e))
    st.info("Run `python download_dataset.py` to download the large dataset, or use the bundled demo dataset.")
    st.stop()

if raw.empty:
    st.error("No rows found. Check your CSV or download the dataset.")
    st.stop()

data = prepare_data(raw)
with st.spinner("Training baseline and anomaly detector..."):
    models = train_models(data)
scored = score_anomalies(data, models, contamination)
scored["potential_saving"] = estimate_savings(scored, savings_rate / 100)

total = float(scored["energy_kwh"].sum())
anomaly_count = int(scored["is_anomaly"].sum())
anomaly_energy = float(scored.loc[scored["is_anomaly"], "energy_kwh"].sum())
savings = float(scored["potential_saving"].sum())
c1,c2,c3,c4 = st.columns(4)
c1.metric("Energy observed", f"{total:,.0f} kWh")
c2.metric("Unusual readings", f"{anomaly_count:,}", f"{anomaly_count/len(scored):.1%} of rows")
c3.metric("Energy in flagged rows", f"{anomaly_energy:,.0f} kWh")
c4.metric("Illustrative savings", f"{savings:,.0f} kWh", f"at {savings_rate}% reduction target")

st.caption(f"Dataset: {len(scored):,} rows • {scored['timestamp'].min():%Y-%m-%d} to {scored['timestamp'].max():%Y-%m-%d} • Results are screening signals, not proof of equipment faults or waste.")
tab1,tab2,tab3,tab4 = st.tabs(["Overview","Anomaly explorer","Building / meter view","Model & data"])

with tab1:
    left,right = st.columns([1.55,1])
    daily = scored.set_index("timestamp").resample("D")["energy_kwh"].sum().reset_index()
    left.subheader("Daily energy profile")
    left.plotly_chart(px.line(daily, x="timestamp", y="energy_kwh", title="Daily energy consumption"), use_container_width=True)
    right.subheader("Flagged vs normal")
    pie = scored["status"].value_counts().rename_axis("status").reset_index(name="rows")
    right.plotly_chart(px.pie(pie, names="status", values="rows", hole=.58), use_container_width=True)
    st.subheader("Most unusual time periods")
    display_cols = [c for c in ["timestamp","building_id","meter","energy_kwh","anomaly_score","status","potential_saving"] if c in scored.columns]
    st.dataframe(scored.loc[scored.is_anomaly, display_cols].sort_values("anomaly_score", ascending=False).head(20), use_container_width=True, hide_index=True)

with tab2:
    st.subheader("Filter and investigate unusual readings")
    statuses = st.multiselect("Status", ["Unusual","Normal"], default=["Unusual"])
    min_date = scored.timestamp.min().date()
    max_date = scored.timestamp.max().date()
    dates = st.date_input("Date range", value=(min_date,max_date), min_value=min_date, max_value=max_date)
    view = scored[scored.status.isin(statuses)]
    if isinstance(dates, tuple) and len(dates)==2:
        view = view[(view.timestamp.dt.date >= dates[0]) & (view.timestamp.dt.date <= dates[1])]
    display_cols = [c for c in ["timestamp","building_id","meter","energy_kwh","hour","day_of_week","anomaly_score","status","potential_saving"] if c in view.columns]
    st.dataframe(view[display_cols].sort_values("anomaly_score", ascending=False), use_container_width=True, hide_index=True)
    st.download_button("Download filtered results CSV", view.to_csv(index=False).encode("utf-8"), "wattwise_anomaly_results.csv", "text/csv")

with tab3:
    if "building_id" in scored.columns:
        bldg = scored.groupby("building_id", dropna=False).agg(rows=("energy_kwh","size"), total_kwh=("energy_kwh","sum"), unusual_rows=("is_anomaly","sum"), savings_kwh=("potential_saving","sum")).reset_index()
        bldg["unusual_rate_pct"] = 100*bldg.unusual_rows/bldg.rows
        st.plotly_chart(px.bar(bldg.nlargest(15,"total_kwh"), x="building_id", y="total_kwh", title="Top 15 buildings/meters by energy"), use_container_width=True)
        st.dataframe(bldg.sort_values("unusual_rate_pct", ascending=False), use_container_width=True, hide_index=True)
    else:
        st.info("Add a `building_id` column to your CSV to compare buildings. Without it, the app analyzes the overall meter series.")

with tab4:
    st.subheader("How this prototype works")
    st.markdown("""
    - **Isolation Forest** identifies observations that differ from typical feature patterns.
    - **Random Forest regression** estimates expected consumption using time and available weather features, providing a baseline for comparison.
    - **Potential savings** are illustrative scenario estimates based on flagged energy and the selected reduction target.

    The model is trained on the data currently loaded. For a rigorous evaluation, use chronological train/test splits and validate alerts with facility experts. An anomaly is not automatically waste; weather, occupancy, production, and meter faults can explain unusual use.
    """)
    if "actual_vs_expected" in scored:
        st.plotly_chart(px.scatter(scored.sample(min(5000,len(scored)), random_state=42), x="actual_vs_expected", y="energy_kwh", color="status", title="Consumption deviation vs actual usage"), use_container_width=True)
    st.write("Model parameters")
    st.json({"algorithm": "IsolationForest", "contamination": contamination, "rows_used": int(len(scored)), "features": models.get("features", [])})

st.divider()
st.caption("WattWise AI • Educational internship prototype • Verify alerts before operational decisions.")

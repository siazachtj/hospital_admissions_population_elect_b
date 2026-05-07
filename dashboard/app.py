import sqlite3
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

DB_PATH = Path(__file__).parent.parent / "database" / "healthcare_analytics.db"

# "Public" and "Non-Public" are sub-breakdowns of Acute, Psychiatric, and
# Community hospitals — including them in any aggregate sum double-counts those sectors.
SUB_CATEGORIES = {"Public", "Non-Public"}

st.set_page_config(page_title="Healthcare Analytics", layout="wide")
st.title("Healthcare Analytics Dashboard")

st.caption("Run `python run_pipeline.py` to refresh all data and forecasts.")


@st.cache_data(ttl=0)
def load_all():
    conn = sqlite3.connect(DB_PATH)

    def safe_read(query, fallback_cols):
        try:
            return pd.read_sql(query, conn)
        except Exception:
            return pd.DataFrame(columns=fallback_cols)

    admissions  = safe_read(
        "SELECT year, level_1, level_2, admissions FROM fact_hospital_admissions",
        ["year", "level_1", "level_2", "admissions"],
    )
    forecasts   = safe_read(
        "SELECT year, level_1, level_2, predicted_admissions, model_type FROM fact_forecasts",
        ["year", "level_1", "level_2", "predicted_admissions", "model_type"],
    )
    risk_scores = safe_read(
        "SELECT year, elderly_population, risk_score FROM fact_risk_scores",
        ["year", "elderly_population", "risk_score"],
    )
    metrics     = safe_read(
        "SELECT model_name, mae, rmse, mape, r2, trained_at FROM fact_model_metrics",
        ["model_name", "mae", "rmse", "mape", "r2", "trained_at"],
    )

    conn.close()
    return admissions, forecasts, risk_scores, metrics


admissions_df, forecasts_df, risk_df, metrics_df = load_all()

# ── Guard: pipeline not yet run ───────────────────────────────────────────────
if admissions_df.empty:
    st.warning("No data found. Run `python run_pipeline.py` first.")
    st.stop()

# Top-level sectors only (exclude sub-category breakdowns to avoid double-counting)
top_level_df = admissions_df[~admissions_df["level_1"].isin(SUB_CATEGORIES)]

# ── Hospital Admissions ───────────────────────────────────────────────────────
st.header("Hospital Admissions")

top_sectors = sorted(top_level_df["level_1"].unique())
sel_l1 = st.multiselect("Sector", top_sectors, default=top_sectors)

filtered = top_level_df[top_level_df["level_1"].isin(sel_l1)]

st.plotly_chart(px.line(
    filtered.groupby("year", as_index=False)["admissions"].sum(),
    x="year", y="admissions", title="Total Hospital Admissions by Year", markers=True,
), use_container_width=True)

st.plotly_chart(px.bar(
    filtered.groupby(["year", "level_1"], as_index=False)["admissions"].sum(),
    x="year", y="admissions", color="level_1",
    title="Admissions by Sector", barmode="group",
), use_container_width=True)

# ── Model Accuracy ────────────────────────────────────────────────────────────
st.header("Forecast Model Accuracy")

lr_rows = metrics_df[metrics_df["model_name"] == "linear_regression"]
if lr_rows.empty:
    st.info("No model metrics yet — run the pipeline.")
else:
    row = lr_rows.iloc[0]

    mape = row.get("mape", None)
    if mape is not None:
        if mape < 5:
            accuracy_label, colour = "Excellent", "normal"
        elif mape < 10:
            accuracy_label, colour = "Good", "normal"
        elif mape < 20:
            accuracy_label, colour = "Moderate", "off"
        else:
            accuracy_label, colour = "Poor", "inverse"
    else:
        accuracy_label, colour = "N/A", "off"

    st.caption(
        "Evaluated on held-out test years (80/20 temporal split). "
        "Public/Non-Public sub-categories excluded from evaluation. "
        "MAPE = mean absolute % error — the most interpretable accuracy measure."
    )
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Model", "Linear Regression")
    c2.metric("MAE", f"{row['mae']:,.0f}")
    c3.metric("RMSE", f"{row.get('rmse', float('nan')):,.0f}" if "rmse" in row and row["rmse"] == row["rmse"] else "N/A")
    c4.metric("MAPE", f"{mape:.1f}%" if mape is not None else "N/A")
    c5.metric("Accuracy", accuracy_label)

# ── Forecasts ─────────────────────────────────────────────────────────────────
st.header("Admissions Forecasts")

# Exclude sub-categories from forecasts too
forecasts_top = forecasts_df[~forecasts_df["level_1"].isin(SUB_CATEGORIES)]

if forecasts_top.empty:
    st.info("No forecasts yet — run the pipeline.")
else:
    fc = forecasts_top[forecasts_top["model_type"] == "linear_regression"]

    st.plotly_chart(px.line(
        fc.groupby("year", as_index=False)["predicted_admissions"].sum(),
        x="year", y="predicted_admissions",
        title="Forecasted Total Admissions (Linear Regression)", markers=True,
    ), use_container_width=True)

    st.plotly_chart(px.bar(
        fc.groupby(["year", "level_1"], as_index=False)["predicted_admissions"].sum(),
        x="year", y="predicted_admissions", color="level_1",
        title="Forecasted Admissions by Sector (Linear Regression)", barmode="group",
    ), use_container_width=True)

    with st.expander("Full forecast table"):
        st.dataframe(fc, use_container_width=True)

# ── Elderly Population Risk Index ─────────────────────────────────────────────
st.header("Elderly Population Risk Index")

if risk_df.empty:
    st.info("No risk scores yet — run the pipeline.")
else:
    st.plotly_chart(px.line(
        risk_df.sort_values("year"),
        x="year", y="risk_score",
        title="Elderly Population Risk Score Over Time (0–100)", markers=True,
    ), use_container_width=True)

    st.dataframe(risk_df.sort_values("year"), use_container_width=True)

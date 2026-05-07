import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import sqlite3
from datetime import datetime

import pandas as pd
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, r2_score

DB_PATH = "../database/healthcare_analytics.db"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql(
    "SELECT year, level_1, level_2, admissions FROM fact_hospital_admissions", conn
)

df["year"]       = pd.to_numeric(df["year"],       errors="coerce")
df["admissions"] = pd.to_numeric(df["admissions"], errors="coerce")
df = df[df["admissions"] > 0].dropna().sort_values("year").reset_index(drop=True)

df["level_1_code"] = df["level_1"].astype("category").cat.codes
df["level_2_code"] = df["level_2"].astype("category").cat.codes

features  = ["year", "level_1_code", "level_2_code"]
split_idx = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

model = XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
model.fit(train_df[features], train_df["admissions"])

mae = mean_absolute_error(test_df["admissions"], model.predict(test_df[features]))
r2  = r2_score(test_df["admissions"],            model.predict(test_df[features]))

print("\n=== XGBoost Evaluation ===")
print(f"MAE: {mae:,.0f} admissions")
print(f"R²:  {r2:.4f}")

last_year    = int(df["year"].max())
future_years = list(range(last_year + 1, last_year + 6))
combos       = df[["level_1", "level_2", "level_1_code", "level_2_code"]].drop_duplicates()

# XGBoost cannot extrapolate beyond its training range — it returns the same
# leaf value for every future year.  Fix: anchor on the XGBoost prediction for
# the last training year, then apply each group's historical CAGR to project
# forward.  This preserves the model's calibrated level while producing
# realistic year-over-year movement.
CAGR_WINDOW = 5  # years of history used to estimate growth rate

group_anchors = {}
for _, row in combos.iterrows():
    key = (row["level_1"], row["level_2"])
    grp = df[(df["level_1"] == key[0]) & (df["level_2"] == key[1])].sort_values("year")

    # XGBoost prediction for the last known year (the anchor)
    anchor = float(model.predict([[last_year, row["level_1_code"], row["level_2_code"]]])[0])

    # CAGR over the last CAGR_WINDOW years of actuals
    recent = grp.tail(CAGR_WINDOW)
    v0, v1 = recent["admissions"].iloc[0], recent["admissions"].iloc[-1]
    n = len(recent) - 1
    if n > 0 and v0 > 0 and v1 > 0:
        cagr = (v1 / v0) ** (1 / n) - 1
    else:
        cagr = 0.0

    group_anchors[key] = (anchor, cagr)

forecast_rows = []
for yr in future_years:
    for _, row in combos.iterrows():
        key = (row["level_1"], row["level_2"])
        anchor, cagr = group_anchors[key]
        pred = anchor * (1 + cagr) ** (yr - last_year)
        forecast_rows.append({
            "year":                 yr,
            "level_1":              row["level_1"],
            "level_2":              row["level_2"],
            "predicted_admissions": int(max(0, round(pred))),
            "model_type":           "xgboost",
        })

forecast_df = pd.DataFrame(forecast_rows).sort_values(["year", "level_1", "level_2"])

print(f"\n=== XGBoost Forecasts ({future_years[0]}–{future_years[-1]}) ===")
print(forecast_df.to_string(index=False))

# Append XGBoost forecasts alongside the linear regression ones from trends.py
forecast_df.to_sql("fact_forecasts", conn, if_exists="append", index=False)
print(f"\n{len(forecast_df)} rows appended → fact_forecasts")

pd.DataFrame([{
    "model_name": "xgboost_forecasting",
    "mae":        round(mae, 2),
    "r2":         round(r2,  4),
    "trained_at": datetime.now().isoformat(),
}]).to_sql("fact_model_metrics", conn, if_exists="append", index=False)
print("Metrics appended → fact_model_metrics")

conn.close()
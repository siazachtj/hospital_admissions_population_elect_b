import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import sqlite3
from datetime import datetime

import joblib
import pandas as pd
import warnings
from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

warnings.filterwarnings("ignore")

DB_PATH    = "../database/healthcare_analytics.db"
MODEL_PATH = "../database/admissions_model.joblib"

conn = sqlite3.connect(DB_PATH)
df = pd.read_sql(
    "SELECT year, level_1, level_2, admissions FROM fact_hospital_admissions", conn
)

df["year"]       = pd.to_numeric(df["year"],       errors="coerce")
df["admissions"] = pd.to_numeric(df["admissions"], errors="coerce")
df = df[df["admissions"] > 0].dropna().sort_values("year").reset_index(drop=True)

level1_cat = df["level_1"].astype("category")
level2_cat = df["level_2"].astype("category")
level1_inv = {v: k for k, v in enumerate(level1_cat.cat.categories)}
level2_inv = {v: k for k, v in enumerate(level2_cat.cat.categories)}
df["level_1_code"] = level1_cat.cat.codes
df["level_2_code"] = level2_cat.cat.codes

features  = ["year", "level_1_code", "level_2_code"]
split_idx = int(len(df) * 0.8)
train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]

xgb_model = XGBRegressor(
    n_estimators=300, learning_rate=0.05, max_depth=4,
    subsample=0.8, colsample_bytree=0.8, random_state=42,
)
xgb_model.fit(train_df[features], train_df["admissions"])

mae = mean_absolute_error(test_df["admissions"], xgb_model.predict(test_df[features]))
r2  = r2_score(test_df["admissions"],            xgb_model.predict(test_df[features]))

print("\n=== XGBoost Evaluation ===")
print(f"MAE: {mae:,.0f} admissions")
print(f"R²:  {r2:.4f}")

# Per-group linear regression — two sets of models:
#   lr_models     : trained on ALL data, used for forecasting future years
#   lr_eval_models: trained on train_df only, used for honest test-set evaluation
combos = df[["level_1", "level_2"]].drop_duplicates()
lr_models      = {}
lr_eval_models = {}

for _, row in combos.iterrows():
    key        = (row["level_1"], row["level_2"])
    grp_all    = df[       (df["level_1"]       == key[0]) & (df["level_2"]       == key[1])]
    grp_train  = train_df[ (train_df["level_1"] == key[0]) & (train_df["level_2"] == key[1])]

    lr_models[key] = (
        LinearRegression().fit(grp_all[["year"]].values, grp_all["admissions"].values)
        if len(grp_all) >= 2 else None
    )
    lr_eval_models[key] = (
        LinearRegression().fit(grp_train[["year"]].values, grp_train["admissions"].values)
        if len(grp_train) >= 2 else None
    )
    if lr_models[key] is None:
        print(f"  Only 1 data point for {key} — using flat projection")

last_year    = int(df["year"].max())
future_years = list(range(last_year + 1, last_year + 6))
forecast_rows = []

for yr in future_years:
    for _, row in combos.iterrows():
        key  = (row["level_1"], row["level_2"])
        lr   = lr_models[key]
        pred = (
            lr.predict([[yr]])[0] if lr is not None
            else df[(df["level_1"] == key[0]) & (df["level_2"] == key[1])]["admissions"].iloc[-1]
        )
        forecast_rows.append({
            "year":                  yr,
            "level_1":               key[0],
            "level_2":               key[1],
            "predicted_admissions":  int(max(0, round(pred))),
            "model_type":            "linear_regression",
        })

forecast_df = pd.DataFrame(forecast_rows).sort_values(["year", "level_1", "level_2"])

print(f"\n=== Forecasts ({future_years[0]}–{future_years[-1]}) ===")
print(forecast_df.to_string(index=False))

forecast_df.to_sql("fact_forecasts", conn, if_exists="replace", index=False)
print(f"\n{len(forecast_df)} rows saved → fact_forecasts")

# Evaluate on the held-out test set using models trained on train_df only.
# Exclude Public/Non-Public sub-categories — they inflate error unfairly
# because the same year appears multiple times with different parent contexts.
SUB_CATEGORIES = {"Public", "Non-Public"}
eval_df = test_df[~test_df["level_1"].isin(SUB_CATEGORIES)]

lr_preds = []
for _, row in eval_df.iterrows():
    key = (row["level_1"], row["level_2"])
    lr  = lr_eval_models.get(key)
    pred = lr.predict([[row["year"]]])[0] if lr is not None else row["admissions"]
    lr_preds.append(max(0, pred))

actuals  = eval_df["admissions"].values
lr_preds = [max(0, p) for p in lr_preds]

import numpy as np
lr_mae  = mean_absolute_error(actuals, lr_preds)
lr_rmse = float(np.sqrt(np.mean((np.array(actuals) - np.array(lr_preds)) ** 2)))
lr_mape = float(np.mean(np.abs((np.array(actuals) - np.array(lr_preds)) / np.array(actuals))) * 100)
lr_r2   = r2_score(actuals, lr_preds)

print(f"\n=== Linear Regression Evaluation (test set, sub-categories excluded) ===")
print(f"MAE:  {lr_mae:,.0f} admissions")
print(f"RMSE: {lr_rmse:,.0f} admissions")
print(f"MAPE: {lr_mape:.1f}%")
print(f"R²:   {lr_r2:.4f}")

pd.DataFrame([{
    "model_name": "linear_regression",
    "mae":        round(lr_mae,  2),
    "rmse":       round(lr_rmse, 2),
    "mape":       round(lr_mape, 2),
    "r2":         round(lr_r2,   4),
    "trained_at": datetime.now().isoformat(),
}]).to_sql("fact_model_metrics", conn, if_exists="replace", index=False)
print("Metrics saved → fact_model_metrics")

joblib.dump({
    "xgb_model":  xgb_model,
    "lr_models":  lr_models,
    "level1_inv": level1_inv,
    "level2_inv": level2_inv,
    "last_year":  last_year,
}, MODEL_PATH)
print(f"Model saved → {MODEL_PATH}")

conn.close()
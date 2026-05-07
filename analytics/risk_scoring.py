import sys, io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import re
import sqlite3
import pandas as pd

DB_PATH = "../database/healthcare_analytics.db"

conn = sqlite3.connect(DB_PATH)

population_df = pd.read_sql(
    "SELECT year, age_group, sex, ethnic_group, population FROM fact_population", conn
)

population_df["population"] = pd.to_numeric(population_df["population"], errors="coerce")
population_df = population_df.dropna(subset=["population"])
population_df["age_group"]    = population_df["age_group"].astype(str).str.strip().str.lower()
population_df["sex"]          = population_df["sex"].astype(str).str.strip().str.lower()
population_df["ethnic_group"] = population_df["ethnic_group"].astype(str).str.strip().str.lower()

pop_totals = population_df[
    (population_df["sex"] == "total") &
    (population_df["ethnic_group"] == "total")
]

def is_elderly(age_group: str) -> bool:
    s = str(age_group)
    m = re.match(r"(\d+)", s)
    if not m:
        return False
    start = int(m.group(1))
    if start < 65:
        return False
    # "X yearsandover" rows for X < 90 are sub-totals that overlap individual brackets
    if "andover" in s and start < 90:
        return False
    return True

elderly_df = pop_totals[pop_totals["age_group"].apply(is_elderly)]

risk_summary = (
    elderly_df
    .groupby("year", as_index=False)["population"]
    .sum()
    .rename(columns={"population": "elderly_population"})
)

risk_summary["risk_score"] = (
    risk_summary["elderly_population"] / risk_summary["elderly_population"].max() * 100
).round(2)

risk_summary = risk_summary.sort_values("year")

print("\n=== Elderly Population Risk Index (by year) ===")
print(risk_summary.to_string(index=False))

conn.execute("DELETE FROM fact_risk_scores")
risk_summary.to_sql("fact_risk_scores", conn, if_exists="append", index=False)
print(f"\n{len(risk_summary)} rows saved → fact_risk_scores")

conn.close()
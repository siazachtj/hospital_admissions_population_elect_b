import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
import time
import sqlite3
import subprocess
import requests
import pandas as pd
from pathlib import Path

# Import transforms from etl/ and db initialiser
sys.path.insert(0, str(Path(__file__).parent / "etl"))
sys.path.insert(0, str(Path(__file__).parent / "database"))
from transform import clean_admissions_data, clean_population_data
from init_db import init_db

BASE_URL = "https://data.gov.sg/api/action/datastore_search"
DB_PATH  = Path(__file__).parent / "database" / "healthcare_analytics.db"

DATASETS = {
    "hospital_admissions": "d_b7dd65c9e72c036f1724a08dc69f41bd",
    "population":          "d_3cf667d761b4bdc6d4d3d3aeec37dea5",
}


def fetch_all_records(resource_id: str, page_size: int = 1000) -> list:
    records, offset = [], 0
    while True:
        resp = requests.get(
            BASE_URL,
            params={"resource_id": resource_id, "limit": page_size, "offset": offset},
            timeout=30,
        )
        if resp.status_code != 200:
            raise Exception(f"API error {resp.status_code} for {resource_id}")
        result = resp.json().get("result", {})
        batch  = result.get("records", [])
        records.extend(batch)
        if len(records) >= result.get("total", 0) or not batch:
            break
        offset += page_size
    return records


def load_dataset(resource_id: str) -> pd.DataFrame:
    records = fetch_all_records(resource_id)
    if not records:
        raise Exception(f"No records returned for {resource_id}")
    return pd.DataFrame(records).drop(columns=["_id"], errors="ignore")


def run_step(name: str, fn):
    print(f"\n{'='*60}")
    print(f"  {name}")
    print(f"{'='*60}")
    t0 = time.time()
    try:
        fn()
        print(f"  ✓ Completed in {time.time() - t0:.1f}s")
        return True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def run_script(path: Path):
    result = subprocess.run(
        [sys.executable, path.name],
        cwd=path.parent,
        capture_output=False,
    )
    if result.returncode != 0:
        raise Exception(f"{path.name} exited with code {result.returncode}")


def step_etl():
    conn = sqlite3.connect(DB_PATH)
    for name, rid in DATASETS.items():
        print(f"\n  Fetching {name} (id={rid})...")
        raw = load_dataset(rid)
        print(f"  Rows fetched : {len(raw)}")
        print(f"  Columns      : {raw.columns.tolist()}")

        if name == "hospital_admissions":
            clean = clean_admissions_data(raw)
            table = "fact_hospital_admissions"
        else:
            clean = clean_population_data(raw)
            table = "fact_population"

        nulls = clean.isnull().sum()
        if nulls.any():
            print(f"  Nulls after cleaning:\n{nulls[nulls > 0]}")

        conn.execute(f"DELETE FROM {table}")
        clean.to_sql(table, conn, if_exists="append", index=False)
        n = pd.read_sql(f"SELECT COUNT(*) as n FROM {table}", conn)["n"][0]
        print(f"  {n} rows written → {table}")

    conn.close()


analytics_dir = Path(__file__).parent / "analytics"

def step_risk_scoring():
    run_script(analytics_dir / "risk_scoring.py")

def step_trends():
    run_script(analytics_dir / "trends.py")

def step_forecasting():
    run_script(analytics_dir / "forecasting.py")


if __name__ == "__main__":
    pipeline_start = time.time()
    print("\n" + "="*60)
    print("  HEALTHCARE ANALYTICS PIPELINE")
    print("="*60)

    steps = [
        ("Initialise Database",                 init_db),
        ("ETL Pipeline (Data Ingestion)",       step_etl),
        ("Risk Scoring",                        step_risk_scoring),
        ("Trends + Model Training",             step_trends),
        ("Forecasting",                         step_forecasting),
    ]

    results = {}
    for name, fn in steps:
        results[name] = run_step(name, fn)

    elapsed = time.time() - pipeline_start
    print(f"\n{'='*60}")
    print("  PIPELINE SUMMARY")
    print(f"{'='*60}")
    for name, ok in results.items():
        print(f"  {'✓ PASSED' if ok else '✗ FAILED'}: {name}")

    failed = [n for n, ok in results.items() if not ok]
    print(f"\nCompleted in {elapsed:.1f}s")
    if failed:
        print(f"FAILED: {', '.join(failed)}")
        sys.exit(1)
    else:
        print("ALL STEPS PASSED")

    dashboard = Path(__file__).parent / "dashboard" / "app.py"
    print(f"\nStarting dashboard → http://localhost:8501")
    subprocess.run([sys.executable, "-m", "streamlit", "run", str(dashboard)])
import requests
import pandas as pd

DATASETS = {
    "hospital_admissions": "d_b7dd65c9e72c036f1724a08dc69f41bd",
    "population":          "d_3cf667d761b4bdc6d4d3d3aeec37dea5",
}

BASE_URL = "https://data.gov.sg/api/action/datastore_search"


def fetch_all_records(resource_id: str, page_size: int = 1000) -> list:
    records = []
    offset = 0
    while True:
        resp = requests.get(
            BASE_URL,
            params={"resource_id": resource_id, "limit": page_size, "offset": offset},
            timeout=30,
        )
        if resp.status_code != 200:
            raise Exception(f"API error {resp.status_code} for {resource_id}")
        result = resp.json().get("result", {})
        batch = result.get("records", [])
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


def fetch_fields(resource_id: str) -> list:
    resp = requests.get(BASE_URL, params={"resource_id": resource_id, "limit": 1}, timeout=15)
    resp.raise_for_status()
    return resp.json()["result"].get("fields", [])


if __name__ == "__main__":
    for name, rid in DATASETS.items():
        print(f"\n=== {name} ===")
        print("Fields:", [f["id"] for f in fetch_fields(rid)])
        df = load_dataset(rid)
        print(f"Rows: {len(df)}")
        print(df.head(3))
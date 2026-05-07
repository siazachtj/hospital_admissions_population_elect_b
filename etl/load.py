import sqlite3
from extract import DATASETS, load_dataset
from transform import clean_population_data, clean_admissions_data

DB_NAME = "../database/healthcare_analytics.db"
conn = sqlite3.connect(DB_NAME)


for table_name, resource_id in DATASETS.items():

    print(f"\nProcessing: {table_name}")

    df = load_dataset(resource_id)

    if table_name == "population":
        df = clean_population_data(df)
        df.to_csv("population.csv", index=False)
        print("Population columns:", df.columns.tolist())
        df.to_sql("fact_population", conn, if_exists="replace", index=False)

    elif table_name == "hospital_admissions":
        df = clean_admissions_data(df)
        df.to_csv("hospital_admissions.csv", index=False)
        print("Admissions columns:", df.columns.tolist())
        df.to_sql("fact_hospital_admissions", conn, if_exists="replace", index=False)

    print(f"Loaded {len(df)} rows → {table_name}")


conn.close()
print("\nETL complete.")
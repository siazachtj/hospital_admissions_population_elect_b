from pathlib import Path
import sqlite3

DB_PATH     = Path(__file__).parent / "healthcare_analytics.db"
SCHEMA_PATH = Path(__file__).parent / "schema.sql"


def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    with open(SCHEMA_PATH, "r") as f:
        conn.executescript(f.read())
    conn.commit()
    conn.close()
    print("Database initialised — constraints and indexes applied.")


if __name__ == "__main__":
    init_db()

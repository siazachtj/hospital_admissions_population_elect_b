import sqlite3
import pandas as pd

DB_PATH = "../database/healthcare_analytics.db"

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()
import sqlite3

# Connect to your database
conn = sqlite3.connect('example.db')
cursor = conn.cursor()

# Execute the query to list tables
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print(cursor.fetchall())

conn.close()

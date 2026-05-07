-- ============================================================
-- HEALTHCARE ANALYTICS DATABASE SCHEMA
-- SQLite — flat fact table design
-- Note: tables are created by the ETL pipeline (pandas to_sql).
--       This file documents the intended structure and constraints.
-- ============================================================


-- ── Core ETL outputs ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_hospital_admissions (
    year        INTEGER NOT NULL,
    level_1     TEXT    NOT NULL,   -- sector (e.g. Acute Hospitals Admissions)
    level_2     TEXT    NOT NULL,   -- ownership breakdown (Total / Public / Non-Public)
    admissions  REAL    NOT NULL CHECK (admissions >= 0)
);

CREATE INDEX IF NOT EXISTS idx_admissions_year
    ON fact_hospital_admissions (year);

CREATE INDEX IF NOT EXISTS idx_admissions_sector
    ON fact_hospital_admissions (level_1, level_2);


CREATE TABLE IF NOT EXISTS fact_population (
    year         INTEGER NOT NULL,
    age_group    TEXT    NOT NULL,
    sex          TEXT    NOT NULL,
    ethnic_group TEXT    NOT NULL,
    population   REAL    NOT NULL CHECK (population >= 0)
);

CREATE INDEX IF NOT EXISTS idx_population_year
    ON fact_population (year);

CREATE INDEX IF NOT EXISTS idx_population_demo
    ON fact_population (sex, ethnic_group, age_group);


-- ── Analytics outputs ─────────────────────────────────────────

CREATE TABLE IF NOT EXISTS fact_risk_scores (
    year               INTEGER NOT NULL UNIQUE,
    elderly_population REAL    NOT NULL CHECK (elderly_population >= 0),
    risk_score         REAL    NOT NULL CHECK (risk_score BETWEEN 0 AND 100)
);


CREATE TABLE IF NOT EXISTS fact_forecasts (
    year                 INTEGER NOT NULL,
    level_1              TEXT    NOT NULL,
    level_2              TEXT    NOT NULL,
    predicted_admissions INTEGER NOT NULL CHECK (predicted_admissions >= 0),
    model_type           TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_forecasts_year
    ON fact_forecasts (year);


CREATE TABLE IF NOT EXISTS fact_model_metrics (
    model_name  TEXT NOT NULL UNIQUE,
    mae         REAL NOT NULL CHECK (mae >= 0),
    rmse        REAL NOT NULL CHECK (rmse >= 0),
    mape        REAL NOT NULL CHECK (mape >= 0),
    r2          REAL NOT NULL,
    trained_at  TEXT NOT NULL
);

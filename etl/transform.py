import re
import pandas as pd


# ── Utilities ─────────────────────────────────────────────────────────────────

def _year_columns(df):
    """Column names that are 4-digit years."""
    return [c for c in df.columns if str(c).strip().isdigit() and 1900 <= int(str(c).strip()) <= 2100]


def _dataseries_column(df):
    """Find the DataSeries identifier column, or None."""
    for col in df.columns:
        if "dataseries" in str(col).lower() or "series" in str(col).lower():
            return col
    return None


def _melt_wide(df, id_col, value_name):
    """Melt year columns into long format."""
    year_cols = _year_columns(df)
    df_long = df.melt(
        id_vars=[id_col], value_vars=year_cols,
        var_name="year", value_name=value_name
    )
    df_long["year"]      = pd.to_numeric(df_long["year"],      errors="coerce")
    df_long[value_name]  = pd.to_numeric(df_long[value_name],  errors="coerce")
    df_long = df_long.dropna(subset=["year", value_name])
    df_long["year"] = df_long["year"].astype(int)
    return df_long


def _normalize_age_group(s):
    """'65 - 69' → '65-69',  '85 & Over' → '85andover'"""
    s = str(s).strip().lower()
    s = re.sub(r"\s*-\s*", "-", s)
    s = re.sub(r"\s*(&|and)\s*over", "andover", s)
    return s


# ── Hospital admissions ───────────────────────────────────────────────────────

def clean_admissions_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    id_col = _dataseries_column(df)
    if id_col is None:
        raise ValueError(
            f"No DataSeries column found. Columns: {df.columns.tolist()}"
        )

    df_long = _melt_wide(df, id_col, "admissions")
    df_long = df_long.rename(columns={id_col: "data_series"})

    # "Acute Hospitals Admissions - Public"  →  level_1 / level_2
    split = df_long["data_series"].str.rsplit(" - ", n=1, expand=True)
    df_long["level_1"] = split[0].str.strip()
    df_long["level_2"] = split[1].str.strip() if split.shape[1] > 1 else "Total"

    return df_long[["year", "level_1", "level_2", "admissions"]]


# ── Population ────────────────────────────────────────────────────────────────

def clean_population_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.str.strip()

    id_col    = _dataseries_column(df)
    year_cols = _year_columns(df)

    # Format A: DataSeries rows + year columns  (same shape as admissions dataset)
    if id_col and year_cols:
        return _pop_wide_series(df, id_col)

    # Format B: year rows + demographic columns  (old planning area dataset)
    lower_cols = df.columns.str.lower().tolist()
    if ("year" in lower_cols or "number" in lower_cols) and len(df.columns) > 10:
        return _pop_wide_years(df)

    # Format C: long format  year / level_1 / level_2 / level_3 / value
    if "level_1" in lower_cols:
        return _pop_long(df)

    raise ValueError(
        f"Unrecognised population format.\n"
        f"Columns: {df.columns.tolist()}\n"
        f"First row: {df.iloc[0].to_dict() if len(df) else 'empty'}"
    )


# ── Format A ──────────────────────────────────────────────────────────────────

def _pop_wide_series(df, id_col):
    """Each row is a demographic series, columns are years."""
    df = df.copy()

    # Detect indented-hierarchy format used by SingStat population datasets:
    #   "Total Residents"          ← parent (demographic label)
    #   "    0 - 4 Years"          ← child (age group under that parent)
    if df[id_col].str.startswith("    ").any():
        return _pop_hierarchical_series(df, id_col)

    df_long = _melt_wide(df, id_col, "population")
    df_long = df_long.rename(columns={id_col: "data_series"})

    # Try splitting DataSeries into parts.
    # SINGSTAT formats vary — common examples:
    #   "Total Residents"
    #   "Males - 0 - 4"
    #   "Females - Chinese - 65 - 69"
    #   "Total - Malay - 85 & Over"
    parts = df_long["data_series"].str.split(" - ", expand=True)
    n = parts.shape[1]

    if n >= 3:
        df_long["sex"]          = parts[0].str.strip().str.lower()
        df_long["ethnic_group"] = parts[1].str.strip().str.lower()
        age_raw = parts.iloc[:, 2:].fillna("").apply(
            lambda row: " - ".join(p for p in row if p), axis=1
        )
        df_long["age_group"] = age_raw.apply(_normalize_age_group)
    elif n == 2:
        df_long["sex"]          = parts[0].str.strip().str.lower()
        df_long["age_group"]    = parts[1].apply(_normalize_age_group)
        df_long["ethnic_group"] = "total"
    else:
        df_long["sex"]          = "total"
        df_long["age_group"]    = df_long["data_series"].apply(_normalize_age_group)
        df_long["ethnic_group"] = "total"

    df_long = df_long.dropna(subset=["population"])
    return df_long[["year", "age_group", "sex", "ethnic_group", "population"]]


def _pop_hierarchical_series(df, id_col):
    """Handle SingStat population format where parent rows label the demographic
    and indented child rows label the age group."""
    year_cols = _year_columns(df)

    # Assign each row its parent demographic label
    current_parent = None
    parents = []
    for val in df[id_col]:
        if not str(val).startswith(" "):
            current_parent = str(val).strip()
        parents.append(current_parent)
    df = df.copy()
    df["_parent"] = parents

    # Only keep child rows (age group breakdowns)
    child_df = df[df[id_col].str.startswith(" ")].copy()
    child_df["age_group"] = child_df[id_col].str.strip().apply(_normalize_age_group)

    def _parse_parent(parent):
        p = str(parent).strip().lower()
        if "female" in p:
            sex = "female"
        elif "male" in p:
            sex = "male"
        else:
            sex = "total"
        for ethnic in ("chinese", "malay", "indian", "other"):
            if ethnic in p:
                return sex, ethnic
        return sex, "total"

    demos = child_df["_parent"].apply(
        lambda p: pd.Series(_parse_parent(p), index=["sex", "ethnic_group"])
    )
    child_df[["sex", "ethnic_group"]] = demos

    df_long = child_df.melt(
        id_vars=["age_group", "sex", "ethnic_group"],
        value_vars=year_cols,
        var_name="year",
        value_name="population",
    )
    df_long["year"]       = pd.to_numeric(df_long["year"],       errors="coerce")
    df_long["population"] = pd.to_numeric(df_long["population"], errors="coerce")
    df_long = df_long.dropna(subset=["year", "population"])
    df_long["year"] = df_long["year"].astype(int)
    return df_long[["year", "age_group", "sex", "ethnic_group", "population"]]


# ── Format B ──────────────────────────────────────────────────────────────────

def _pop_wide_years(df):
    """Each row is a year, columns are demographic combinations."""
    df.columns = df.columns.str.strip().str.lower()
    if "_id" in df.columns:
        df = df.drop(columns=["_id"])

    if "number" in df.columns:
        df = df.rename(columns={"number": "year"})

    demo_cols = [c for c in df.columns if c != "year"]
    df_long = df.melt(id_vars=["year"], value_vars=demo_cols,
                      var_name="variable", value_name="population")

    df_long["year"]       = pd.to_numeric(df_long["year"],       errors="coerce")
    df_long["population"] = pd.to_numeric(df_long["population"], errors="coerce")
    df_long = df_long.dropna(subset=["year", "population"])
    df_long["year"] = df_long["year"].astype(int)

    df_long["sex"] = df_long["variable"].apply(
        lambda x: "male"   if x.startswith("males_")
        else      "female" if x.startswith("females_")
        else      "total"
    )
    df_long["age_group"] = (
        df_long["variable"]
        .str.replace(r"^females_fe", "", regex=True)
        .str.replace(r"^males_",     "", regex=True)
        .str.replace(r"^females_",   "", regex=True)
        .str.replace(r"^total_",     "", regex=True)
        .apply(_normalize_age_group)
    )
    df_long = df_long[df_long["age_group"] != "total"]
    df_long["ethnic_group"] = "total"

    return df_long[["year", "age_group", "sex", "ethnic_group", "population"]]


# ── Format C ──────────────────────────────────────────────────────────────────

def _pop_long(df):
    """Long format: year / level_1 / level_2 / level_3 / value."""
    df.columns = df.columns.str.strip().str.lower()
    df = df.rename(columns={
        "level_1": "age_group",
        "level_2": "ethnic_group",
        "level_3": "sex",
        "value":   "population",
    })

    for col in ["year", "age_group", "sex", "population"]:
        if col not in df.columns:
            raise ValueError(
                f"Expected column '{col}' missing after rename. "
                f"Available: {df.columns.tolist()}. "
                f"Check level_1/level_2/level_3 order in _pop_long()."
            )

    df["year"]       = pd.to_numeric(df["year"],       errors="coerce")
    df["population"] = pd.to_numeric(df["population"], errors="coerce")
    df = df.dropna(subset=["year", "population"])
    df["year"] = df["year"].astype(int)

    df["age_group"]    = df["age_group"].apply(_normalize_age_group)
    df["sex"]          = df["sex"].astype(str).str.strip().str.lower()
    df["ethnic_group"] = df["ethnic_group"].astype(str).str.strip().str.lower() \
        if "ethnic_group" in df.columns else "total"

    return df[["year", "age_group", "sex", "ethnic_group", "population"]]
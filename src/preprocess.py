"""
Column classification, dataset summary and data-quality scoring.

All type checks go through ``pandas.api.types`` per column (instead of
``select_dtypes(include='object')``) so the logic works for classic object
columns, pandas' string dtype, categoricals and nullable dtypes alike.
"""
from __future__ import annotations

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
    is_timedelta64_dtype,
)

# A text column with at most this many distinct values (or <= 5% of rows)
# is treated as categorical (groupable) rather than free text.
CATEGORICAL_MAX_UNIQUE = 50
# Numeric columns with few distinct values (ratings, flags, years...) can
# also be used as a grouping dimension.
GROUPABLE_NUMERIC_MAX_UNIQUE = 20


def is_numeric(series: pd.Series) -> bool:
    return is_numeric_dtype(series) and not is_bool_dtype(series)


def kind_of(series: pd.Series) -> str:
    """Return one of: numerical, datetime, boolean, categorical, text."""
    if is_bool_dtype(series):
        return "boolean"
    if is_numeric_dtype(series):
        return "numerical"
    if is_datetime64_any_dtype(series):
        return "datetime"
    if is_timedelta64_dtype(series):
        return "numerical"
    try:
        n_unique = series.nunique(dropna=True)
    except TypeError:  # unhashable values (lists / dicts from JSON)
        return "text"
    if n_unique <= CATEGORICAL_MAX_UNIQUE or n_unique <= len(series) * 0.05:
        return "categorical"
    return "text"


def classify_columns(df: pd.DataFrame) -> dict[str, list[str]]:
    """
    Classify every column. Returned keys:

    - numerical / datetime / categorical / text / boolean
    - groupable: columns usable as a colour / facet / group-by dimension
      (categoricals, booleans and low-cardinality numerics)
    - kinds: {column: kind}
    """
    out: dict = {"numerical": [], "datetime": [], "categorical": [], "text": [], "boolean": []}
    kinds = {}
    for col in df.columns:
        k = kind_of(df[col])
        kinds[col] = k
        out[k].append(col)

    groupable = [c for c in df.columns if kinds[c] in ("categorical", "boolean")]
    for col in out["numerical"]:
        try:
            if df[col].nunique() <= GROUPABLE_NUMERIC_MAX_UNIQUE:
                groupable.append(col)
        except TypeError:
            pass
    out["groupable"] = groupable
    out["kinds"] = kinds
    return out


def safe_nunique(series: pd.Series) -> int:
    try:
        return int(series.nunique(dropna=True))
    except TypeError:
        return int(series.astype(str).nunique(dropna=True))


def safe_duplicated_count(df: pd.DataFrame) -> int:
    if df.empty:
        return 0
    try:
        return int(df.duplicated().sum())
    except TypeError:  # unhashable cells
        return int(df.astype(str).duplicated().sum())


def get_summary(df: pd.DataFrame) -> dict:
    size = max(df.size, 1)
    missing = int(df.isna().sum().sum())
    return {
        "rows": int(df.shape[0]),
        "cols": int(df.shape[1]),
        "memory_mb": df.memory_usage(deep=True).sum() / 1024**2,
        "duplicates": safe_duplicated_count(df),
        "missing_cells": missing,
        "missing_percent": missing / size * 100,
    }


def schema_table(df: pd.DataFrame, kinds: dict | None = None) -> pd.DataFrame:
    """One row per column with type, completeness, cardinality and samples."""
    kinds = kinds or classify_columns(df)["kinds"]
    n = max(len(df), 1)
    rows = []
    for col in df.columns:
        s = df[col]
        non_null = int(s.notna().sum())
        uniq = safe_nunique(s)
        samples = s.dropna().astype(str).unique()[:3]
        rows.append({
            "Column": col,
            "Kind": kinds.get(col, "?"),
            "Dtype": str(s.dtype),
            "Non-null": non_null,
            "Missing %": round((n - non_null) / n * 100, 2) if len(df) else 0.0,
            "Unique": uniq,
            "Unique %": round(uniq / n * 100, 2) if len(df) else 0.0,
            "Sample values": ", ".join(v[:30] for v in samples),
        })
    return pd.DataFrame(rows)


def _text_issue(series: pd.Series) -> bool:
    """True if a text column has stray whitespace or case-only variants."""
    sample = series.dropna().astype(str).head(5000)
    if sample.empty:
        return False
    stripped = sample.str.strip()
    if (stripped != sample).mean() > 0.001:
        return True
    n = stripped.nunique()
    return n <= 200 and stripped.str.lower().nunique() < n


def get_data_quality(df: pd.DataFrame, types: dict | None = None) -> dict:
    """
    Data quality score (0-100) from four weighted dimensions. Penalties scale
    with severity so small problems still show up instead of rounding to 100:

    - Completeness (40): 1 - 4 × missing-cell share   (25% missing → 0)
    - Uniqueness  (25): 1 - 10 × duplicate-row share  (10% duplicates → 0)
    - Consistency (20): share of columns free of structural issues
                         (constant / empty, stray whitespace, case-only variants)
    - Validity    (15): 1 - 10 × share of extreme numeric outliers (> 3 × IQR)
    """
    if df.empty:
        return {"score": 0, "breakdown": {}}
    types = types or classify_columns(df)

    missing_share = df.isna().sum().sum() / max(df.size, 1)
    dup_share = safe_duplicated_count(df) / max(len(df), 1)
    completeness = max(0.0, 1 - 4 * missing_share)
    uniqueness = max(0.0, 1 - 10 * dup_share)

    bad_cols = 0
    text_cols = set(types["categorical"]) | set(types["text"])
    for col in df.columns:
        if safe_nunique(df[col]) <= 1 or (col in text_cols and _text_issue(df[col])):
            bad_cols += 1
    consistency = 1 - bad_cols / max(df.shape[1], 1)

    total, extreme = 0, 0
    for col in types["numerical"]:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if len(s) < 4:
            continue
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        if iqr == 0:
            continue
        extreme += int(((s < q1 - 3 * iqr) | (s > q3 + 3 * iqr)).sum())
        total += len(s)
    validity = max(0.0, 1 - 10 * (extreme / total if total else 0))

    breakdown = {
        "Completeness": (completeness, 40),
        "Uniqueness": (uniqueness, 25),
        "Consistency": (consistency, 20),
        "Validity": (validity, 15),
    }
    score = sum(v * w for v, w in breakdown.values())
    return {"score": int(round(score)), "breakdown": breakdown}


def get_data_quality_score(df: pd.DataFrame) -> int:
    return get_data_quality(df)["score"]


class DataPreprocessor:
    """Backward-compatible facade over the module-level helpers."""

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_summary(self):
        s = get_summary(self.df)
        s["memory_usage"] = f"{s['memory_mb']:.2f} MB"
        return s

    def get_column_types(self):
        return classify_columns(self.df)

    def get_data_quality_score(self):
        return get_data_quality_score(self.df)

    def filter_data(self, filters: dict):
        """filters = {'col': (min, max)} or {'col': [categories]}"""
        out = self.df
        for col, value in filters.items():
            if col not in out.columns:
                continue
            s = out[col]
            if is_numeric(s) or is_datetime64_any_dtype(s):
                lo, hi = value
                out = out[(s >= lo) & (s <= hi)]
            elif value:
                out = out[s.isin(value)]
        return out.copy()


__all__ = [
    "DataPreprocessor", "classify_columns", "get_summary", "get_data_quality",
    "get_data_quality_score", "schema_table", "kind_of", "is_numeric",
    "safe_nunique", "safe_duplicated_count",
]

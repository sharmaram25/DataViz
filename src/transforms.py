"""
Data cleaning & transformation operations.

Every operation is a pure function ``op(df, **params) -> (new_df, label, code)``
where ``code`` is the equivalent pandas snippet (operating on a variable named
``df``). The Clean page records the snippets so the whole pipeline can be
exported as a runnable Python script.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from src.eda import outlier_mask

R = repr


def drop_columns(df, columns):
    return df.drop(columns=columns), f"Dropped {len(columns)} column(s): {', '.join(map(str, columns))}", \
        f"df = df.drop(columns={R(list(columns))})"


def keep_columns(df, columns):
    return df[list(columns)], f"Kept {len(columns)} column(s)", f"df = df[{R(list(columns))}]"


def rename_column(df, old, new):
    if not new or new in df.columns:
        raise ValueError("New name is empty or already exists.")
    return df.rename(columns={old: new}), f"Renamed '{old}' → '{new}'", f"df = df.rename(columns={{{R(old)}: {R(new)}}})"


def convert_type(df, column, to, date_format=None):
    out = df.copy()
    s = out[column]
    if to == "number":
        cleaned = s.astype(str).str.replace(r"[^\d.\-eE]", "", regex=True).replace("", np.nan)
        out[column] = pd.to_numeric(cleaned, errors="coerce")
        code = (f"df[{R(column)}] = pd.to_numeric(df[{R(column)}].astype(str)"
                f".str.replace(r'[^\\d.\\-eE]', '', regex=True).replace('', np.nan), errors='coerce')")
    elif to == "integer":
        out[column] = pd.to_numeric(s, errors="coerce").round().astype("Int64")
        code = f"df[{R(column)}] = pd.to_numeric(df[{R(column)}], errors='coerce').round().astype('Int64')"
    elif to == "date":
        if date_format:
            out[column] = pd.to_datetime(s, errors="coerce", format=date_format)
            code = f"df[{R(column)}] = pd.to_datetime(df[{R(column)}], errors='coerce', format={R(date_format)})"
        else:
            out[column] = pd.to_datetime(s, errors="coerce", format="mixed")
            code = f"df[{R(column)}] = pd.to_datetime(df[{R(column)}], errors='coerce', format='mixed')"
    elif to == "text":
        out[column] = s.astype("string")
        code = f"df[{R(column)}] = df[{R(column)}].astype('string')"
    elif to == "category":
        out[column] = s.astype("category")
        code = f"df[{R(column)}] = df[{R(column)}].astype('category')"
    elif to == "boolean":
        mapping = {"true": True, "yes": True, "y": True, "1": True, "t": True,
                   "false": False, "no": False, "n": False, "0": False, "f": False}
        out[column] = s.astype(str).str.strip().str.lower().map(mapping).astype("boolean")
        code = (f"mapping = {R(mapping)}\n"
                f"df[{R(column)}] = df[{R(column)}].astype(str).str.strip().str.lower().map(mapping).astype('boolean')")
    else:
        raise ValueError(to)
    lost = int(out[column].isna().sum() - s.isna().sum())
    label = f"Converted '{column}' to {to}" + (f" ({lost} values could not be converted)" if lost > 0 else "")
    return out, label, code


def fill_missing(df, columns, method, value=None):
    out = df.copy()
    codes = []
    for c in columns:
        s = out[c]
        if method == "mean":
            v = pd.to_numeric(s, errors="coerce").mean()
            out[c] = s.fillna(v)
            codes.append(f"df[{R(c)}] = df[{R(c)}].fillna(df[{R(c)}].mean())")
        elif method == "median":
            v = pd.to_numeric(s, errors="coerce").median()
            out[c] = s.fillna(v)
            codes.append(f"df[{R(c)}] = df[{R(c)}].fillna(df[{R(c)}].median())")
        elif method == "mode":
            m = s.mode(dropna=True)
            if len(m):
                out[c] = s.fillna(m.iloc[0])
            codes.append(f"df[{R(c)}] = df[{R(c)}].fillna(df[{R(c)}].mode().iloc[0])")
        elif method == "constant":
            v = value
            if pd.api.types.is_numeric_dtype(s):
                try:
                    v = float(value)
                except (TypeError, ValueError):
                    pass
            out[c] = s.fillna(v)
            codes.append(f"df[{R(c)}] = df[{R(c)}].fillna({R(v)})")
        elif method == "forward fill":
            out[c] = s.ffill()
            codes.append(f"df[{R(c)}] = df[{R(c)}].ffill()")
        elif method == "backward fill":
            out[c] = s.bfill()
            codes.append(f"df[{R(c)}] = df[{R(c)}].bfill()")
        elif method == "interpolate":
            out[c] = pd.to_numeric(s, errors="coerce").interpolate(limit_direction="both")
            codes.append(f"df[{R(c)}] = df[{R(c)}].interpolate(limit_direction='both')")
        else:
            raise ValueError(method)
    filled = int(df[columns].isna().sum().sum() - out[columns].isna().sum().sum())
    return out, f"Filled {filled:,} missing value(s) in {len(columns)} column(s) using {method}", "\n".join(codes)


def drop_missing(df, columns=None, how="any"):
    subset = list(columns) if columns else None
    out = df.dropna(subset=subset, how=how)
    n = len(df) - len(out)
    scope = f"in {', '.join(map(str, subset))}" if subset else "in any column"
    return out, f"Dropped {n:,} row(s) with missing values {scope}", f"df = df.dropna(subset={R(subset)}, how={R(how)})"


def drop_duplicates(df, columns=None, keep="first"):
    subset = list(columns) if columns else None
    k = False if keep == "none" else keep
    out = df.drop_duplicates(subset=subset, keep=k)
    return out, f"Removed {len(df) - len(out):,} duplicate row(s)", f"df = df.drop_duplicates(subset={R(subset)}, keep={R(k)})"


def handle_outliers(df, column, method="IQR", threshold=1.5, action="remove"):
    mask = outlier_mask(df[column], method, threshold).fillna(False)
    n = int(mask.sum())
    s = pd.to_numeric(df[column], errors="coerce")
    if action == "remove":
        out = df[~mask]
        label = f"Removed {n:,} outlier row(s) in '{column}' ({method}, {threshold})"
    elif action == "cap":
        lo, hi = s[~mask].min(), s[~mask].max()
        out = df.copy()
        out[column] = s.clip(lo, hi)
        label = f"Capped {n:,} outlier(s) in '{column}' to [{lo:.4g}, {hi:.4g}]"
    elif action == "set missing":
        out = df.copy()
        out.loc[mask, column] = np.nan
        label = f"Set {n:,} outlier(s) in '{column}' to missing"
    else:
        raise ValueError(action)
    if method == "IQR":
        code = (f"q1, q3 = df[{R(column)}].quantile([0.25, 0.75]); iqr = q3 - q1\n"
                f"mask = (df[{R(column)}] < q1 - {threshold} * iqr) | (df[{R(column)}] > q3 + {threshold} * iqr)")
    elif method == "Z-score":
        code = (f"z = (df[{R(column)}] - df[{R(column)}].mean()) / df[{R(column)}].std(ddof=0)\n"
                f"mask = z.abs() > {threshold}")
    else:
        code = (f"med = df[{R(column)}].median(); mad = (df[{R(column)}] - med).abs().median()\n"
                f"mask = (0.6745 * (df[{R(column)}] - med) / mad).abs() > {threshold}")
    if action == "remove":
        code += "\ndf = df[~mask]"
    elif action == "cap":
        code += f"\ndf[{R(column)}] = df[{R(column)}].clip(df.loc[~mask, {R(column)}].min(), df.loc[~mask, {R(column)}].max())"
    else:
        code += f"\ndf.loc[mask, {R(column)}] = np.nan"
    return out, label, code


TEXT_OPS = ["trim whitespace", "lowercase", "UPPERCASE", "Title Case", "collapse spaces", "remove punctuation"]


def text_clean(df, columns, ops):
    out = df.copy()
    codes = []
    for c in columns:
        s = out[c]
        notna = s.notna()
        t = s.astype("string")
        expr = f"df[{R(c)}].astype('string')"
        for op in ops:
            if op == "trim whitespace":
                t, expr = t.str.strip(), expr + ".str.strip()"
            elif op == "lowercase":
                t, expr = t.str.lower(), expr + ".str.lower()"
            elif op == "UPPERCASE":
                t, expr = t.str.upper(), expr + ".str.upper()"
            elif op == "Title Case":
                t, expr = t.str.title(), expr + ".str.title()"
            elif op == "collapse spaces":
                t, expr = t.str.replace(r"\s+", " ", regex=True), expr + ".str.replace(r'\\s+', ' ', regex=True)"
            elif op == "remove punctuation":
                t, expr = t.str.replace(r"[^\w\s]", "", regex=True), expr + ".str.replace(r'[^\\w\\s]', '', regex=True)"
        out[c] = t.where(notna, pd.NA)
        codes.append(f"df[{R(c)}] = {expr}")
    return out, f"Text cleanup ({', '.join(ops)}) on {len(columns)} column(s)", "\n".join(codes)


def replace_values(df, column, find, replace, regex=False, whole=True):
    out = df.copy()
    s = out[column]
    if whole and not regex:
        target = find
        if pd.api.types.is_numeric_dtype(s):
            try:
                target = float(find)
                replace = float(replace) if replace not in ("", None) else np.nan
            except ValueError:
                pass
        out[column] = s.replace(target, replace)
        code = f"df[{R(column)}] = df[{R(column)}].replace({R(target)}, {R(replace)})"
    else:
        out[column] = s.astype("string").str.replace(find, replace, regex=regex)
        code = f"df[{R(column)}] = df[{R(column)}].astype('string').str.replace({R(find)}, {R(replace)}, regex={regex})"
    changed = int((out[column].astype(str) != s.astype(str)).sum())
    return out, f"Replaced '{find}' → '{replace}' in '{column}' ({changed:,} cell(s))", code


def calculated_column(df, name, expression):
    if not name:
        raise ValueError("Give the new column a name.")
    out = df.copy()
    out[name] = out.eval(expression, engine="python")
    return out, f"Created '{name}' = {expression}", f"df[{R(name)}] = df.eval({R(expression)}, engine='python')"


def bin_column(df, column, bins, strategy="equal width", name=None, labels=None):
    name = name or f"{column}_bin"
    out = df.copy()
    s = pd.to_numeric(out[column], errors="coerce")
    lab = [x.strip() for x in labels.split(",")] if labels else None
    if lab and len(lab) != bins:
        raise ValueError(f"Provide exactly {bins} labels (got {len(lab)}).")
    if strategy == "quantile":
        out[name] = pd.qcut(s, q=bins, labels=lab, duplicates="drop")
        code = f"df[{R(name)}] = pd.qcut(df[{R(column)}], q={bins}, labels={R(lab)}, duplicates='drop')"
    else:
        out[name] = pd.cut(s, bins=bins, labels=lab)
        code = f"df[{R(name)}] = pd.cut(df[{R(column)}], bins={bins}, labels={R(lab)})"
    out[name] = out[name].astype(str).replace("nan", np.nan)
    code += f"\ndf[{R(name)}] = df[{R(name)}].astype(str).replace('nan', np.nan)"
    return out, f"Binned '{column}' into {bins} {strategy} bins → '{name}'", code


MATH_OPS = ["log (natural, log1p)", "log10", "square root", "square", "z-score standardize",
            "min-max normalize (0-1)", "absolute value", "round", "rank (percentile)"]


def math_transform(df, column, op, new_name=None, decimals=2):
    out = df.copy()
    s = pd.to_numeric(out[column], errors="coerce")
    target = new_name or column
    c = f"df[{R(column)}]"
    if op.startswith("log (natural"):
        if (s < 0).any():
            raise ValueError("log1p needs values ≥ 0 (found negatives). Try z-score instead.")
        out[target], expr = np.log1p(s), f"np.log1p({c})"
    elif op == "log10":
        if (s <= 0).any():
            raise ValueError("log10 needs strictly positive values.")
        out[target], expr = np.log10(s), f"np.log10({c})"
    elif op == "square root":
        if (s < 0).any():
            raise ValueError("square root needs values ≥ 0.")
        out[target], expr = np.sqrt(s), f"np.sqrt({c})"
    elif op == "square":
        out[target], expr = s ** 2, f"{c} ** 2"
    elif op == "z-score standardize":
        out[target], expr = (s - s.mean()) / s.std(), f"({c} - {c}.mean()) / {c}.std()"
    elif op == "min-max normalize (0-1)":
        rng = s.max() - s.min()
        out[target], expr = (s - s.min()) / (rng if rng else 1), f"({c} - {c}.min()) / ({c}.max() - {c}.min())"
    elif op == "absolute value":
        out[target], expr = s.abs(), f"{c}.abs()"
    elif op == "round":
        out[target], expr = s.round(decimals), f"{c}.round({decimals})"
    elif op == "rank (percentile)":
        out[target], expr = s.rank(pct=True), f"{c}.rank(pct=True)"
    else:
        raise ValueError(op)
    return out, f"Applied {op} to '{column}'" + (f" → '{target}'" if target != column else ""), f"df[{R(target)}] = {expr}"


DATE_PARTS = ["year", "quarter", "month", "month name", "week", "day", "weekday", "weekday name", "hour", "is weekend"]


def extract_datetime(df, column, parts):
    out = df.copy()
    s = pd.to_datetime(out[column], errors="coerce")
    codes = [f"_d = pd.to_datetime(df[{R(column)}], errors='coerce')"]
    for p in parts:
        name = f"{column}_{p.replace(' ', '_')}"
        if p == "year":
            out[name], e = s.dt.year, "_d.dt.year"
        elif p == "quarter":
            out[name], e = s.dt.quarter, "_d.dt.quarter"
        elif p == "month":
            out[name], e = s.dt.month, "_d.dt.month"
        elif p == "month name":
            out[name], e = s.dt.month_name(), "_d.dt.month_name()"
        elif p == "week":
            out[name], e = s.dt.isocalendar().week.astype("Int64"), "_d.dt.isocalendar().week.astype('Int64')"
        elif p == "day":
            out[name], e = s.dt.day, "_d.dt.day"
        elif p == "weekday":
            out[name], e = s.dt.weekday, "_d.dt.weekday"
        elif p == "weekday name":
            out[name], e = s.dt.day_name(), "_d.dt.day_name()"
        elif p == "hour":
            out[name], e = s.dt.hour, "_d.dt.hour"
        elif p == "is weekend":
            out[name], e = s.dt.weekday >= 5, "_d.dt.weekday >= 5"
        else:
            continue
        codes.append(f"df[{R(name)}] = {e}")
    return out, f"Extracted {', '.join(parts)} from '{column}'", "\n".join(codes)


def filter_rows(df, expression):
    out = df.query(expression, engine="python")
    return out, f"Kept rows where {expression} ({len(df) - len(out):,} removed)", f"df = df.query({R(expression)}, engine='python')"


def one_hot(df, columns, drop_first=False):
    out = pd.get_dummies(df, columns=list(columns), drop_first=drop_first, dtype=int)
    added = out.shape[1] - df.shape[1] + len(columns)
    return out, f"One-hot encoded {', '.join(map(str, columns))} (+{added} columns)", \
        f"df = pd.get_dummies(df, columns={R(list(columns))}, drop_first={drop_first}, dtype=int)"


def sort_rows(df, columns, ascending=True):
    out = df.sort_values(list(columns), ascending=ascending, kind="stable")
    return out, f"Sorted by {', '.join(map(str, columns))} ({'ascending' if ascending else 'descending'})", \
        f"df = df.sort_values({R(list(columns))}, ascending={ascending}, kind='stable')"


def split_column(df, column, sep, max_parts=2, prefix=None):
    prefix = prefix or column
    parts = df[column].astype("string").str.split(sep, n=max_parts - 1, expand=True, regex=False)
    parts.columns = [f"{prefix}_{i + 1}" for i in range(parts.shape[1])]
    out = pd.concat([df, parts], axis=1)
    code = (f"_parts = df[{R(column)}].astype('string').str.split({R(sep)}, n={max_parts - 1}, expand=True, regex=False)\n"
            f"_parts.columns = [f'{prefix}_{{i + 1}}' for i in range(_parts.shape[1])]\n"
            f"df = pd.concat([df, _parts], axis=1)")
    return out, f"Split '{column}' on '{sep}' into {parts.shape[1]} column(s)", code


def sample_rows(df, n=None, frac=None, seed=42):
    out = df.sample(n=n, frac=frac, random_state=seed)
    what = f"n={n}" if n else f"frac={frac}"
    return out, f"Random sample ({len(out):,} rows)", f"df = df.sample({what}, random_state={seed})"


def add_column(df, name, values):
    out = df.copy()
    out[name] = values.values if hasattr(values, "values") else values
    return out, f"Added column '{name}'", f"# '{name}' was computed interactively (e.g. clustering labels)"


def pipeline_script(file_name: str, steps: list[dict], filters_code: str = "") -> str:
    """Assemble the transform log into a standalone, runnable script."""
    ext = file_name.rsplit(".", 1)[-1].lower() if file_name and "." in file_name else "csv"
    reader = {"xlsx": "read_excel", "xls": "read_excel", "json": "read_json", "parquet": "read_parquet"}.get(ext, "read_csv")
    safe = re.sub(r"[^\w.\- ]", "_", file_name or "data.csv")
    lines = [
        '"""',
        "Data preparation pipeline generated by DataViz.",
        f"Source file: {safe}",
        '"""',
        "import numpy as np",
        "import pandas as pd",
        "",
        f"df = pd.{reader}({R(safe)})",
        "",
    ]
    for i, st_ in enumerate(steps, 1):
        lines.append(f"# Step {i}: {st_['label']}")
        lines.append(st_["code"])
        lines.append("")
    if filters_code:
        lines.append("# Active global filters")
        lines.append(filters_code)
        lines.append("")
    lines.append("print(df.shape)")
    lines.append("df.to_csv('cleaned_data.csv', index=False)")
    return "\n".join(lines)

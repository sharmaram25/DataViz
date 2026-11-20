"""
Data ingestion: files (CSV/TSV/TXT, Excel, JSON/JSONL, Parquet), URLs and
built-in sample datasets, followed by automatic type inference (numbers and
dates stored as text are converted so every downstream tool sees real types).
"""
from __future__ import annotations

import csv
import datetime as _dt
import io
import json
import re
import urllib.parse
import urllib.request
import warnings

import numpy as np
import pandas as pd
import streamlit as st
from pandas.api.types import is_numeric_dtype

SUPPORTED_EXTENSIONS = ["csv", "tsv", "txt", "xlsx", "xlsm", "xls", "json", "jsonl", "ndjson", "parquet"]
ENCODINGS = ["utf-8-sig", "utf-8", "cp1252", "latin-1"]
MAX_URL_BYTES = 300 * 1024 * 1024  # 300 MB


# --------------------------------------------------------------------------
# Reading raw bytes
# --------------------------------------------------------------------------

def _extension(filename: str) -> str:
    return filename.rsplit(".", 1)[-1].lower() if "." in filename else ""


def _decode(data: bytes, encoding: str | None) -> tuple[str, str]:
    if encoding:
        return data.decode(encoding, errors="replace"), encoding
    for enc in ENCODINGS:
        try:
            return data.decode(enc), enc
        except UnicodeDecodeError:
            continue
    return data.decode("latin-1", errors="replace"), "latin-1"


def _sniff_delimiter(text: str, default: str = ",") -> str:
    sample = text[:65536]
    try:
        return csv.Sniffer().sniff(sample, delimiters=",;\t|").delimiter
    except csv.Error:
        # Fall back to the most frequent candidate in the first lines
        head = "\n".join(sample.splitlines()[:20])
        counts = {d: head.count(d) for d in [",", ";", "\t", "|"]}
        best = max(counts, key=counts.get)
        return best if counts[best] else default


def list_excel_sheets(data: bytes) -> list[str]:
    try:
        return list(pd.ExcelFile(io.BytesIO(data)).sheet_names)
    except Exception:
        return []


def _read_json(text: str) -> pd.DataFrame:
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        # Probably JSON Lines
        return pd.read_json(io.StringIO(text), lines=True)

    if isinstance(obj, list):
        return pd.json_normalize(obj) if obj and isinstance(obj[0], dict) else pd.DataFrame(obj)
    if isinstance(obj, dict):
        # Common API shapes: {"data": [...]} / {"records": [...]} / {"results": [...]}
        for key, value in obj.items():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return pd.json_normalize(value)
        try:
            return pd.DataFrame(obj)
        except ValueError:
            return pd.json_normalize(obj)
    raise ValueError("JSON root must be an array or an object.")


def read_bytes(
    data: bytes,
    filename: str,
    sheet: str | int | None = None,
    sep: str | None = None,
    encoding: str | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    """Parse raw bytes into a DataFrame. Returns (df, notes)."""
    ext = _extension(filename)
    notes: list[str] = []

    if ext in ("csv", "tsv", "txt", ""):
        text, enc = _decode(data, encoding)
        if sep is None:
            sep = "\t" if ext == "tsv" else _sniff_delimiter(text)
        if enc not in ("utf-8", "utf-8-sig"):
            notes.append(f"Decoded using **{enc}** encoding.")
        if sep != ",":
            notes.append(f"Detected delimiter **{repr(sep)}**.")
        df = pd.read_csv(io.StringIO(text), sep=sep, low_memory=False, on_bad_lines="warn")
    elif ext in ("xlsx", "xlsm", "xls"):
        try:
            df = pd.read_excel(io.BytesIO(data), sheet_name=sheet if sheet is not None else 0)
        except ImportError as e:
            raise ValueError(
                "Legacy .xls files need the 'xlrd' package (pip install xlrd), "
                "or re-save the file as .xlsx."
            ) from e
        if sheet is not None:
            notes.append(f"Loaded sheet **{sheet}**.")
    elif ext in ("json",):
        text, _ = _decode(data, encoding)
        df = _read_json(text)
    elif ext in ("jsonl", "ndjson"):
        text, _ = _decode(data, encoding)
        df = pd.read_json(io.StringIO(text), lines=True)
    elif ext == "parquet":
        df = pd.read_parquet(io.BytesIO(data))
    else:
        raise ValueError(
            f"Unsupported file type '.{ext}'. Supported: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return df, notes


# --------------------------------------------------------------------------
# Cleaning column names & inferring types
# --------------------------------------------------------------------------

def clean_column_names(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    notes = []
    new_cols, seen = [], {}
    changed = False
    for i, c in enumerate(df.columns):
        name = str(c).strip()
        if not name or name.lower().startswith("unnamed:"):
            name = f"column_{i + 1}"
        base = name
        if base in seen:
            seen[base] += 1
            name = f"{base}_{seen[base]}"
        else:
            seen[base] = 0
        changed |= name != c
        new_cols.append(name)
    if changed:
        df = df.copy()
        df.columns = new_cols
        notes.append("Cleaned column names (trimmed whitespace, named blanks, de-duplicated).")
    return df, notes


_CURRENCY_RE = re.compile(r"[\s$€£¥₹%]")
_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?$")
_DATE_HINT_RE = re.compile(
    r"(?:\d{1,4}[-/.]\d{1,2}[-/.]\d{1,4})|(?:\d{4}-\d{2})|"
    r"(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*[\s,.-]+\d",
    re.IGNORECASE,
)


def _to_numeric_text(s: pd.Series) -> pd.Series | None:
    """Try to convert a text series to numbers. Returns None if it isn't numeric."""
    text = s.dropna().astype(str).str.strip()
    if text.empty:
        return None
    sample = text.sample(min(len(text), 2000), random_state=0)

    # Leading zeros (zip codes, IDs like 00123) must stay text
    if (sample.str.match(r"^0\d+$")).mean() > 0.05:
        return None

    cleaned = sample.str.replace(_CURRENCY_RE, "", regex=True)
    has_thousands = cleaned.str.match(_THOUSANDS_RE.pattern).fillna(False).astype(bool)
    if has_thousands.any():
        cleaned = cleaned.where(~has_thousands, cleaned.str.replace(",", "", regex=False))
    conv = pd.to_numeric(cleaned, errors="coerce")
    if conv.notna().mean() < 0.95:
        return None

    full = s.astype("string").str.strip().str.replace(_CURRENCY_RE, "", regex=True)
    mask = full.str.match(_THOUSANDS_RE.pattern).fillna(False)
    full = full.where(~mask, full.str.replace(",", "", regex=False))
    return pd.to_numeric(full, errors="coerce").astype("float64")


def _to_datetime_text(s: pd.Series) -> pd.Series | None:
    text = s.dropna().astype(str).str.strip()
    if text.empty:
        return None
    sample = text.sample(min(len(text), 500), random_state=0)
    if sample.str.contains(_DATE_HINT_RE).mean() < 0.9:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        parsed = pd.to_datetime(sample, errors="coerce")
        fmt_ok = parsed.notna().mean() >= 0.9
        if fmt_ok:
            full = pd.to_datetime(s, errors="coerce")
        else:
            try:
                parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
            except (TypeError, ValueError):
                return None
            if parsed.notna().mean() < 0.9:
                return None
            full = pd.to_datetime(s, errors="coerce", format="mixed")
    # Drop timezone so every chart / resample works uniformly
    if getattr(full.dt, "tz", None) is not None:
        full = full.dt.tz_convert(None)
    if full.notna().sum() < s.notna().sum() * 0.9:
        return None
    return full


def infer_types(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Convert numeric-looking and date-looking text columns. Returns (df, notes)."""
    notes = []
    df = df.copy()
    to_num, to_date, mixed = [], [], []
    for col in df.columns:
        s = df[col]
        if is_numeric_dtype(s) or pd.api.types.is_datetime64_any_dtype(s) or pd.api.types.is_bool_dtype(s):
            continue
        non_null = s.dropna()
        if non_null.empty:
            continue
        sample = non_null.head(2000)
        if not all(isinstance(v, str) for v in sample):
            # Nested JSON → serialise to text
            if any(isinstance(v, (list, dict)) for v in sample.head(50)):
                df[col] = s.map(lambda v: json.dumps(v, default=str) if isinstance(v, (list, dict)) else v)
                mixed.append(col)
                continue
            # Mixed-type column (common in Excel): datetimes / numbers / text in one column
            dt_share = sample.map(lambda v: isinstance(v, (_dt.date, pd.Timestamp, np.datetime64))).mean()
            num_share = sample.map(lambda v: isinstance(v, (int, float, np.number)) and not isinstance(v, bool)).mean()
            if dt_share >= 0.9:
                df[col] = pd.to_datetime(s.map(lambda v: v if isinstance(v, (_dt.date, pd.Timestamp, np.datetime64)) else None),
                                         errors="coerce")
                to_date.append(col)
                continue
            if num_share >= 0.9:
                df[col] = pd.to_numeric(s, errors="coerce").astype("float64")
                to_num.append(col)
                continue
            s = s.map(lambda v: v if v is None or (isinstance(v, float) and np.isnan(v)) else str(v))
            df[col] = s
            mixed.append(col)
        num = _to_numeric_text(s)
        if num is not None:
            df[col] = num
            to_num.append(col)
            continue
        dt = _to_datetime_text(s)
        if dt is not None:
            df[col] = dt
            to_date.append(col)
    if to_num:
        notes.append(f"Converted text → **number**: {', '.join(map(str, to_num))}")
    if to_date:
        notes.append(f"Converted text → **date**: {', '.join(map(str, to_date))}")
    if mixed:
        notes.append(f"Unified mixed-type values as **text**: {', '.join(map(str, mixed))}")
    return df, notes


@st.cache_data(show_spinner=False, max_entries=6)
def load_bytes(
    data: bytes,
    filename: str,
    sheet: str | int | None = None,
    sep: str | None = None,
    encoding: str | None = None,
    auto_types: bool = True,
) -> tuple[pd.DataFrame | None, list[str], str | None]:
    """Full pipeline. Returns (df, notes, error)."""
    try:
        df, notes = read_bytes(data, filename, sheet=sheet, sep=sep, encoding=encoding)
        if df.empty and df.shape[1] == 0:
            return None, notes, "The file contains no data."
        df, n2 = clean_column_names(df)
        notes += n2
        if auto_types:
            df, n3 = infer_types(df)
            notes += n3
        return df.reset_index(drop=True), notes, None
    except Exception as e:  # surface parse problems to the user
        return None, [], f"Could not load **{filename}**: {e}"


def load_data(uploaded_file):
    """Backward-compatible API: (df, error) from a Streamlit UploadedFile."""
    df, _notes, err = load_bytes(uploaded_file.getvalue(), uploaded_file.name)
    return df, err


def fetch_url(url: str) -> tuple[bytes, str]:
    """Download a remote file (http/https only). Returns (bytes, filename)."""
    parsed = urllib.parse.urlparse(url.strip())
    if parsed.scheme not in ("http", "https"):
        raise ValueError("Only http(s) URLs are supported.")
    # GitHub "blob" pages → raw file
    if parsed.netloc == "github.com" and "/blob/" in parsed.path:
        url = url.replace("github.com", "raw.githubusercontent.com", 1).replace("/blob/", "/", 1)
        parsed = urllib.parse.urlparse(url)
    req = urllib.request.Request(url, headers={"User-Agent": "DataViz/3.0"})
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = resp.read(MAX_URL_BYTES + 1)
    if len(data) > MAX_URL_BYTES:
        raise ValueError("Remote file is larger than 300 MB.")
    name = urllib.parse.unquote(parsed.path.rsplit("/", 1)[-1]) or "remote.csv"
    if _extension(name) not in SUPPORTED_EXTENSIONS:
        name += ".csv"
    return data, name


# --------------------------------------------------------------------------
# Sample datasets
# --------------------------------------------------------------------------

def _synthetic_sales(n: int = 5000, seed: int = 7) -> pd.DataFrame:
    """A realistic, deliberately messy sales dataset for exploring every tool."""
    rng = np.random.default_rng(seed)
    regions = {"North": ["Oslo", "Stockholm", "Helsinki"], "South": ["Madrid", "Rome", "Athens"],
               "West": ["Lisbon", "Dublin", "Paris"], "East": ["Warsaw", "Prague", "Vienna"]}
    categories = {"Electronics": (450, 180), "Furniture": (320, 120), "Office": (45, 20), "Apparel": (70, 30)}
    region = rng.choice(list(regions), n, p=[0.2, 0.3, 0.3, 0.2])
    city = [rng.choice(regions[r]) for r in region]
    cat = rng.choice(list(categories), n, p=[0.3, 0.2, 0.3, 0.2])
    base = np.array([categories[c][0] for c in cat])
    spread = np.array([categories[c][1] for c in cat])
    price = np.abs(rng.normal(base, spread)).round(2)
    qty = rng.poisson(3, n) + 1
    dates = pd.Timestamp("2023-01-01") + pd.to_timedelta(rng.integers(0, 730, n), unit="D")
    season = 1 + 0.25 * np.sin((dates.day_of_year.values / 365) * 2 * np.pi - 1.2)
    revenue = (price * qty * season).round(2)
    discount = rng.choice([0, 0.05, 0.1, 0.15, 0.2], n, p=[0.5, 0.2, 0.15, 0.1, 0.05])
    profit = (revenue * (0.28 - discount) + rng.normal(0, 25, n)).round(2)
    satisfaction = np.clip(rng.normal(4 - discount * 2, 0.8, n), 1, 5).round(1)
    channel = rng.choice(["Online", "Retail", "Partner"], n, p=[0.5, 0.35, 0.15])
    df = pd.DataFrame({
        "order_id": [f"ORD-{100000 + i}" for i in range(n)],
        "order_date": dates.strftime("%Y-%m-%d"),
        "region": region, "city": city, "category": cat, "channel": channel,
        "unit_price": price, "quantity": qty, "discount": discount,
        "revenue": revenue, "profit": profit, "satisfaction": satisfaction,
    })
    # Messiness: missing values, outliers, inconsistent text, duplicates
    df.loc[rng.choice(n, 150, replace=False), "satisfaction"] = np.nan
    df.loc[rng.choice(n, 60, replace=False), "channel"] = np.nan
    df.loc[rng.choice(n, 12, replace=False), "revenue"] *= 25
    idx = rng.choice(n, 80, replace=False)
    df.loc[idx, "region"] = df.loc[idx, "region"].str.lower() + " "
    df = pd.concat([df, df.sample(40, random_state=seed)], ignore_index=True)
    return df


SAMPLE_DATASETS = {
    "sales": ("🛒 Retail Sales (messy)", "5k orders with dates, missing values, outliers & duplicates — great for cleaning."),
    "gapminder": ("🌍 Gapminder", "Life expectancy, GDP & population for 142 countries, 1952–2007."),
    "tips": ("🍽️ Restaurant Tips", "Bills, tips and diner attributes — ideal for statistical tests."),
    "iris": ("🌸 Iris Flowers", "Classic 3-species measurements — perfect for PCA & clustering."),
    "stocks": ("📈 Tech Stocks", "Daily normalized prices for 6 tech stocks (2018–2019) — time series."),
    "wind": ("🌬️ Wind", "Wind direction & strength frequencies — composition charts."),
}


def sample_name(key: str) -> str:
    """Display name for a sample, e.g. 'Retail Sales (sample)'."""
    title = SAMPLE_DATASETS[key][0].split(" ", 1)[1]
    return re.sub(r"\s*\(.*?\)", "", title).strip() + " (sample)"


@st.cache_data(show_spinner=False)
def load_sample(key: str) -> tuple[pd.DataFrame, list[str]]:
    import plotly.express as px

    if key == "sales":
        df = _synthetic_sales()
    else:
        loader = getattr(px.data, key)
        df = loader()
        if not isinstance(df, pd.DataFrame):  # narwhals / other backends
            df = pd.DataFrame(df)
    df, notes = infer_types(df)
    return df.reset_index(drop=True), notes

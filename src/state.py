"""
Session state: the original dataset, the working copy (after transforms),
an undo stack, the transform log (with generated pandas code), global
filters and the pinned dashboard charts.
"""
from __future__ import annotations

import json
import uuid

import pandas as pd
import streamlit as st
from pandas.api.types import is_datetime64_any_dtype

from src.preprocess import is_numeric

MAX_UNDO = 15
PERSIST_PREFIX = "p_"  # widget keys that should survive page switches

DEFAULTS = {
    "raw_df": None,
    "df": None,
    "file_name": None,
    "load_notes": [],
    "history": [],      # [{label, code}]
    "undo": [],         # [(df, history_len)]
    "filters": [],      # [{id, col, kind, value}]
    "dashboard": [],    # [{id, spec}]
    "version": 0,
    "nav": "Overview",
}


def init_state():
    for k, v in DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v.copy() if isinstance(v, list) else v
    # Keep widget values when the user switches pages (Streamlit drops state
    # for widgets that are not rendered on the current run).
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(PERSIST_PREFIX):
            st.session_state[k] = st.session_state[k]


def goto(page: str):
    """Button callback: switch page (callbacks run before widgets are created)."""
    st.session_state.nav = page


def has_data() -> bool:
    return st.session_state.get("df") is not None


def set_dataset(df: pd.DataFrame, name: str, notes: list[str] | None = None):
    st.session_state.raw_df = df
    st.session_state.df = df
    st.session_state.file_name = name
    st.session_state.load_notes = notes or []
    st.session_state.history = []
    st.session_state.undo = []
    st.session_state.filters = []
    st.session_state.dashboard = []
    st.session_state.version += 1
    st.session_state.nav = "Overview"
    _clear_page_widgets()


def unload():
    for k in ("raw_df", "df", "file_name"):
        st.session_state[k] = None
    st.session_state.history, st.session_state.undo = [], []
    st.session_state.filters, st.session_state.dashboard = [], []
    st.session_state.version += 1
    _clear_page_widgets()


def _clear_page_widgets():
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and (k.startswith(PERSIST_PREFIX) or k.startswith("flt_")):
            del st.session_state[k]


def apply_step(new_df: pd.DataFrame, label: str, code: str):
    st.session_state.undo.append((st.session_state.df, len(st.session_state.history)))
    st.session_state.undo = st.session_state.undo[-MAX_UNDO:]
    st.session_state.df = new_df.reset_index(drop=True) if not isinstance(new_df.index, pd.RangeIndex) else new_df
    st.session_state.history.append({"label": label, "code": code})
    st.session_state.version += 1
    _prune_filters()


def undo() -> bool:
    if not st.session_state.undo:
        return False
    df, hist_len = st.session_state.undo.pop()
    st.session_state.df = df
    st.session_state.history = st.session_state.history[:hist_len]
    st.session_state.version += 1
    _prune_filters()
    return True


def reset_to_original():
    st.session_state.df = st.session_state.raw_df
    st.session_state.history, st.session_state.undo = [], []
    st.session_state.version += 1
    _prune_filters()


def _prune_filters():
    cols = set(st.session_state.df.columns)
    st.session_state.filters = [f for f in st.session_state.filters if f["col"] in cols]


# --------------------------------------------------------------------------
# Global filters
# --------------------------------------------------------------------------

def filter_kind(series: pd.Series) -> str:
    if is_datetime64_any_dtype(series):
        return "date"
    if is_numeric(series):
        return "range"
    return "values"


def add_filter(df: pd.DataFrame, col: str):
    kind = filter_kind(df[col])
    s = df[col].dropna()
    if kind == "range":
        value = [float(s.min()), float(s.max())] if len(s) else [0.0, 0.0]
    elif kind == "date":
        value = [s.min().date(), s.max().date()] if len(s) else None
    else:
        value = []
    st.session_state.filters.append({"id": uuid.uuid4().hex[:8], "col": col, "kind": kind, "value": value,
                                     "exclude": False, "keep_na": False})


def remove_filter(fid: str):
    st.session_state.filters = [f for f in st.session_state.filters if f["id"] != fid]
    for k in list(st.session_state.keys()):
        if isinstance(k, str) and k.startswith(f"flt_{fid}"):
            del st.session_state[k]


def apply_filters(df: pd.DataFrame, filters: list[dict]) -> pd.DataFrame:
    """Pure function: apply filter dicts to a frame."""
    mask = pd.Series(True, index=df.index)
    for f in filters:
        col = f["col"]
        if col not in df.columns or f.get("value") in (None, []):
            continue
        s = df[col]
        if f["kind"] == "range":
            lo, hi = f["value"]
            m = (s >= lo) & (s <= hi)
        elif f["kind"] == "date":
            lo, hi = pd.Timestamp(f["value"][0]), pd.Timestamp(f["value"][1]) + pd.Timedelta(days=1)
            m = (s >= lo) & (s < hi)
        elif f["kind"] == "text":
            m = s.astype(str).str.contains(str(f["value"]), case=False, regex=False, na=False)
        else:
            m = s.astype(str).isin([str(v) for v in f["value"]])
        if f.get("exclude"):
            m = ~m
        if f.get("keep_na"):
            m = m | s.isna()
        mask &= m.fillna(False)
    return df[mask]


def get_view() -> pd.DataFrame:
    """Working dataset with global filters applied (memoised per version + filter set)."""
    df = st.session_state.df
    filters = [f for f in st.session_state.filters if f.get("value") not in (None, [])]
    if not filters:
        return df
    key = (st.session_state.version, json.dumps(filters, default=str, sort_keys=True))
    cache = st.session_state.get("_view_cache")
    if cache and cache[0] == key:
        return cache[1]
    view = apply_filters(df, filters)
    st.session_state["_view_cache"] = (key, view)
    return view


def active_filter_count() -> int:
    return sum(1 for f in st.session_state.filters if f.get("value") not in (None, []))


def filters_as_code(filters: list[dict]) -> str:
    lines = []
    for f in filters:
        if f.get("value") in (None, []):
            continue
        c = repr(f["col"])
        neg = "~" if f.get("exclude") else ""
        if f["kind"] == "range":
            cond = f"df[{c}].between({f['value'][0]!r}, {f['value'][1]!r})"
        elif f["kind"] == "date":
            cond = f"df[{c}].between('{f['value'][0]}', '{f['value'][1]} 23:59:59')"
        else:
            cond = f"df[{c}].astype(str).isin({[str(v) for v in f['value']]!r})"
        lines.append(f"df = df[{neg}({cond})]")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# Dashboard pins
# --------------------------------------------------------------------------

def pin_chart(spec: dict):
    st.session_state.dashboard.append({"id": uuid.uuid4().hex[:8], "spec": json.loads(json.dumps(spec, default=str))})


def unpin_chart(cid: str):
    st.session_state.dashboard = [c for c in st.session_state.dashboard if c["id"] != cid]


# --------------------------------------------------------------------------
# Widget helpers that respect persisted state
# --------------------------------------------------------------------------

def _valid(key, options):
    if key in st.session_state and st.session_state[key] not in options:
        del st.session_state[key]


def select(label, options, key, default=None, **kw):
    """Selectbox whose value survives page switches and option changes."""
    options = list(options)
    _valid(key, options)
    if key not in st.session_state:
        idx = options.index(default) if default in options else 0
        return st.selectbox(label, options, index=idx if options else None, key=key, **kw)
    return st.selectbox(label, options, key=key, **kw)


def multiselect(label, options, key, default=None, **kw):
    options = list(options)
    if key in st.session_state:
        st.session_state[key] = [v for v in st.session_state[key] if v in options]
        return st.multiselect(label, options, key=key, **kw)
    return st.multiselect(label, options, default=[d for d in (default or []) if d in options], key=key, **kw)

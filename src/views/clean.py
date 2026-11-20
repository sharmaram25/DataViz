"""Clean & Transform: no-code data preparation with undo and code export."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from src import state, transforms as T
from src.preprocess import classify_columns, get_summary
from src.ui_components import esc, html_block, page_header, section, show_df, tiles

OPS = {
    "🩹 Missing values": ["Fill missing values", "Drop rows with missing values"],
    "🧾 Rows": ["Remove duplicates", "Handle outliers", "Filter rows (expression)", "Sort rows", "Random sample"],
    "🏛️ Columns": ["Drop columns", "Keep only columns", "Rename column", "Change type", "Split column"],
    "🔤 Text": ["Text cleanup", "Find & replace"],
    "⚗️ Feature engineering": ["Calculated column", "Bin numeric column", "Math transform", "Extract date parts",
                               "One-hot encode"],
}
HELP = {
    "Fill missing values": "Impute blanks with a statistic, a constant or neighbouring values.",
    "Drop rows with missing values": "Remove rows that have blanks (in any column, or in the columns you choose).",
    "Remove duplicates": "Drop repeated rows — optionally judged only on some columns.",
    "Handle outliers": "Remove, cap (winsorize) or blank extreme values.",
    "Filter rows (expression)": "Keep rows matching a pandas expression, e.g. `revenue > 100 and region == 'North'`. "
                                "Wrap column names with spaces in backticks.",
    "Sort rows": "Order the dataset by one or more columns.",
    "Random sample": "Down-sample a large dataset to speed everything up.",
    "Drop columns": "Remove columns you don't need.",
    "Keep only columns": "Keep a subset of columns and drop the rest.",
    "Rename column": "Give a column a clearer name.",
    "Change type": "Convert a column to number, integer, date, text, category or boolean.",
    "Split column": "Split text on a delimiter into several new columns.",
    "Text cleanup": "Trim spaces and normalise casing so 'North ', 'north' and 'NORTH' become one category.",
    "Find & replace": "Replace specific values (or substrings / regex) in a column.",
    "Calculated column": "Create a column from a formula, e.g. `profit / revenue` or `price * quantity`.",
    "Bin numeric column": "Turn a numeric column into ranges (equal width or quantiles).",
    "Math transform": "Log, sqrt, standardise, normalise, rank…",
    "Extract date parts": "Create year / month / weekday… columns from a date.",
    "One-hot encode": "Expand categories into 0/1 indicator columns (for modelling).",
}


def _params(op: str, df: pd.DataFrame, types: dict):
    """Render parameter widgets; return (function, kwargs) or (None, message)."""
    cols = list(df.columns)
    nums = types["numerical"]
    text_like = types["categorical"] + types["text"]
    with_missing = [c for c in cols if df[c].isna().any()]

    if op == "Fill missing values":
        pick = st.multiselect("Columns", cols, default=with_missing[:3], key="cl_fill_cols",
                              help="Columns with blanks are pre-selected.")
        method = st.selectbox("Method", ["median", "mean", "mode", "constant", "forward fill", "backward fill", "interpolate"],
                              key="cl_fill_m")
        if method in ("mean", "median", "interpolate"):
            bad = [c for c in pick if c not in nums]
            if bad:
                return None, f"'{method}' needs numeric columns — remove: {', '.join(map(str, bad))}"
        value = st.text_input("Fill value", "0", key="cl_fill_v") if method == "constant" else None
        return (T.fill_missing, dict(columns=pick, method=method, value=value)) if pick else (None, "Choose columns.")
    if op == "Drop rows with missing values":
        pick = st.multiselect("Only consider these columns (empty = all)", cols, key="cl_dm_cols")
        how = st.radio("Drop a row when", ["any", "all"], horizontal=True, key="cl_dm_how",
                       format_func=lambda h: "any selected value is missing" if h == "any" else "all selected values are missing")
        return T.drop_missing, dict(columns=pick or None, how=how)
    if op == "Remove duplicates":
        pick = st.multiselect("Judge duplicates on (empty = all columns)", cols, key="cl_dd_cols")
        keep = st.radio("Keep", ["first", "last", "none"], horizontal=True, key="cl_dd_keep")
        return T.drop_duplicates, dict(columns=pick or None, keep=keep)
    if op == "Handle outliers":
        if not nums:
            return None, "No numeric columns."
        col = st.selectbox("Column", nums, key="cl_out_col")
        method = st.selectbox("Detection", ["IQR", "Z-score", "Modified Z (MAD)"], key="cl_out_m")
        default = {"IQR": 1.5, "Z-score": 3.0, "Modified Z (MAD)": 3.5}[method]
        thr = st.slider("Threshold", 0.5, 6.0, default, 0.1, key=f"cl_out_t_{method}")
        action = st.radio("Action", ["remove", "cap", "set missing"], horizontal=True, key="cl_out_a")
        return T.handle_outliers, dict(column=col, method=method, threshold=thr, action=action)
    if op == "Filter rows (expression)":
        ex = st.text_input("Keep rows where…", placeholder="e.g. revenue > 100 and region == 'North'", key="cl_q")
        st.caption("Columns: " + ", ".join(f"`{c}`" for c in cols[:30]))
        return (T.filter_rows, dict(expression=ex)) if ex else (None, "Enter an expression.")
    if op == "Sort rows":
        pick = st.multiselect("Sort by", cols, key="cl_sort_cols")
        asc = st.toggle("Ascending", True, key="cl_sort_asc")
        return (T.sort_rows, dict(columns=pick, ascending=asc)) if pick else (None, "Choose columns.")
    if op == "Random sample":
        mode = st.radio("Sample by", ["Row count", "Fraction"], horizontal=True, key="cl_samp_mode")
        if mode == "Row count":
            n = st.number_input("Rows", 1, len(df), min(len(df), 10000), key="cl_samp_n")
            return T.sample_rows, dict(n=int(n))
        frac = st.slider("Fraction", 0.01, 1.0, 0.1, key="cl_samp_f")
        return T.sample_rows, dict(frac=frac)
    if op == "Drop columns":
        pick = st.multiselect("Columns to drop", cols, key="cl_drop_cols")
        return (T.drop_columns, dict(columns=pick)) if pick else (None, "Choose columns.")
    if op == "Keep only columns":
        pick = st.multiselect("Columns to keep", cols, default=cols, key="cl_keep_cols")
        return (T.keep_columns, dict(columns=pick)) if pick else (None, "Keep at least one column.")
    if op == "Rename column":
        old = st.selectbox("Column", cols, key="cl_ren_old")
        new = st.text_input("New name", value=str(old), key=f"cl_ren_new_{old}")
        return (T.rename_column, dict(old=old, new=new.strip())) if new.strip() and new.strip() != old else (None, "Type a new name.")
    if op == "Change type":
        col = st.selectbox("Column", cols, key="cl_ct_col")
        st.caption(f"Current type: `{df[col].dtype}`")
        to = st.selectbox("Convert to", ["number", "integer", "date", "text", "category", "boolean"], key="cl_ct_to")
        fmt = st.text_input("Date format (optional)", placeholder="%d/%m/%Y", key="cl_ct_fmt") if to == "date" else None
        return T.convert_type, dict(column=col, to=to, date_format=fmt or None)
    if op == "Split column":
        if not text_like:
            return None, "No text columns."
        col = st.selectbox("Column", text_like, key="cl_sp_col")
        sep = st.text_input("Delimiter", ",", key="cl_sp_sep")
        parts = st.number_input("Max parts", 2, 10, 2, key="cl_sp_n")
        return (T.split_column, dict(column=col, sep=sep, max_parts=int(parts))) if sep else (None, "Enter a delimiter.")
    if op == "Text cleanup":
        if not text_like:
            return None, "No text columns."
        pick = st.multiselect("Columns", text_like, default=text_like[:1], key="cl_tc_cols")
        ops = st.multiselect("Operations (applied in order)", T.TEXT_OPS, default=["trim whitespace", "collapse spaces"], key="cl_tc_ops")
        return (T.text_clean, dict(columns=pick, ops=ops)) if pick and ops else (None, "Choose columns and operations.")
    if op == "Find & replace":
        col = st.selectbox("Column", cols, key="cl_fr_col")
        c1, c2 = st.columns(2)
        find = c1.text_input("Find", key="cl_fr_find")
        repl = c2.text_input("Replace with", key="cl_fr_repl")
        mode = st.radio("Match", ["whole value", "substring", "regex"], horizontal=True, key="cl_fr_mode")
        if not find:
            return None, "Enter a value to find."
        return T.replace_values, dict(column=col, find=find, replace=repl, regex=mode == "regex", whole=mode == "whole value")
    if op == "Calculated column":
        name = st.text_input("New column name", key="cl_calc_name")
        ex = st.text_input("Formula", placeholder="e.g. profit / revenue * 100", key="cl_calc_ex")
        st.caption("Use column names, + − × ÷, comparisons, and functions like `abs()`. Backtick names with spaces.")
        return (T.calculated_column, dict(name=name.strip(), expression=ex)) if name and ex else (None, "Enter a name and formula.")
    if op == "Bin numeric column":
        if not nums:
            return None, "No numeric columns."
        col = st.selectbox("Column", nums, key="cl_bin_col")
        bins = st.slider("Number of bins", 2, 20, 4, key="cl_bin_n")
        strat = st.radio("Strategy", ["equal width", "quantile"], horizontal=True, key="cl_bin_s")
        labels = st.text_input("Labels (optional, comma-separated)", placeholder="Low, Medium, High, Very high", key="cl_bin_l")
        name = st.text_input("New column name", f"{col}_bin", key=f"cl_bin_name_{col}")
        return T.bin_column, dict(column=col, bins=bins, strategy=strat, name=name, labels=labels or None)
    if op == "Math transform":
        if not nums:
            return None, "No numeric columns."
        col = st.selectbox("Column", nums, key="cl_m_col")
        fn = st.selectbox("Transform", T.MATH_OPS, key="cl_m_op")
        dec = st.number_input("Decimals", 0, 10, 2, key="cl_m_dec") if fn == "round" else 2
        new = st.text_input("Save as (blank = overwrite)", key="cl_m_new")
        return T.math_transform, dict(column=col, op=fn, new_name=new.strip() or None, decimals=int(dec))
    if op == "Extract date parts":
        dates = types["datetime"]
        if not dates:
            return None, "No date columns. Convert one with **Change type → date** first."
        col = st.selectbox("Date column", dates, key="cl_dt_col")
        parts = st.multiselect("Parts", T.DATE_PARTS, default=["year", "month", "weekday name"], key="cl_dt_parts")
        return (T.extract_datetime, dict(column=col, parts=parts)) if parts else (None, "Choose parts.")
    if op == "One-hot encode":
        cats = types["categorical"]
        if not cats:
            return None, "No categorical columns."
        pick = st.multiselect("Columns", cats, key="cl_oh_cols")
        df_ = st.toggle("Drop first level (avoid collinearity)", False, key="cl_oh_df")
        return (T.one_hot, dict(columns=pick, drop_first=df_)) if pick else (None, "Choose columns.")
    return None, "Unknown operation."


def render(ctx):
    df = st.session_state.df
    types = classify_columns(df) if ctx.filtered else ctx.types
    page_header("🧹", "Clean & Transform",
                "Every step is recorded, undoable and exportable as a Python script · works on the full dataset")

    raw = st.session_state.raw_df
    s_now, s_raw = get_summary(df), get_summary(raw)
    tiles([
        {"label": "Rows", "value": f"{s_now['rows']:,}", "sub": f"{s_now['rows'] - s_raw['rows']:+,} vs original",
         "status": "neutral"},
        {"label": "Columns", "value": s_now["cols"], "sub": f"{s_now['cols'] - s_raw['cols']:+} vs original", "status": "neutral"},
        {"label": "Missing cells", "value": f"{s_now['missing_cells']:,}", "sub": f"{s_now['missing_cells'] - s_raw['missing_cells']:+,} vs original",
         "status": "good" if s_now["missing_cells"] == 0 else "warn"},
        {"label": "Duplicates", "value": f"{s_now['duplicates']:,}", "sub": "✓ none" if not s_now["duplicates"] else "▲ present",
         "status": "good" if not s_now["duplicates"] else "warn"},
        {"label": "Steps applied", "value": len(st.session_state.history), "sub": "↶ undo available" if st.session_state.undo else "",
         "status": "neutral"},
    ])

    left, right = st.columns([1.25, 1], gap="large")
    with left:
        with st.container(border=True):
            group = st.segmented_control("Category", list(OPS), default=list(OPS)[0], key="p_cl_group",
                                         label_visibility="collapsed") or list(OPS)[0]
            op = st.selectbox("Operation", OPS[group], key=f"p_cl_op_{group}")
            st.caption(HELP.get(op, ""))
            fn, kwargs = _params(op, df, types)
            ready = callable(fn)
            if not ready:
                st.info(kwargs, icon="👉")
            c1, c2 = st.columns(2)
            preview = c1.button("👁 Preview", disabled=not ready, width="stretch", key="cl_preview")
            apply = c2.button("✅ Apply step", type="primary", disabled=not ready, width="stretch", key="cl_apply")
            if ready and (preview or apply):
                try:
                    new_df, label, code = fn(df, **kwargs)
                except Exception as e:
                    st.error(f"Could not apply: {e}", icon="🚫")
                else:
                    if apply:
                        state.apply_step(new_df, label, code)
                        st.toast(f"✅ {label}")
                        st.rerun()
                    st.session_state["cl_preview_result"] = (label, code, new_df.head(200), new_df.shape)
        if st.session_state.get("cl_preview_result"):
            label, code, head, shape = st.session_state.pop("cl_preview_result")
            with st.container(border=True):
                st.markdown(f"**Preview:** {label}")
                st.caption(f"Result: {shape[0]:,} rows × {shape[1]} columns (was {df.shape[0]:,} × {df.shape[1]})")
                show_df(head, height=260)
                st.code(code, language="python")

    with right:
        with st.container(border=True):
            st.markdown("##### 🧾 Pipeline")
            hist = st.session_state.history
            if not hist:
                st.caption("No steps yet. Applied operations appear here in order.")
            else:
                html_block("".join(
                    f'<div class="dv-step"><div class="dv-step-n">{i}</div><div class="dv-step-t">{esc(h["label"])}</div></div>'
                    for i, h in enumerate(hist, 1)))
            c1, c2 = st.columns(2)
            if c1.button("↶ Undo last", disabled=not st.session_state.undo, width="stretch", key="cl_undo"):
                state.undo()
                st.rerun()
            if c2.button("⟲ Reset all", disabled=not hist, width="stretch", key="cl_reset"):
                state.reset_to_original()
                st.rerun()
            script = T.pipeline_script(st.session_state.file_name, hist)
            st.download_button("⬇️ Download pipeline (.py)", script, file_name="dataviz_pipeline.py", mime="text/x-python",
                               width="stretch", disabled=not hist)
            if hist:
                with st.expander("View generated code"):
                    st.code(script, language="python")

    section("Current data", f"{len(df):,} rows × {df.shape[1]} columns")
    show_df(df.head(300), height=380)

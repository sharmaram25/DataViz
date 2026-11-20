"""
DataViz — Analytics Studio.

Run with:  streamlit run main.py
"""
from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import streamlit as st

st.set_page_config(layout="wide", page_title="DataViz · Analytics Studio", page_icon="📊",
                   initial_sidebar_state="expanded")

from src import state  # noqa: E402
from src.preprocess import classify_columns  # noqa: E402
from src.ui_components import footer, setup_page, side_label, sidebar_brand, sidebar_dataset_card  # noqa: E402
from src.views import (  # noqa: E402
    clean, dashboard, explore, export, home, overview, pivot, profiler, sql_lab, stats_lab, studio,
)

VIEW_MODULES = {
    "overview": overview, "profiler": profiler, "clean": clean, "explore": explore, "stats_lab": stats_lab,
    "pivot": pivot, "sql_lab": sql_lab, "studio": studio, "dashboard": dashboard, "export": export,
}

# label → (icon, module, short description)
PAGES = {
    "Overview": ("🏠", "overview", "Health, schema and preview"),
    "Profiler": ("🔬", "profiler", "Column-by-column deep profile"),
    "Clean & Transform": ("🧹", "clean", "Fix, reshape and enrich data"),
    "Explore": ("🧭", "explore", "Correlations, drivers, time series, ML"),
    "Stats Lab": ("🧪", "stats_lab", "Hypothesis tests in plain English"),
    "Pivot & Group": ("🧮", "pivot", "Pivot tables and aggregations"),
    "SQL Lab": ("🗄️", "sql_lab", "Query with SQL"),
    "Chart Studio": ("🎨", "studio", "Build any chart"),
    "Dashboard": ("📌", "dashboard", "Your pinned charts, live"),
    "Export & Report": ("📤", "export", "Reports, files and code"),
}


@dataclass
class Context:
    df: pd.DataFrame          # filtered view (what analysis pages use)
    full: pd.DataFrame        # working dataset (all rows, after transforms)
    types: dict               # classify_columns(df)
    name: str
    filtered: bool


def _classify(df: pd.DataFrame):
    """Column classification, memoised per dataset version + filtered view."""
    key = (st.session_state.version, id(df), len(df))
    memo = st.session_state.get("_types_memo")
    if memo and memo[0] == key:
        return memo[1]
    types = classify_columns(df)
    st.session_state["_types_memo"] = (key, types)
    return types


def render_filters(full: pd.DataFrame):
    filters = st.session_state.filters
    n_active = state.active_filter_count()
    with st.sidebar.expander(f"🔎 Global filters{f' · {n_active} active' if n_active else ''}", expanded=bool(filters)):
        st.caption("Filters apply to every analysis page. Cleaning always works on the full data.")
        cols = [c for c in full.columns if c not in {f["col"] for f in filters}]
        new_col = st.selectbox("Add a filter on…", [None] + cols, format_func=lambda v: "— choose a column —" if v is None else str(v),
                               key="flt_new_col")
        if new_col is not None:
            state.add_filter(full, new_col)
            del st.session_state["flt_new_col"]
            st.rerun()

        for f in filters:
            col, fid = f["col"], f["id"]
            s = full[col]
            head = st.columns([5, 1], vertical_alignment="center")
            head[0].markdown(f"**{col}**")
            if head[1].button("✕", key=f"flt_{fid}_rm", help="Remove filter", type="tertiary"):
                state.remove_filter(fid)
                st.rerun()
            if f["kind"] == "range":
                lo, hi = float(s.min()), float(s.max())
                if lo == hi:
                    st.caption(f"Constant value {lo}")
                    continue
                v = f["value"] or [lo, hi]
                v = [max(lo, min(hi, v[0])), max(lo, min(hi, v[1]))]
                is_int = pd.api.types.is_integer_dtype(s)
                if is_int:
                    f["value"] = list(st.slider("Range", int(lo), int(hi), (int(v[0]), int(v[1])), key=f"flt_{fid}_v",
                                                label_visibility="collapsed"))
                else:
                    f["value"] = list(st.slider("Range", lo, hi, (v[0], v[1]), key=f"flt_{fid}_v", label_visibility="collapsed"))
            elif f["kind"] == "date":
                dmin, dmax = s.min().date(), s.max().date()
                v = f["value"] or [dmin, dmax]
                picked = st.date_input("Dates", value=(v[0], v[1]), min_value=dmin, max_value=dmax, key=f"flt_{fid}_v",
                                       label_visibility="collapsed")
                if isinstance(picked, (list, tuple)) and len(picked) == 2:
                    f["value"] = [picked[0], picked[1]]
            else:
                opts = s.dropna().astype(str).value_counts().index.tolist()[:500]
                f["value"] = st.multiselect("Values", opts, default=[x for x in f["value"] if x in opts], key=f"flt_{fid}_v",
                                            placeholder="Choose values…", label_visibility="collapsed")
            c1, c2 = st.columns(2)
            f["exclude"] = c1.checkbox("Exclude", value=f.get("exclude", False), key=f"flt_{fid}_ex")
            f["keep_na"] = c2.checkbox("Keep blanks", value=f.get("keep_na", False), key=f"flt_{fid}_na")
        if filters and st.button("Clear all filters", key="flt_clear", width="stretch"):
            for f in list(filters):
                state.remove_filter(f["id"])
            st.rerun()


def render_sidebar() -> str:
    with st.sidebar:
        sidebar_brand()
        full = st.session_state.df
        view = state.get_view()
        sidebar_dataset_card(st.session_state.file_name, len(full), full.shape[1], len(view),
                             state.active_filter_count(), len(st.session_state.history))
        side_label("Workspace")
        labels = list(PAGES)
        if st.session_state.get("nav") not in labels:
            st.session_state.nav = labels[0]
        page = st.radio("Navigate", labels, key="nav", label_visibility="collapsed",
                        format_func=lambda p: f"{PAGES[p][0]}  {p}")
    render_filters(full)
    with st.sidebar:
        st.divider()
        c1, c2 = st.columns(2)
        if c1.button("↶ Undo", disabled=not st.session_state.undo, width="stretch", help="Undo the last transform"):
            state.undo()
            st.rerun()
        if c2.button("⏏ Unload", width="stretch", help="Close this dataset"):
            state.unload()
            st.rerun()
        st.caption("Built by **Ram Sharma** · runs 100% locally")
    return page


def _apply_deep_link():
    """Support shareable links such as ?sample=gapminder&page=Explore (applied once per session)."""
    if st.session_state.get("_deep_link_done"):
        return
    st.session_state["_deep_link_done"] = True
    qp = st.query_params
    sample = qp.get("sample")
    if sample and not state.has_data():
        from src.loader import SAMPLE_DATASETS, load_sample, sample_name
        if sample in SAMPLE_DATASETS:
            df, notes = load_sample(sample)
            state.set_dataset(df, sample_name(sample), notes)
    page = qp.get("page")
    if page:
        match = next((p for p in PAGES if p.lower().replace(" ", "").replace("&", "") ==
                      page.lower().replace(" ", "").replace("&", "").replace("-", "").replace("_", "")), None)
        if match:
            st.session_state.nav = match


def main():
    setup_page()
    state.init_state()

    _apply_deep_link()

    if not state.has_data():
        home.render()
        footer()
        return

    page = render_sidebar()
    view = state.get_view()
    ctx = Context(df=view, full=st.session_state.df, types=_classify(view),
                  name=st.session_state.file_name, filtered=state.active_filter_count() > 0)

    if view.empty:
        st.warning("Your global filters exclude every row. Loosen them in the sidebar.", icon="🔎")
        return

    _icon, module, _desc = PAGES[page]
    VIEW_MODULES[module].render(ctx)
    footer()


main()

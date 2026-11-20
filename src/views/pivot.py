"""Pivot & Group: interactive pivot tables and multi-metric group-bys."""
from __future__ import annotations

import io

import pandas as pd
import plotly.express as px
import streamlit as st

from src import state
from src.ui_components import page_header, section, show_chart, show_df
from src.visuals import SEQUENTIAL, apply_chart_style, build_chart

AGGS = ["sum", "mean", "median", "count", "min", "max", "std", "nunique"]


def _downloads(df: pd.DataFrame, stem: str, key: str):
    c1, c2 = st.columns(2)
    c1.download_button("⬇️ CSV", df.to_csv().encode("utf-8"), f"{stem}.csv", "text/csv", width="stretch", key=f"{key}_csv")
    buf = io.BytesIO()
    df.to_excel(buf)
    c2.download_button("⬇️ Excel", buf.getvalue(), f"{stem}.xlsx",
                       "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", width="stretch", key=f"{key}_xlsx")


def render(ctx):
    df, types = ctx.df, ctx.types
    page_header("🧮", "Pivot & Group", "Summarise data by any dimension — then chart or export the result")
    tab_pivot, tab_group = st.tabs(["🔄 Pivot table", "📚 Group-by summary"])
    dims = types["groupable"] + types["datetime"] + [c for c in types["text"] if df[c].nunique() <= 500]
    nums = types["numerical"]

    with tab_pivot:
        if not dims:
            st.info("No groupable columns.")
        else:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                rows = state.multiselect("Rows", dims, key="p_pv_rows", default=dims[:1])
            with c2:
                cols = state.multiselect("Columns", [d for d in dims if d not in rows], key="p_pv_cols")
            with c3:
                val = state.select("Values", [None] + nums, key="p_pv_val", format_func=lambda v: "Row count" if v is None else v)
            with c4:
                agg = state.select("Aggregate", AGGS, key="p_pv_agg", disabled=val is None)
            c5, c6, c7 = st.columns(3)
            totals = c5.toggle("Show totals", True, key="p_pv_tot")
            pct = c6.selectbox("Show as", ["Values", "% of total", "% of row", "% of column"], key="p_pv_pct")
            heat = c7.toggle("Heat-map colouring", True, key="p_pv_heat")
            if rows:
                try:
                    if val is None:
                        pv = pd.pivot_table(df.assign(_n=1), index=rows, columns=cols or None, values="_n", aggfunc="sum",
                                            margins=totals, margins_name="Total", observed=True, fill_value=0)
                    else:
                        pv = pd.pivot_table(df, index=rows, columns=cols or None, values=val, aggfunc=agg,
                                            margins=totals, margins_name="Total", observed=True)
                    if isinstance(pv, pd.Series):
                        pv = pv.to_frame()
                    if pct != "Values":
                        core = pv.drop(index="Total", errors="ignore").drop(columns="Total", errors="ignore")
                        if pct == "% of total":
                            pv = core / core.values.sum() * 100
                        elif pct == "% of row":
                            pv = core.div(core.sum(axis=1), axis=0) * 100
                        else:
                            pv = core.div(core.sum(axis=0), axis=1) * 100
                    if isinstance(pv.columns, pd.MultiIndex):
                        pv.columns = [" · ".join(map(str, c)).strip(" ·") for c in pv.columns]
                    st.caption(f"{pv.shape[0]:,} rows × {pv.shape[1]:,} columns")
                    fmt = "{:,.1f}%" if pct != "Values" else "{:,.4g}"
                    styled = pv.style.format(fmt, na_rep="–")
                    if heat and pv.size <= 20000:
                        core_cols = [c for c in pv.columns if c != "Total"]
                        core_rows = [r for r in pv.index if r != "Total" and not (isinstance(r, tuple) and "Total" in r)]
                        styled = styled.background_gradient(cmap="Blues", subset=pd.IndexSlice[core_rows, core_cols])
                    show_df(styled, height=min(38 * len(pv) + 40, 560))
                    _downloads(pv, "pivot_table", "pv")
                    if len(rows) == 1 and pv.shape[1] <= 60 and pv.shape[0] <= 80:
                        core = pv.drop(index="Total", errors="ignore").drop(columns="Total", errors="ignore")
                        fig = px.imshow(core, aspect="auto", color_continuous_scale=SEQUENTIAL,
                                        text_auto=".3s" if core.size <= 300 else False)
                        show_chart(apply_chart_style(fig, "<b>Pivot heatmap</b>", max(320, 26 * len(core) + 140)), key="pv_heat")
                except Exception as e:
                    st.error(f"Could not build pivot: {e}")

    with tab_group:
        if not dims:
            st.info("No groupable columns.")
            return
        by = state.multiselect("Group by", dims, key="p_gb_by", default=dims[:1])
        metrics = state.multiselect("Metrics", nums, key="p_gb_metrics", default=nums[:2])
        funcs = state.multiselect("Statistics", AGGS, key="p_gb_funcs", default=["count", "mean", "sum"])
        if by and metrics and funcs:
            g = df.groupby(by, observed=True, dropna=False)[metrics].agg(funcs)
            g.columns = [f"{m} · {f}" for m, f in g.columns]
            sort_col = state.select("Sort by", list(g.columns), key="p_gb_sort")
            g = g.sort_values(sort_col, ascending=False)
            show_df(g.style.format("{:,.4g}", na_rep="–").background_gradient(cmap="Blues", subset=[sort_col]),
                    height=min(38 * len(g) + 40, 560))
            _downloads(g, "group_summary", "gb")
            if len(by) == 1:
                section("Chart")
                m, f = sort_col.split(" · ")
                spec = {"type": "Bar", "x": by[0], "y": None if f == "count" else m, "agg": f if f != "count" else "count",
                        "options": {"top_n": 25, "sort": "Value ↓"}}
                show_chart(build_chart(df, spec).fig, key="gb_chart")

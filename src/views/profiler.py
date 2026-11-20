"""Profiler: a deep, single-column profile plus an all-columns distribution overview."""
from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from src import state
from src.eda import normality_tests, outlier_mask, smart_insights, value_counts_table
from src.preprocess import safe_nunique
from src.ui_components import (
    KIND_META, fmt_num, insight_cards, kind_chip, html_block, page_header, section, show_chart, show_df, status_for, tiles,
)
from src.visuals import CATEGORICAL, apply_chart_style, plot_histogram, plot_small_multiples, plot_value_counts


def render(ctx):
    df, types = ctx.df, ctx.types
    page_header("🔬", "Column Profiler", "Everything about one column — or a bird's-eye view of all of them")

    tab_one, tab_all = st.tabs(["🎯 Single column", "🗺️ All numeric distributions"])

    with tab_one:
        cols = list(df.columns)
        col = state.select("Column", cols, key="p_prof_col",
                           format_func=lambda c: f"{KIND_META[types['kinds'][c]][2]}  {c}")
        s = df[col]
        kind = types["kinds"][col]
        n = len(s)
        missing = int(s.isna().sum())
        uniq = safe_nunique(s)
        html_block(kind_chip(kind) + f'<span class="dv-chip">dtype · {s.dtype}</span>')

        base = [
            {"label": "Non-null", "value": f"{n - missing:,}", "sub": f"of {n:,} rows", "status": "neutral"},
            {"label": "Missing", "value": f"{missing / max(n, 1):.1%}", "sub": f"{missing:,} cells",
             "status": status_for(missing / max(n, 1) * 100, 1, 10, False)},
            {"label": "Distinct", "value": f"{uniq:,}", "sub": f"{uniq / max(n, 1):.1%} of rows", "status": "neutral"},
        ]

        if kind == "numerical":
            v = pd.to_numeric(s, errors="coerce").dropna()
            out_n = int(outlier_mask(v).sum()) if len(v) else 0
            tiles(base + [
                {"label": "Mean", "value": fmt_num(v.mean()) if len(v) else "–", "sub": f"median {fmt_num(v.median())}" if len(v) else ""},
                {"label": "Std dev", "value": fmt_num(v.std()) if len(v) > 1 else "–",
                 "sub": f"range {fmt_num(v.min())} – {fmt_num(v.max())}" if len(v) else ""},
                {"label": "Outliers (IQR)", "value": f"{out_n:,}", "sub": f"{out_n / max(len(v), 1):.1%} of values",
                 "status": status_for(out_n / max(len(v), 1) * 100, 1, 5, False)},
            ])
            c1, c2 = st.columns([1.6, 1], gap="medium")
            with c1:
                show_chart(plot_histogram(df, col), key="prof_hist")
            with c2:
                with st.container(border=True):
                    st.markdown("##### Statistics")
                    q = v.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]) if len(v) else pd.Series(dtype=float)
                    stats_df = pd.DataFrame({
                        "Statistic": ["Min", "1%", "5%", "Q1 (25%)", "Median", "Q3 (75%)", "95%", "99%", "Max",
                                      "Mean", "Std", "Skew", "Kurtosis", "Sum", "Zeros", "Negatives"],
                        "Value": [v.min(), *q.values, v.max(), v.mean(), v.std(), v.skew(), v.kurtosis(), v.sum(),
                                  int((v == 0).sum()), int((v < 0).sum())] if len(v) else [None] * 16,
                    })
                    show_df(stats_df.style.format({"Value": "{:,.4g}"}, na_rep="–"), hide_index=True, height=420)
            nt = normality_tests(s)
            if not nt.empty:
                section("Normality", "Is this variable normally distributed?")
                show_df(nt.style.format({"Statistic": "{:.4f}", "p-value": "{:.4g}"}), hide_index=True)
        elif kind == "datetime":
            v = s.dropna()
            tiles(base + [
                {"label": "Earliest", "value": f"{v.min():%Y-%m-%d}" if len(v) else "–"},
                {"label": "Latest", "value": f"{v.max():%Y-%m-%d}" if len(v) else "–"},
                {"label": "Span", "value": f"{(v.max() - v.min()).days:,} days" if len(v) else "–"},
            ])
            if len(v):
                c1, c2 = st.columns(2)
                with c1:
                    fig = px.histogram(v.to_frame(), x=col, nbins=60, color_discrete_sequence=[CATEGORICAL[0]])
                    show_chart(apply_chart_style(fig, f"<b>Records over time: {col}</b>", 380), key="prof_dt_hist")
                with c2:
                    wd = v.dt.day_name().value_counts().reindex(
                        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]).fillna(0)
                    fig = px.bar(x=wd.index, y=wd.values, labels={"x": "", "y": "Rows"}, color_discrete_sequence=[CATEGORICAL[2]])
                    show_chart(apply_chart_style(fig, "<b>Records by weekday</b>", 380), key="prof_dt_wd")
        else:
            vc = value_counts_table(s, top=25)
            top_share = vc["Share %"].iloc[0] if len(vc) else 0
            lengths = s.dropna().astype(str).str.len()
            tiles(base + [
                {"label": "Most common", "value": str(vc["Value"].iloc[0])[:18] if len(vc) else "–", "sub": f"{top_share:.1f}% of rows"},
                {"label": "Avg length", "value": f"{lengths.mean():.1f}" if len(lengths) else "–",
                 "sub": f"{lengths.min()}–{lengths.max()} chars" if len(lengths) else ""},
            ])
            c1, c2 = st.columns([1.5, 1])
            with c1:
                show_chart(plot_value_counts(vc, col), key="prof_vc")
            with c2:
                show_df(vc, hide_index=True, height=520,
                        column_config={"Share %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")})

        col_insights = [i for i in smart_insights(df) if i.get("column") == col]
        if col_insights:
            section("Diagnostics for this column")
            insight_cards(col_insights)

    with tab_all:
        nums = types["numerical"]
        if not nums:
            st.info("No numeric columns.")
        else:
            st.caption(f"Small multiples of {min(len(nums), 24)} numeric column(s) — spot skew, gaps and spikes at a glance.")
            show_chart(plot_small_multiples(df, nums), key="prof_small")

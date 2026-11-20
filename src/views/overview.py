"""Overview: dataset health at a glance."""
from __future__ import annotations

import streamlit as st

from src import state
from src.eda import categorical_profile, numeric_profile, smart_insights
from src.preprocess import get_data_quality, get_summary, schema_table
from src.ui_components import (
    KIND_META, html_block, insight_cards, kind_legend, page_header, quality_bars, section, show_chart, show_df,
    status_for, tiles,
)
from src.visuals import plot_score_gauge


def render(ctx):
    df, types = ctx.df, ctx.types
    page_header("🏠", ctx.name, "Dataset health, structure and a first look at the data"
                + (" · <b style='color:#7a5200'>filtered view</b>" if ctx.filtered else ""))

    for note in st.session_state.get("load_notes", []):
        st.toast(note, icon="🪄")
    st.session_state.load_notes = []

    summary = get_summary(df)
    quality = get_data_quality(df, types)
    score = quality["score"]
    miss_status = status_for(summary["missing_percent"], 1, 10, higher_is_better=False)
    dup_pct = summary["duplicates"] / max(summary["rows"], 1) * 100
    dup_status = status_for(dup_pct, 0, 2, higher_is_better=False)
    q_status = status_for(score, 95, 75)
    insights = smart_insights(df)
    counts = {lvl: sum(i["level"] == lvl for i in insights) for lvl in ("critical", "warning", "info", "good")}
    if counts["critical"] and q_status == "good":
        q_status = "warn"
    q_sub = {"good": "✓ Healthy", "warn": "▲ Needs attention", "crit": "! Critical issues"}[q_status]
    if counts["critical"] or counts["warning"]:
        q_sub += f" · {counts['critical'] + counts['warning']} issue(s)"

    tiles([
        {"label": "Rows", "value": f"{summary['rows']:,}",
         "sub": f"of {len(ctx.full):,} total" if ctx.filtered else "records", "status": "neutral"},
        {"label": "Columns", "value": summary["cols"],
         "sub": f"{len(types['numerical'])} numeric · {len(types['categorical'])} categorical", "status": "neutral"},
        {"label": "Missing cells", "value": f"{summary['missing_percent']:.1f}%",
         "sub": f"{summary['missing_cells']:,} cells · " + {"good": "✓ low", "warn": "▲ moderate", "crit": "! high"}[miss_status],
         "status": miss_status},
        {"label": "Duplicate rows", "value": f"{summary['duplicates']:,}",
         "sub": ("✓ none" if not summary["duplicates"] else f"▲ {dup_pct:.1f}% of rows"), "status": dup_status},
        {"label": "Memory", "value": f"{summary['memory_mb']:.1f} MB", "sub": "in-memory size", "status": "neutral"},
        {"label": "Quality score", "value": f"{score}/100", "sub": q_sub, "status": q_status},
    ])

    c1, c2 = st.columns([1, 1.5], gap="large")
    with c1:
        with st.container(border=True):
            st.markdown("##### Data quality")
            show_chart(plot_score_gauge(score), key="ov_gauge")
            quality_bars(quality["breakdown"])
    with c2:
        with st.container(border=True, height=520):
            st.markdown("##### Smart diagnostics")
            html_block(
                f'<span class="dv-badge crit">! {counts["critical"]} critical</span> '
                f'<span class="dv-badge warn">▲ {counts["warning"]} warnings</span> '
                f'<span class="dv-badge info">i {counts["info"]} notes</span><div style="height:.6rem"></div>')
            if insights:
                insight_cards(insights)
            else:
                st.success("No data-quality issues detected.", icon="✅")

    section("Columns", "Every column with its detected kind, completeness and cardinality")
    kind_legend(types)
    sch = schema_table(df, types["kinds"])
    sch["Kind"] = sch["Kind"].map(lambda k: f"{KIND_META.get(k, ('?', '', '?'))[2]}  {KIND_META.get(k, (k,))[0]}")
    show_df(
        sch, hide_index=True, height=min(38 * len(sch) + 40, 460),
        column_config={
            "Missing %": st.column_config.ProgressColumn("Missing %", min_value=0, max_value=100, format="%.1f%%"),
            "Unique %": st.column_config.ProgressColumn("Unique %", min_value=0, max_value=100, format="%.1f%%"),
            "Non-null": st.column_config.NumberColumn(format="%d"),
        },
    )

    section("Data preview", f"{len(df):,} rows")
    t1, t2, t3, t4 = st.tabs(["First rows", "Last rows", "Random sample", "🔍 Search"])
    with t1:
        n = st.slider("Rows to show", 10, 500, 50, step=10, key="p_ov_n")
        show_df(df.head(n), height=420)
    with t2:
        show_df(df.tail(50), height=420)
    with t3:
        if st.button("🎲 New sample", key="ov_resample"):
            st.session_state["ov_seed"] = st.session_state.get("ov_seed", 0) + 1
        show_df(df.sample(min(50, len(df)), random_state=st.session_state.get("ov_seed", 0)), height=420)
    with t4:
        q = st.text_input("Search all text in the table", placeholder="Type a value to find matching rows…", key="p_ov_q")
        if q:
            hits = df[df.astype(str).apply(lambda col: col.str.contains(q, case=False, regex=False, na=False)).any(axis=1)]
            st.caption(f"{len(hits):,} matching rows")
            show_df(hits.head(1000), height=420)

    section("Summary statistics")
    s1, s2 = st.tabs([f"Numeric ({len(types['numerical'])})", f"Categorical ({len(types['categorical']) + len(types['text'])})"])
    with s1:
        prof = numeric_profile(df, types["numerical"])
        if prof.empty:
            st.info("No numeric columns.")
        else:
            show_df(prof.style.format(precision=3, thousands=",").background_gradient(
                subset=["Missing %"], cmap="Reds", vmin=0, vmax=100).background_gradient(
                subset=["Outliers (IQR)"], cmap="Oranges"), height=min(38 * len(prof) + 40, 520))
    with s2:
        cp = categorical_profile(df, types["categorical"] + types["text"] + types["boolean"])
        if cp.empty:
            st.info("No categorical columns.")
        else:
            show_df(cp, height=min(38 * len(cp) + 40, 520),
                    column_config={"Top share %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")})

    with st.container(border=True):
        a, b, c = st.columns(3)
        a.markdown("**Next steps**")
        b.button("🧹 Fix issues in Clean & Transform", width="stretch", on_click=state.goto, args=("Clean & Transform",))
        c.button("🎨 Start charting in Chart Studio", width="stretch", on_click=state.goto, args=("Chart Studio",))

"""Export & Report: data files, analysis reports and reproducible code."""
from __future__ import annotations

import io
import re

import streamlit as st

from src import state, transforms as T
from src.report import build_html_report, build_markdown_report
from src.ui_components import page_header, section

XLSX = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"


def _stem(name: str) -> str:
    return re.sub(r"[^\w\-]+", "_", (name or "data").rsplit(".", 1)[0]).strip("_") or "data"


def _excel(df) -> bytes:
    buf = io.BytesIO()
    out = df.copy()
    for c in out.select_dtypes(include=["datetimetz"]).columns:
        out[c] = out[c].dt.tz_localize(None)
    out.to_excel(buf, index=False, sheet_name="data")
    return buf.getvalue()


def _parquet(df) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    return buf.getvalue()


def render(ctx):
    page_header("📤", "Export & Report", "Take your results anywhere — files, shareable reports and reproducible code")
    stem = _stem(ctx.name)

    section("1 · Data", "Download the dataset in any format")
    scope = st.radio("Which rows?", ["Filtered view", "Full working dataset", "Original upload"], horizontal=True,
                     key="p_exp_scope", index=0 if ctx.filtered else 1)
    data = {"Filtered view": ctx.df, "Full working dataset": ctx.full, "Original upload": st.session_state.raw_df}[scope]
    st.caption(f"{len(data):,} rows × {data.shape[1]} columns")
    c1, c2, c3, c4 = st.columns(4)
    c1.download_button("📄 CSV", lambda: data.to_csv(index=False).encode("utf-8"), f"{stem}.csv", "text/csv",
                       width="stretch", key="exp_csv", on_click="ignore")
    c2.download_button("📗 Excel", lambda: _excel(data), f"{stem}.xlsx", XLSX, width="stretch", key="exp_xlsx", on_click="ignore")
    c3.download_button("🧱 Parquet", lambda: _parquet(data), f"{stem}.parquet", "application/octet-stream",
                       width="stretch", key="exp_parquet", on_click="ignore")
    c4.download_button("🧾 JSON", lambda: data.to_json(orient="records", date_format="iso", indent=1),
                       f"{stem}.json", "application/json", width="stretch", key="exp_json", on_click="ignore")

    section("2 · Analysis report", "A self-contained summary you can share with anyone")
    with st.container(border=True):
        opts = {"overview": "Key metrics", "quality": "Quality score", "insights": "Diagnostics", "schema": "Column table",
                "stats": "Summary statistics", "charts": "Charts", "dashboard": "Pinned dashboard charts",
                "pipeline": "Transformation log"}
        chosen = st.pills("Include", list(opts), default=list(opts), selection_mode="multi", key="p_exp_sections",
                          format_func=lambda k: opts[k])
        embed = st.toggle("Embed chart library (works offline, ≈ +4 MB)", True, key="p_exp_embed")
        c1, c2 = st.columns(2)
        hist, pinned = st.session_state.history, st.session_state.dashboard
        c1.download_button("🌐 HTML report (interactive)",
                           lambda: build_html_report(ctx.df, ctx.name, hist, pinned, embed, set(chosen or [])),
                           f"{stem}_report.html", "text/html", type="primary", width="stretch", key="exp_html",
                           on_click="ignore")
        c2.download_button("📝 Markdown report", lambda: build_markdown_report(ctx.df, ctx.name, hist),
                           f"{stem}_report.md", "text/markdown", width="stretch", key="exp_md", on_click="ignore")
        st.caption("Reports describe the current view" + (" (filters applied)." if ctx.filtered else "."))

    section("3 · Reproducible code", "Re-run your entire preparation on new data")
    script = T.pipeline_script(ctx.name, st.session_state.history, state.filters_as_code(st.session_state.filters))
    c1, c2 = st.columns([1, 3])
    c1.download_button("🐍 Pipeline script (.py)", script, f"{stem}_pipeline.py", "text/x-python", width="stretch",
                       key="exp_py")
    with c2.expander("Preview script", expanded=False):
        st.code(script, language="python")

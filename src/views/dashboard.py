"""Dashboard: pinned charts, re-rendered live from the (filtered) data."""
from __future__ import annotations

import html
import json

import plotly.io as pio
import streamlit as st

from src import state
from src.ui_components import empty_state, page_header, show_chart, tiles
from src.visuals import build_chart


def dashboard_html(charts, df, title: str) -> str:
    parts = []
    for i, c in enumerate(charts):
        res = build_chart(df, c["spec"])
        if res.fig is not None:
            parts.append(pio.to_html(res.fig, include_plotlyjs=(i == 0), full_html=False))
    body = "".join(f'<div class="card">{p}</div>' for p in parts)
    title = html.escape(title)
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>{title}</title>
<meta name="viewport" content="width=device-width, initial-scale=1">
<style>body{{font-family:Inter,system-ui,sans-serif;background:#f7f7f5;margin:0;padding:24px;color:#1b1b1a}}
h1{{font-size:24px;margin:0 0 16px}} .grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(560px,1fr));gap:16px}}
.card{{background:#fff;border:1px solid #e1e0d9;border-radius:14px;padding:8px}}
@media (max-width:640px){{.grid{{grid-template-columns:1fr}}body{{padding:16px}}}}</style></head>
<body><h1>{title}</h1><div class="grid">{body}</div></body></html>"""


def render(ctx):
    df = ctx.df
    page_header("📌", "Dashboard", "Charts pinned from Chart Studio — they update live with your global filters")
    charts = st.session_state.dashboard

    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    cols_per_row = c1.segmented_control("Layout", [1, 2, 3], default=2, key="p_db_cols",
                                        format_func=lambda n: f"{n} per row") or 2
    c2.download_button("⬇️ Export HTML", data=lambda: dashboard_html(charts, df, f"DataViz dashboard · {ctx.name}"),
                       file_name="dataviz_dashboard.html", mime="text/html", disabled=not charts, width="stretch",
                       key="db_html", on_click="ignore")
    c3.download_button("💾 Save layout", data=json.dumps([c["spec"] for c in charts], indent=2, default=str),
                       file_name="dashboard_layout.json", mime="application/json", disabled=not charts, width="stretch",
                       key="db_save", on_click="ignore")
    with c4.popover("📂 Load layout", width="stretch"):
        up = st.file_uploader("Layout JSON", type=["json"], key="db_load", label_visibility="collapsed")
        if up is not None and st.button("Load", key="db_load_btn"):
            try:
                for spec in json.loads(up.getvalue()):
                    state.pin_chart(spec)
                st.rerun()
            except Exception as e:
                st.error(f"Invalid layout file: {e}")

    if not charts:
        empty_state("📌", "No charts pinned yet.<br>Build a chart in <b>Chart Studio</b> and click <b>📌 Pin to dashboard</b>.")
        st.button("🎨 Open Chart Studio", type="primary", on_click=state.goto, args=("Chart Studio",), key="db_go")
        return

    tiles([
        {"label": "Charts", "value": len(charts), "status": "neutral"},
        {"label": "Rows in view", "value": f"{len(df):,}", "sub": "filtered" if ctx.filtered else "all rows",
         "status": "warn" if ctx.filtered else "neutral"},
    ])

    for start in range(0, len(charts), cols_per_row):
        row = st.columns(cols_per_row, gap="medium")
        for col, item in zip(row, charts[start:start + cols_per_row]):
            with col:
                spec = dict(item["spec"])
                if cols_per_row > 1:
                    spec["height"] = min(spec.get("height") or 460, 460)
                try:
                    res = build_chart(df, spec)
                except Exception as e:
                    res = None
                    st.warning(f"Chart failed: {e}")
                if res is not None and res.fig is not None:
                    show_chart(res.fig, key=f"db_{item['id']}")
                elif res is not None:
                    st.info(" ".join(res.notes))
                b1, b2, b3 = st.columns(3)
                idx = charts.index(item)
                if b1.button("◀", key=f"db_l_{item['id']}", disabled=idx == 0, width="stretch", help="Move earlier"):
                    charts[idx - 1], charts[idx] = charts[idx], charts[idx - 1]
                    st.rerun()
                if b2.button("▶", key=f"db_r_{item['id']}", disabled=idx == len(charts) - 1, width="stretch", help="Move later"):
                    charts[idx + 1], charts[idx] = charts[idx], charts[idx + 1]
                    st.rerun()
                if b3.button("🗑", key=f"db_x_{item['id']}", width="stretch", help="Remove"):
                    state.unpin_chart(item["id"])
                    st.rerun()

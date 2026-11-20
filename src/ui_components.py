"""Reusable UI building blocks (HTML components styled by assets/style.css)."""
from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

ASSETS = Path(__file__).resolve().parent.parent / "assets"

KIND_META = {
    "numerical": ("Numeric", "var(--dv-k-num)", "#"),
    "categorical": ("Category", "var(--dv-k-cat)", "◆"),
    "datetime": ("Date", "var(--dv-k-date)", "◷"),
    "text": ("Text", "var(--dv-k-text)", "T"),
    "boolean": ("Boolean", "var(--dv-k-bool)", "◐"),
}
LEVEL_META = {
    "critical": ("crit", "!", "Critical"),
    "warning": ("warning", "▲", "Warning"),
    "info": ("info", "i", "Note"),
    "good": ("good", "✓", "Good"),
}
PLOTLY_CONFIG = {
    "displaylogo": False,
    "toImageButtonOptions": {"format": "png", "scale": 2, "filename": "dataviz_chart"},
    "modeBarButtonsToRemove": ["lasso2d", "select2d"],
}

esc = lambda v: html.escape(str(v))  # noqa: E731


def setup_page():
    css = (ASSETS / "style.css").read_text(encoding="utf-8") if (ASSETS / "style.css").exists() else ""
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def html_block(markup: str):
    st.markdown(markup, unsafe_allow_html=True)


LOGO_SVG = (
    '<svg width="22" height="22" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">'
    '<rect x="3" y="12" width="4" height="9" rx="1.5" fill="white" opacity=".75"/>'
    '<rect x="10" y="7" width="4" height="14" rx="1.5" fill="white" opacity=".9"/>'
    '<rect x="17" y="3" width="4" height="18" rx="1.5" fill="white"/></svg>'
)


def sidebar_brand():
    html_block(
        f'<div class="dv-brand"><div class="dv-logo">{LOGO_SVG}</div>'
        '<div><div class="dv-brand-name">DataViz</div><div class="dv-brand-tag">Analytics Studio · v3</div></div></div>'
    )


def sidebar_dataset_card(name: str, rows: int, cols: int, view_rows: int, n_filters: int, n_steps: int):
    filt = ""
    if n_filters:
        filt = f'<div class="dv-ds-filter">⏷ {n_filters} filter(s) · showing {view_rows:,} of {rows:,} rows</div>'
    steps = f" · {n_steps} transform(s)" if n_steps else ""
    html_block(
        f'<div class="dv-ds-card"><div class="dv-ds-name">📄 {esc(name)}</div>'
        f'<div class="dv-ds-meta">{rows:,} rows × {cols} columns{steps}</div>{filt}</div>'
    )


def side_label(text: str):
    html_block(f'<div class="dv-side-label">{esc(text)}</div>')


def page_header(icon: str, title: str, subtitle: str = ""):
    html_block(
        f'<div class="dv-page-head"><div class="dv-page-icon">{icon}</div><div>'
        f'<div class="dv-page-title">{esc(title)}</div>'
        f'<div class="dv-page-sub">{subtitle}</div></div></div>'
    )


def section(title: str, caption: str = ""):
    html_block(f'<div class="dv-section"><div class="dv-section-title">{esc(title)}</div>'
               f'<div class="dv-section-cap">{caption}</div></div>')


def tiles(items: list[dict]):
    """items: {label, value, sub?, status? ∈ good|warn|crit|neutral|None}"""
    parts = []
    for it in items:
        cls = it.get("status") or ""
        sub = f'<div class="dv-tile-sub">{it["sub"]}</div>' if it.get("sub") else ""
        parts.append(
            f'<div class="dv-tile {cls}"><div class="dv-tile-label">{esc(it["label"])}</div>'
            f'<div class="dv-tile-value">{esc(it["value"])}</div>{sub}</div>'
        )
    html_block(f'<div class="dv-tiles">{"".join(parts)}</div>')


def badge(text: str, level: str = "neutral") -> str:
    return f'<span class="dv-badge {level}">{text}</span>'


def kind_chip(kind: str, label: str | None = None) -> str:
    name, color, _ = KIND_META.get(kind, ("?", "#898781", "?"))
    return f'<span class="dv-chip" style="--c:{color}"><i></i>{esc(label or name)}</span>'


def kind_legend(types: dict):
    chips = [kind_chip(k, f"{KIND_META[k][0]} · {len(types.get(k, []))}") for k in KIND_META if types.get(k)]
    html_block("<div>" + "".join(chips) + "</div>")


def insight_cards(insights: list[dict], limit: int | None = None):
    for ins in insights[:limit] if limit else insights:
        cls, icon, label = LEVEL_META.get(ins["level"], LEVEL_META["info"])
        html_block(
            f'<div class="dv-insight {cls}"><div class="dv-insight-icon" title="{label}">{icon}</div>'
            f'<div><div class="dv-insight-title">{esc(ins["title"])}</div>'
            f'<div class="dv-insight-text">{_md_bold(ins["detail"])}</div></div></div>'
        )


def _md_bold(text: str) -> str:
    """Escape, then render **bold** segments."""
    out, parts = [], esc(text).split("**")
    for i, p in enumerate(parts):
        out.append(f"<b>{p}</b>" if i % 2 else p)
    return "".join(out)


def status_for(value: float, good: float, warn: float, higher_is_better: bool = True) -> str:
    if higher_is_better:
        return "good" if value >= good else "warn" if value >= warn else "crit"
    return "good" if value <= good else "warn" if value <= warn else "crit"


STATUS_LABEL = {"good": "✓ Healthy", "warn": "▲ Needs attention", "crit": "! Critical"}
STATUS_COLOR = {"good": "var(--dv-good)", "warn": "var(--dv-warn)", "crit": "var(--dv-crit)"}


def quality_bars(breakdown: dict):
    rows = []
    for name, (val, weight) in breakdown.items():
        pct = val * 100
        stt = status_for(pct, 95, 80)
        rows.append(
            f'<div class="dv-qbar"><div class="dv-qbar-head"><span>{esc(name)} <span style="color:var(--dv-muted);font-weight:500">'
            f'(weight {weight})</span></span><span>{pct:.1f}%</span></div>'
            f'<div class="dv-qbar-track"><div class="dv-qbar-fill" style="width:{pct:.1f}%;--c:{STATUS_COLOR[stt]}"></div></div></div>'
        )
    html_block("".join(rows))


def result_card(level: str, title: str, text: str):
    cls = {"critical": "crit"}.get(level, level)
    html_block(f'<div class="dv-result {cls}"><div class="dv-result-title">{title}</div>'
               f'<div class="dv-result-text">{text}</div></div>')


def empty_state(icon: str, message: str):
    html_block(f'<div class="dv-empty"><b>{icon}</b>{message}</div>')


def show_chart(fig, key: str | None = None, **kwargs):
    """Render a Plotly figure with the DataViz template (not Streamlit's theme)."""
    if fig is None:
        return None
    return st.plotly_chart(fig, theme=None, config=PLOTLY_CONFIG, key=key, **kwargs)


def show_df(df: pd.DataFrame, height: int | str = "auto", **kwargs):
    return st.dataframe(df, height=height, **kwargs)


def fmt_num(v) -> str:
    try:
        v = float(v)
    except (TypeError, ValueError):
        return str(v)
    a = abs(v)
    if a >= 1e9:
        return f"{v / 1e9:.2f}B"
    if a >= 1e6:
        return f"{v / 1e6:.2f}M"
    if a >= 1e4:
        return f"{v / 1e3:.1f}K"
    if a >= 100 or v == int(v):
        return f"{v:,.0f}"
    return f"{v:,.3g}"


def footer():
    html_block('<div class="dv-footer">DataViz Analytics Studio · Built by <b>Ram Sharma</b> · '
               'Everything runs locally — your data never leaves this machine.</div>')

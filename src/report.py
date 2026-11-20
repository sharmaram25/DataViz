"""Report generation: self-contained HTML (with interactive charts) and Markdown."""
from __future__ import annotations

import html
from datetime import datetime

import pandas as pd
import plotly.io as pio

from src import eda
from src.preprocess import classify_columns, get_data_quality, get_summary, schema_table
from src.visuals import (
    build_chart, plot_heatmap, plot_missing_bars, plot_small_multiples, plot_value_counts,
)

esc = lambda v: html.escape(str(v))  # noqa: E731
LEVEL = {"critical": ("#d03b3b", "#fcebeb", "!", "Critical"), "warning": ("#fab219", "#fff6df", "▲", "Warning"),
         "info": ("#2a78d6", "#eaf2fd", "i", "Note"), "good": ("#0ca30c", "#e9f6e9", "✓", "Good")}


def _status(v, good, warn, higher=True):
    if higher:
        return "#0ca30c" if v >= good else "#fab219" if v >= warn else "#d03b3b"
    return "#0ca30c" if v <= good else "#fab219" if v <= warn else "#d03b3b"


def _table(df: pd.DataFrame, index=True, max_rows=200) -> str:
    return df.head(max_rows).to_html(index=index, border=0, classes="tbl", float_format=lambda x: f"{x:,.4g}", na_rep="–")


def build_html_report(df: pd.DataFrame, name: str, history: list[dict] | None = None, pinned: list[dict] | None = None,
                      embed_js: bool = True, sections: set[str] | None = None) -> str:
    sections = sections or {"overview", "quality", "insights", "schema", "stats", "charts", "dashboard", "pipeline"}
    types = classify_columns(df)
    summary = get_summary(df)
    quality = get_data_quality(df, types)
    insights = eda.smart_insights(df)
    figs: list[tuple[str, object]] = []
    first_js = {"done": False}

    def fig_html(fig) -> str:
        inc = (embed_js and not first_js["done"]) or ("cdn" if not embed_js and not first_js["done"] else False)
        first_js["done"] = True
        return pio.to_html(fig, include_plotlyjs=inc, full_html=False, config={"displaylogo": False})

    out = []
    score = quality["score"]
    if "overview" in sections:
        dup_pct = summary["duplicates"] / max(summary["rows"], 1) * 100
        kpis = [
            ("Rows", f"{summary['rows']:,}", "#c3c2b7"), ("Columns", summary["cols"], "#c3c2b7"),
            ("Missing", f"{summary['missing_percent']:.1f}%", _status(summary["missing_percent"], 1, 10, False)),
            ("Duplicates", f"{summary['duplicates']:,}", _status(dup_pct, 0, 2, False)),
            ("Quality score", f"{score}/100", _status(score, 95, 75)),
        ]
        out.append('<div class="kpis">' + "".join(
            f'<div class="kpi" style="--c:{c}"><span>{esc(k)}</span><b>{esc(v)}</b></div>' for k, v, c in kpis) + "</div>")
        chips = " ".join(f'<span class="chip">{esc(k)} · {len(types[k])}</span>'
                         for k in ("numerical", "categorical", "datetime", "text", "boolean") if types[k])
        out.append(f"<p>{chips}</p>")

    if "quality" in sections:
        bars = "".join(
            f'<div class="qb"><div class="qh"><span>{esc(k)} <i>(weight {w})</i></span><span>{v * 100:.1f}%</span></div>'
            f'<div class="qt"><div style="width:{v * 100:.1f}%;background:{_status(v * 100, 95, 80)}"></div></div></div>'
            for k, (v, w) in quality["breakdown"].items())
        out.append(f"<h2>Data quality</h2>{bars}")

    if "insights" in sections:
        cards = "".join(
            f'<div class="ins" style="--c:{LEVEL[i["level"]][0]};--bg:{LEVEL[i["level"]][1]}">'
            f'<div class="ii">{LEVEL[i["level"]][2]}</div><div><b>{esc(i["title"])}</b><br>'
            f'<span>{esc(i["detail"]).replace("**", "")}</span></div></div>' for i in insights) or "<p>No issues detected.</p>"
        out.append(f"<h2>Diagnostics</h2>{cards}")

    if "schema" in sections:
        out.append(f"<h2>Columns</h2>{_table(schema_table(df, types['kinds']), index=False)}")

    if "stats" in sections:
        prof = eda.numeric_profile(df, types["numerical"])
        if not prof.empty:
            out.append(f"<h2>Numeric summary</h2>{_table(prof)}")
        cp = eda.categorical_profile(df, types["categorical"] + types["boolean"])
        if not cp.empty:
            out.append(f"<h2>Categorical summary</h2>{_table(cp)}")

    if "charts" in sections:
        out.append("<h2>Visual analysis</h2>")
        mb = plot_missing_bars(df)
        if mb is not None:
            figs.append(("Missing values", mb))
        if len(types["numerical"]) >= 2:
            corr = eda.correlation_matrix(df, types["numerical"][:25])
            figs.append(("Correlation", plot_heatmap(corr, "Pearson correlation")))
        if types["numerical"]:
            figs.append(("Distributions", plot_small_multiples(df, types["numerical"][:12], cols=3)))
        for c in types["categorical"][:4]:
            figs.append((c, plot_value_counts(eda.value_counts_table(df[c], 12), c)))
        out.append('<div class="grid">' + "".join(f'<div class="card">{fig_html(f)}</div>' for _t, f in figs) + "</div>")

    if "dashboard" in sections and pinned:
        out.append("<h2>Pinned charts</h2>")
        cards = []
        for item in pinned:
            res = build_chart(df, item["spec"])
            if res.fig is not None:
                cards.append(f'<div class="card">{fig_html(res.fig)}</div>')
        out.append('<div class="grid">' + "".join(cards) + "</div>")

    if "pipeline" in sections and history:
        steps = "".join(f"<li>{esc(h['label'])}<pre>{esc(h['code'])}</pre></li>" for h in history)
        out.append(f"<h2>Transformations applied</h2><ol class='steps'>{steps}</ol>")

    generated = datetime.now().strftime("%Y-%m-%d %H:%M")
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>DataViz report · {esc(name)}</title>
<style>
:root{{color-scheme:light;--ink:#1b1b1a;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--bg:#f7f7f5;--card:#fff}}
*{{box-sizing:border-box}} body{{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 Inter,system-ui,-apple-system,"Segoe UI",sans-serif}}
header{{background:linear-gradient(135deg,#0f1c2e,#13263f);color:#fff;padding:36px 40px}}
header h1{{margin:0;font-size:28px;letter-spacing:-.02em}} header p{{margin:6px 0 0;color:#9fb3cb}}
main{{max-width:1240px;margin:0 auto;padding:28px 40px 60px}}
h2{{font-size:19px;margin:34px 0 12px;letter-spacing:-.01em}}
.kpis{{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}}
.kpi{{background:var(--card);border:1px solid var(--line);border-left:4px solid var(--c);border-radius:12px;padding:12px 14px}}
.kpi span{{display:block;font-size:12px;text-transform:uppercase;letter-spacing:.04em;color:var(--ink2);font-weight:600}}
.kpi b{{font-size:24px}}
.chip{{display:inline-block;border:1px solid var(--line);background:#fff;border-radius:99px;padding:2px 10px;font-size:12px;color:var(--ink2);margin:8px 4px 0 0}}
.qb{{margin:8px 0 12px;max-width:640px}} .qh{{display:flex;justify-content:space-between;font-size:13px;color:var(--ink2);font-weight:600}}
.qh i{{color:var(--muted);font-weight:400}} .qt{{height:8px;background:#efeee9;border-radius:99px;overflow:hidden;margin-top:4px}} .qt div{{height:100%;border-radius:99px}}
.ins{{display:flex;gap:12px;background:var(--card);border:1px solid var(--line);border-left:4px solid var(--c);border-radius:10px;padding:10px 12px;margin-bottom:8px}}
.ins .ii{{width:26px;height:26px;border-radius:7px;background:var(--bg);display:grid;place-items:center;font-weight:800;flex:none;background:var(--bg)}}
.ins span{{color:var(--ink2);font-size:14px}}
.tbl{{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;font-size:13px;display:block;overflow-x:auto}}
.tbl th{{background:#f3f2ee;color:var(--ink2);text-align:left;padding:7px 10px;font-weight:600;white-space:nowrap}}
.tbl td{{padding:6px 10px;border-top:1px solid #efeee9;white-space:nowrap;font-variant-numeric:tabular-nums}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(520px,1fr));gap:14px}}
.card{{background:var(--card);border:1px solid var(--line);border-radius:14px;padding:8px;overflow:hidden}}
.steps li{{margin-bottom:10px}} pre{{background:#0f1c2e;color:#dfe6ef;padding:10px 12px;border-radius:8px;overflow-x:auto;font-size:12px}}
footer{{color:var(--muted);font-size:12px;text-align:center;padding:20px}}
@media (max-width:700px){{main,header{{padding-left:16px;padding-right:16px}} .grid{{grid-template-columns:1fr}}}}
</style></head><body>
<header><h1>📊 {esc(name)}</h1><p>Automated analysis report · generated {generated} by DataViz Analytics Studio</p></header>
<main>{''.join(out)}</main>
<footer>DataViz Analytics Studio · Built by Ram Sharma</footer>
</body></html>"""


def build_markdown_report(df: pd.DataFrame, name: str, history: list[dict] | None = None) -> str:
    types = classify_columns(df)
    s = get_summary(df)
    q = get_data_quality(df, types)
    insights = eda.smart_insights(df)
    icon = {"critical": "🔴", "warning": "🟡", "info": "🔵", "good": "🟢"}
    lines = [
        f"# 📊 Analysis report: {name}",
        f"*Generated {datetime.now():%Y-%m-%d %H:%M} by DataViz Analytics Studio*", "",
        "## 1. Overview", "",
        "| Metric | Value |", "|---|---|",
        f"| Rows | {s['rows']:,} |", f"| Columns | {s['cols']} |",
        f"| Missing cells | {s['missing_cells']:,} ({s['missing_percent']:.2f}%) |",
        f"| Duplicate rows | {s['duplicates']:,} |", f"| Memory | {s['memory_mb']:.2f} MB |",
        f"| **Quality score** | **{q['score']}/100** |", "",
        "### Quality breakdown", "",
    ]
    lines += [f"- **{k}** (weight {w}): {v * 100:.1f}%" for k, (v, w) in q["breakdown"].items()]
    lines += ["", "## 2. Diagnostics", ""]
    lines += [f"- {icon[i['level']]} **{i['title']}** — {i['detail']}" for i in insights] or ["- No issues detected."]
    lines += ["", "## 3. Columns", "", schema_table(df, types["kinds"]).to_markdown(index=False)]
    prof = eda.numeric_profile(df, types["numerical"])
    if not prof.empty:
        lines += ["", "## 4. Numeric summary", "", prof.round(4).to_markdown()]
    cp = eda.categorical_profile(df, types["categorical"] + types["boolean"])
    if not cp.empty:
        lines += ["", "## 5. Categorical summary", "", cp.to_markdown()]
    if len(types["numerical"]) >= 2:
        pairs = eda.top_correlation_pairs(eda.correlation_matrix(df, types["numerical"][:40]), 10)
        if not pairs.empty:
            lines += ["", "## 6. Strongest correlations", "", pairs.round(3).to_markdown(index=False)]
    if history:
        lines += ["", "## 7. Transformations applied", ""]
        for i, h in enumerate(history, 1):
            lines += [f"{i}. {h['label']}", "", "```python", h["code"], "```", ""]
    return "\n".join(lines)

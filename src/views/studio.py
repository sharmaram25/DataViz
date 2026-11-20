"""Chart Studio: build any of 24 chart types, then pin, export or copy the code."""
from __future__ import annotations

import streamlit as st

from src import state
from src.ui_components import empty_state, page_header, show_chart
from src.visuals import (
    AGG_FUNCS, CHART_FAMILIES, CHART_FIELDS, CONTINUOUS_SCALES, MAX_POINTS, PALETTES, build_chart,
)

CHART_ICONS = {
    "Bar": "📊", "Line": "📈", "Area": "🏔️", "Heatmap (pivot)": "🟦", "Funnel": "🔻", "Scatter": "⚬", "Bubble": "🫧",
    "3D Scatter": "🧊", "Scatter Matrix": "🔢", "Parallel Coordinates": "〰️", "Parallel Categories": "🔀",
    "Histogram": "📶", "Box": "📦", "Violin": "🎻", "Strip": "┆", "ECDF": "⤴️", "Density Heatmap": "🌡️",
    "Density Contour": "🗺️", "Pie / Donut": "🍩", "Sunburst": "☀️", "Treemap": "🧱", "Icicle": "🧊",
    "Map (points)": "📍", "Choropleth (countries)": "🌍",
}
NONE = "— none —"


def _options_for(kind: str, df, types) -> list:
    if kind == "any":
        return list(df.columns)
    if kind == "num":
        return types["numerical"]
    if kind == "group":
        return [c for c in types["groupable"] if df[c].nunique() <= 60]
    if kind == "group_or_date":
        return types["groupable"] + types["datetime"]
    if kind == "multi_num":
        return types["numerical"]
    if kind == "multi_cat":
        return [c for c in types["categorical"] + types["boolean"] + types["groupable"] if df[c].nunique() <= 60]
    return list(df.columns)


def _default(chart: str, field: str, opts: list, types: dict, taken: set):
    """Smart defaults so a sensible chart appears immediately."""
    free = [o for o in opts if o not in taken]
    if not free:
        return None
    nums, cats, dates = types["numerical"], types["categorical"] + types["boolean"], types["datetime"]
    pick = lambda pool: next((c for c in pool if c in free), free[0])  # noqa: E731
    lowered = {c: str(c).lower() for c in free}
    if field == "x":
        if chart in ("Line", "Area"):
            return pick(dates + cats + nums)
        if chart in ("Bar", "Funnel", "Heatmap (pivot)"):
            return pick(cats + dates)
        return pick(nums)
    if field == "y":
        if chart == "Heatmap (pivot)":
            return pick([c for c in cats if c not in taken])
        if chart in ("Bar", "Line", "Area", "Funnel"):
            return pick(nums)
        return pick(nums)
    if field == "z" and chart == "3D Scatter":
        return pick(nums)
    if field == "size" and chart == "Bubble":
        return pick(nums)
    if field == "names":
        return pick(cats)
    if field == "dimensions":
        return free[: 4 if chart != "Parallel Categories" else 3]
    if field == "path":
        return [c for c in cats if c in free][:2] or free[:1]
    if field == "lat":
        return next((c for c in free if lowered[c] in ("lat", "latitude")), None)
    if field == "lon":
        return next((c for c in free if lowered[c] in ("lon", "lng", "long", "longitude")), None)
    if field == "locations":
        return next((c for c in free if lowered[c] in ("iso_alpha", "iso3", "iso", "country", "country_code")), free[0])
    return None


def render(ctx):
    df, types = ctx.df, ctx.types
    page_header("🎨", "Chart Studio", "Pick a chart, map your columns — aggregation, sampling and styling are handled for you")

    left, right = st.columns([1, 2.6], gap="large")
    with left:
        with st.container(border=True):
            family = st.segmented_control("Family", list(CHART_FAMILIES), default="Comparison", key="p_st_family",
                                          label_visibility="collapsed") or "Comparison"
            chart = st.pills("Chart type", CHART_FAMILIES[family], default=CHART_FAMILIES[family][0], key=f"p_st_type_{family}",
                             format_func=lambda c: f"{CHART_ICONS.get(c, '')} {c}") or CHART_FAMILIES[family][0]
            st.divider()
            spec: dict = {"type": chart, "options": {}}
            taken: set = set()
            for field, (label, kind, required) in CHART_FIELDS[chart].items():
                opts = _options_for(kind, df, types)
                key = f"p_st_{chart}_{field}"
                if kind.startswith("multi"):
                    val = state.multiselect(label, opts, key=key, default=_default(chart, field, opts, types, taken) or [])
                    spec[field] = val
                    taken.update(val)
                else:
                    choices = opts if required else [None] + opts
                    dflt = _default(chart, field, opts, types, taken) if required or field in ("lat", "lon") else None
                    val = state.select(label, choices, key=key, default=dflt,
                                       format_func=lambda v: NONE if v is None else str(v))
                    spec[field] = val
                    if val:
                        taken.add(val)

            o = spec["options"]
            uses_agg = chart in ("Bar", "Line", "Area", "Funnel", "Heatmap (pivot)", "Pie / Donut", "Choropleth (countries)")
            value_field = {"Heatmap (pivot)": "z", "Pie / Donut": "values", "Choropleth (countries)": "z"}.get(chart, "y")
            if uses_agg and spec.get(value_field):
                spec["agg"] = state.select("Aggregation", AGG_FUNCS, key=f"p_st_{chart}_agg",
                                           default="mean" if chart == "Heatmap (pivot)" else "sum")

            with st.expander("⚙️ Chart options", expanded=False):
                k = f"p_st_{chart}_o_"
                if chart == "Bar":
                    o["orientation"] = st.radio("Orientation", ["Vertical", "Horizontal"], horizontal=True, key=k + "orient")
                    if spec.get("color"):
                        o["barmode"] = st.radio("Bar mode", ["group", "stack", "relative"], horizontal=True, key=k + "mode")
                    o["sort"] = st.selectbox("Sort", ["Value ↓", "Value ↑", "Label A→Z", "Label Z→A", "None"], key=k + "sort")
                    o["top_n"] = st.number_input("Top N categories (0 = all)", 0, 500, 20, key=k + "top")
                    o["text"] = st.toggle("Value labels", False, key=k + "text")
                elif chart in ("Line", "Area"):
                    if chart == "Line":
                        o["markers"] = st.toggle("Markers", False, key=k + "markers")
                        o["rolling"] = st.slider("Rolling mean (points)", 0, 60, 0, key=k + "roll")
                elif chart in ("Scatter", "Bubble"):
                    o["trendline"] = st.selectbox("Trendline", ["None", "OLS (linear)", "LOWESS (smooth)"], key=k + "trend")
                    if spec.get("color") and o["trendline"] != "None":
                        o["trend_scope"] = st.radio("Trend per", ["Each colour", "Overall"], horizontal=True, key=k + "tscope")
                    c1, c2 = st.columns(2)
                    o["log_x"] = c1.toggle("Log X", False, key=k + "lx")
                    o["log_y"] = c2.toggle("Log Y", False, key=k + "ly")
                    o["opacity"] = st.slider("Opacity", 0.1, 1.0, 0.8, key=k + "op")
                    if chart == "Scatter":
                        c1, c2 = st.columns(2)
                        o["marginal_x"] = c1.selectbox("Margin X", ["None", "histogram", "box", "violin", "rug"], key=k + "mx")
                        o["marginal_y"] = c2.selectbox("Margin Y", ["None", "histogram", "box", "violin", "rug"], key=k + "my")
                    o["max_points"] = st.select_slider("Max points drawn", [2000, 5000, 10000, 20000, 50000], MAX_POINTS, key=k + "mp")
                elif chart == "Histogram":
                    o["bins"] = st.slider("Bins", 5, 200, 40, key=k + "bins")
                    o["histnorm"] = st.selectbox("Normalisation", ["count", "percent", "probability density"], key=k + "norm")
                    if spec.get("color"):
                        o["hist_barmode"] = st.radio("Overlap", ["overlay", "stack", "group"], horizontal=True, key=k + "bm")
                    o["marginal"] = st.selectbox("Marginal", ["None", "box", "violin", "rug"], key=k + "marg")
                    c1, c2 = st.columns(2)
                    o["cumulative"] = c1.toggle("Cumulative", False, key=k + "cum")
                    o["log_y"] = c2.toggle("Log Y", False, key=k + "ly")
                elif chart in ("Box", "Violin", "Strip"):
                    if chart != "Strip":
                        o["points"] = st.radio("Points", ["outliers", "all", "none"], horizontal=True, key=k + "pts")
                    if chart == "Box":
                        o["notched"] = st.toggle("Notched (median CI)", False, key=k + "notch")
                    o["log_y"] = st.toggle("Log Y", False, key=k + "ly")
                elif chart == "Density Heatmap":
                    o["bins"] = st.slider("Bins per axis", 10, 120, 40, key=k + "bins")
                elif chart == "Pie / Donut":
                    o["top_n"] = st.slider("Max slices (rest → Other)", 2, 12, 6, key=k + "top")
                    o["donut"] = st.toggle("Donut", True, key=k + "donut")
                elif chart in ("3D Scatter",):
                    o["opacity"] = st.slider("Opacity", 0.1, 1.0, 0.8, key=k + "op")
                    o["max_points"] = st.select_slider("Max points drawn", [2000, 5000, 10000, 20000], 5000, key=k + "mp")
                else:
                    st.caption("No extra options for this chart.")

            with st.expander("🖌️ Style", expanded=False):
                k = "p_st_style_"
                spec["title"] = st.text_input("Title (blank = automatic)", key=f"p_st_{chart}_title")
                spec["height"] = st.slider("Height (px)", 300, 1100, 540, 20, key=k + "h")
                spec["palette"] = st.selectbox("Categorical palette", list(PALETTES), key=k + "pal",
                                               help="The default palette is ordered for colour-blind safety.")
                spec["scale"] = st.selectbox("Continuous scale", list(CONTINUOUS_SCALES), key=k + "scale")
                o["legend"] = st.radio("Legend", ["Right", "Top", "Bottom", "Hidden"], horizontal=True, key=k + "leg")
                o["grid"] = st.toggle("Gridlines", True, key=k + "grid")
                c1, c2 = st.columns(2)
                o["x_title"] = c1.text_input("X axis title", key=f"p_st_{chart}_xt")
                o["y_title"] = c2.text_input("Y axis title", key=f"p_st_{chart}_yt")

    with right:
        try:
            res = build_chart(df, spec)
        except Exception as e:
            res = None
            empty_state("⚠️", f"This combination can't be drawn: <b>{str(e)[:300]}</b><br>Try different columns or options.")
        if res is not None and res.fig is None:
            empty_state("👈", " ".join(res.notes) or "Configure the chart on the left.")
        elif res is not None:
            show_chart(res.fig, key="studio_chart")
            for n in res.notes:
                st.caption(f"ℹ️ {n}")
            c1, c2, c3 = st.columns(3)
            if c1.button("📌 Pin to dashboard", type="primary", width="stretch", key="st_pin"):
                state.pin_chart(spec)
                st.toast(f"📌 Pinned — {len(st.session_state.dashboard)} chart(s) on your dashboard")
            fig = res.fig
            c2.download_button("⬇️ Interactive HTML", data=lambda: fig.to_html(include_plotlyjs=True, full_html=True),
                               file_name=f"{chart.split()[0].lower()}_chart.html", mime="text/html", width="stretch",
                               key="st_html", on_click="ignore")
            c3.markdown("<div style='padding-top:.45rem;color:#52514e;font-size:.85rem'>🖼️ For a PNG, use the camera icon "
                        "in the chart's toolbar</div>", unsafe_allow_html=True)
            with st.expander("🐍 Python code for this chart"):
                st.code(res.code, language="python")
            if ctx.filtered:
                st.caption("🔎 Built from the filtered view — pinned charts follow the global filters live.")

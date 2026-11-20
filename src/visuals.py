"""
Visualization engine.

- A single Plotly template ("dataviz") built from a validated, colour-blind
  aware categorical palette, a one-hue sequential ramp and a blueâ†”red
  diverging scale with a neutral midpoint.
- ``build_chart(df, spec)``: spec-driven builder used by the Studio and the
  Dashboard. It returns the figure *and* reproducible Python code.
- Helper plots used by the analysis pages.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots

# --------------------------------------------------------------------------
# Palette & template
# --------------------------------------------------------------------------

# Validated reference categorical order (CVD-checked on adjacent pairs).
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = [[0.0, "#e34948"], [0.25, "#f0a3a2"], [0.5, "#f0efec"], [0.75, "#86b6ef"], [1.0, "#1c5cab"]]
MISSING_SCALE = [[0, "#e1e0d9"], [1, "#e34948"]]

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#6b6a65"  # axis/label text: 5.4:1 on white (WCAG AA)
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"

PALETTES = {
    "DataViz (colour-blind aware)": CATEGORICAL,
    "Plotly": px.colors.qualitative.Plotly,
    "D3": px.colors.qualitative.D3,
    "Bold": px.colors.qualitative.Bold,
    "Pastel": px.colors.qualitative.Pastel,
    "Safe": px.colors.qualitative.Safe,
    "Vivid": px.colors.qualitative.Vivid,
}
CONTINUOUS_SCALES = {
    "Blues (sequential)": SEQUENTIAL,
    "Diverging (red â†” blue)": DIVERGING,
    "Viridis": px.colors.sequential.Viridis,
    "Cividis": px.colors.sequential.Cividis,
    "Plasma": px.colors.sequential.Plasma,
    "Teal": px.colors.sequential.Teal,
}

FONT = 'Inter, system-ui, -apple-system, "Segoe UI", sans-serif'

_template = go.layout.Template()
_template.layout = go.Layout(
    colorway=CATEGORICAL,
    font=dict(family=FONT, color=INK_2, size=13),
    title=dict(font=dict(family=FONT, size=17, color=INK), x=0, xanchor="left", y=0.97, yanchor="top"),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    margin=dict(t=64, l=16, r=16, b=40),
    hoverlabel=dict(bgcolor="white", bordercolor=GRID, font=dict(family=FONT, size=13, color=INK)),
    legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=INK_2), title=dict(font=dict(color=MUTED))),
    colorscale=dict(sequential=SEQUENTIAL, sequentialminus=SEQUENTIAL[::-1], diverging=DIVERGING),
    xaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="", title=dict(font=dict(color=MUTED)),
               tickfont=dict(color=MUTED), automargin=True),
    yaxis=dict(gridcolor=GRID, linecolor=AXIS, zeroline=False, ticks="", title=dict(font=dict(color=MUTED)),
               tickfont=dict(color=MUTED), automargin=True),
    bargap=0.35,
    barcornerradius=4,
)
pio.templates["dataviz"] = _template
TEMPLATE = "plotly_white+dataviz"
pio.templates.default = TEMPLATE
px.defaults.template = TEMPLATE
px.defaults.color_discrete_sequence = CATEGORICAL
px.defaults.color_continuous_scale = SEQUENTIAL

MAX_POINTS = 10000
TOP_N_DEFAULT = 6


def sample_data(df: pd.DataFrame, n: int = MAX_POINTS) -> pd.DataFrame:
    return df.sample(n, random_state=42) if len(df) > n else df


def apply_chart_style(fig: go.Figure, title: str = "", height: int | None = None) -> go.Figure:
    fig.update_layout(template=TEMPLATE, height=height)
    if title:
        fig.update_layout(title_text=title)
    fig.update_traces(selector=dict(type="bar"), marker_line_width=0)
    return fig


def _category_orders(df: pd.DataFrame, *cols) -> dict:
    """Stable, sorted category order so a category keeps its colour across charts."""
    out = {}
    for c in cols:
        if c and c in df.columns and not pd.api.types.is_numeric_dtype(df[c]) \
                and not pd.api.types.is_datetime64_any_dtype(df[c]):
            try:
                vals = df[c].dropna().unique()
                if len(vals) <= 200:
                    out[c] = sorted(vals, key=lambda v: str(v))
            except TypeError:
                pass
    return out


# --------------------------------------------------------------------------
# Spec-driven builder
# --------------------------------------------------------------------------

CHART_FAMILIES = {
    "Comparison": ["Bar", "Line", "Area", "Heatmap (pivot)", "Funnel"],
    "Relationship": ["Scatter", "Bubble", "3D Scatter", "Scatter Matrix", "Parallel Coordinates", "Parallel Categories"],
    "Distribution": ["Histogram", "Box", "Violin", "Strip", "ECDF", "Density Heatmap", "Density Contour"],
    "Composition": ["Pie / Donut", "Sunburst", "Treemap", "Icicle"],
    "Geographic": ["Map (points)", "Choropleth (countries)"],
}

# field -> (label, kinds allowed, required)
# kinds: any | num | cat | date | multi_num | multi_cat
CHART_FIELDS: dict[str, dict] = {
    "Bar": {"x": ("Category (X)", "any", True), "y": ("Value (Y) â€” empty = row count", "num", False),
            "color": ("Colour / group", "group", False), "facet_col": ("Facet columns", "group", False)},
    "Line": {"x": ("X axis (usually date)", "any", True), "y": ("Value (Y) â€” empty = row count", "num", False),
             "color": ("Colour / series", "group", False), "facet_col": ("Facet columns", "group", False)},
    "Area": {"x": ("X axis (usually date)", "any", True), "y": ("Value (Y) â€” empty = row count", "num", False),
             "color": ("Stack by", "group", False)},
    "Heatmap (pivot)": {"x": ("Columns (X)", "any", True), "y": ("Rows (Y)", "any", True),
                        "z": ("Value â€” empty = row count", "num", False)},
    "Funnel": {"x": ("Stage", "any", True), "y": ("Value â€” empty = row count", "num", False)},
    "Scatter": {"x": ("X axis", "any", True), "y": ("Y axis", "any", True), "color": ("Colour", "any", False),
                "size": ("Size", "num", False), "symbol": ("Symbol", "group", False),
                "facet_col": ("Facet columns", "group", False), "animation_frame": ("Animate over", "group_or_date", False)},
    "Bubble": {"x": ("X axis", "any", True), "y": ("Y axis", "any", True), "size": ("Bubble size", "num", True),
               "color": ("Colour", "any", False), "animation_frame": ("Animate over", "group_or_date", False)},
    "3D Scatter": {"x": ("X axis", "num", True), "y": ("Y axis", "num", True), "z": ("Z axis", "num", True),
                   "color": ("Colour", "any", False), "size": ("Size", "num", False)},
    "Scatter Matrix": {"dimensions": ("Dimensions (2â€“8)", "multi_num", True), "color": ("Colour", "group", False)},
    "Parallel Coordinates": {"dimensions": ("Dimensions", "multi_num", True), "color": ("Colour (numeric)", "num", False)},
    "Parallel Categories": {"dimensions": ("Dimensions", "multi_cat", True), "color": ("Colour (numeric)", "num", False)},
    "Histogram": {"x": ("Variable", "any", True), "color": ("Colour / group", "group", False),
                  "facet_col": ("Facet columns", "group", False)},
    "Box": {"y": ("Value", "num", True), "x": ("Group by", "group", False), "color": ("Colour", "group", False)},
    "Violin": {"y": ("Value", "num", True), "x": ("Group by", "group", False), "color": ("Colour", "group", False)},
    "Strip": {"y": ("Value", "num", True), "x": ("Group by", "group", False), "color": ("Colour", "group", False)},
    "ECDF": {"x": ("Variable", "num", True), "color": ("Colour / group", "group", False)},
    "Density Heatmap": {"x": ("X axis", "num", True), "y": ("Y axis", "num", True),
                        "z": ("Weight (optional)", "num", False), "facet_col": ("Facet columns", "group", False)},
    "Density Contour": {"x": ("X axis", "num", True), "y": ("Y axis", "num", True), "color": ("Colour", "group", False)},
    "Pie / Donut": {"names": ("Slices", "any", True), "values": ("Value â€” empty = row count", "num", False)},
    "Sunburst": {"path": ("Hierarchy (outer â†’ inner)", "multi_cat", True), "values": ("Size value", "num", False),
                 "color": ("Colour", "any", False)},
    "Treemap": {"path": ("Hierarchy (outer â†’ inner)", "multi_cat", True), "values": ("Size value", "num", False),
                "color": ("Colour", "any", False)},
    "Icicle": {"path": ("Hierarchy (outer â†’ inner)", "multi_cat", True), "values": ("Size value", "num", False),
               "color": ("Colour", "any", False)},
    "Map (points)": {"lat": ("Latitude", "num", True), "lon": ("Longitude", "num", True),
                     "color": ("Colour", "any", False), "size": ("Size", "num", False)},
    "Choropleth (countries)": {"locations": ("Country (ISO-3 code or name)", "any", True),
                               "z": ("Value â€” empty = row count", "num", False),
                               "animation_frame": ("Animate over", "group_or_date", False)},
}

AGG_FUNCS = ["sum", "mean", "median", "count", "min", "max", "std", "nunique"]


@dataclass
class ChartResult:
    fig: go.Figure | None
    code: str = ""
    notes: list[str] = field(default_factory=list)


def _q(v) -> str:
    return repr(v)


def _aggregate(df, keys, value, agg):
    """Group & aggregate; value None â†’ row count. Returns (frame, value column name, code)."""
    keys = [k for k in dict.fromkeys(keys) if k]
    if value is None or agg == "count" and value is None:
        out = df.groupby(keys, observed=True, dropna=False).size().reset_index(name="count")
        code = f"plot_df = df.groupby({_q(keys)}, observed=True).size().reset_index(name='count')"
        return out, "count", code
    if value in keys:
        out = df.groupby(keys, observed=True, dropna=False).size().reset_index(name="count")
        code = f"plot_df = df.groupby({_q(keys)}, observed=True).size().reset_index(name='count')"
        return out, "count", code
    out = df.groupby(keys, observed=True, dropna=False)[value].agg(agg).reset_index()
    code = f"plot_df = df.groupby({_q(keys)}, observed=True)[{_q(value)}].agg({_q(agg)}).reset_index()"
    return out, value, code


def _top_n(df, col, value_col, n, code_lines):
    """Keep the top-n categories of ``col`` (by value), fold the rest into 'Other'."""
    totals = df.groupby(col, observed=True)[value_col].sum().sort_values(ascending=False)
    if len(totals) <= n:
        return df
    keep = set(totals.index[:n])
    df = df.copy()
    df[col] = df[col].where(df[col].isin(keep), "Other")
    df = df.groupby([c for c in df.columns if c != value_col], observed=True, dropna=False)[value_col].sum().reset_index()
    code_lines.append(f"top = plot_df.groupby({_q(col)})[{_q(value_col)}].sum().nlargest({n}).index")
    code_lines.append(f"plot_df[{_q(col)}] = plot_df[{_q(col)}].where(plot_df[{_q(col)}].isin(top), 'Other')")
    return df


def build_chart(df: pd.DataFrame, spec: dict) -> ChartResult:
    """
    Build a chart from a spec dict. Keys used:
    type, x, y, z, color, size, symbol, facet_col, animation_frame, dimensions,
    path, names, values, lat, lon, locations, agg, options{...}, title, height,
    palette, scale.
    """
    t = spec.get("type")
    o = spec.get("options", {}) or {}
    g = lambda k: spec.get(k) or None  # noqa: E731
    x, y, z, color, size = g("x"), g("y"), g("z"), g("color"), g("size")
    facet, anim = g("facet_col"), g("animation_frame")
    agg = spec.get("agg") or "sum"
    palette = PALETTES.get(spec.get("palette"), CATEGORICAL)
    scale = CONTINUOUS_SCALES.get(spec.get("scale"), SEQUENTIAL)
    height = spec.get("height") or 520
    max_points = int(o.get("max_points", MAX_POINTS))
    notes: list[str] = []
    pre: list[str] = []  # data-prep code lines
    kw: dict = {}

    # Validate required fields
    for f, (label, _kind, required) in CHART_FIELDS.get(t, {}).items():
        val = spec.get(f)
        if required and (not val or (isinstance(val, list) and len(val) == 0)):
            return ChartResult(None, notes=[f"Choose **{label}** to draw a {t}."])

    used = [c for c in [x, y, z, color, size, facet, anim, g("symbol"), g("names"), g("values"),
                        g("lat"), g("lon"), g("locations")] if c] + list(spec.get("dimensions") or []) + list(spec.get("path") or [])
    missing = [c for c in used if c not in df.columns]
    if missing:
        return ChartResult(None, notes=[f"Column(s) no longer in the data: {', '.join(map(str, missing))}"])

    data = df
    common = dict(color_discrete_sequence=palette, color_continuous_scale=scale)
    func = None

    if t in ("Bar", "Line", "Area", "Funnel"):
        keys = [x, color if color != x else None, facet]
        data, vcol, code = _aggregate(df, keys, y, agg if y else "count")
        pre.append(code)
        if t in ("Bar", "Funnel") and o.get("top_n"):
            data = _top_n(data, x, vcol, int(o["top_n"]), pre)
        if t == "Bar":
            sort = o.get("sort", "Value â†“")
            if sort != "None" and not pd.api.types.is_datetime64_any_dtype(data[x]):
                if sort.startswith("Value"):
                    order = data.groupby(x, observed=True)[vcol].sum().sort_values(ascending=sort.endswith("â†‘")).index.tolist()
                else:
                    order = sorted(data[x].dropna().unique(), key=str, reverse=sort.endswith("â†“"))
                kw["category_orders"] = {x: order}
            horizontal = o.get("orientation") == "Horizontal"
            kw.update(x=vcol if horizontal else x, y=x if horizontal else vcol, color=color, facet_col=facet,
                      barmode=o.get("barmode", "group"), orientation="h" if horizontal else "v")
            if o.get("text"):
                kw["text_auto"] = ".3s"
            func = "bar"
        elif t == "Funnel":
            data = data.sort_values(vcol, ascending=False)
            pre.append(f"plot_df = plot_df.sort_values({_q(vcol)}, ascending=False)")
            kw.update(x=vcol, y=x)
            func = "funnel"
        else:
            data = data.sort_values(x)
            pre.append(f"plot_df = plot_df.sort_values({_q(x)})")
            kw.update(x=x, y=vcol, color=color)
            if t == "Line":
                kw.update(facet_col=facet, markers=bool(o.get("markers", False)))
                if o.get("rolling", 0) and o["rolling"] > 1:
                    w = int(o["rolling"])
                    grp = [c for c in [color, facet] if c]
                    if grp:
                        data[vcol] = data.groupby(grp, observed=True)[vcol].transform(lambda s: s.rolling(w, min_periods=1).mean())
                        pre.append(f"plot_df[{_q(vcol)}] = plot_df.groupby({_q(grp)})[{_q(vcol)}].transform(lambda s: s.rolling({w}, min_periods=1).mean())")
                    else:
                        data[vcol] = data[vcol].rolling(w, min_periods=1).mean()
                        pre.append(f"plot_df[{_q(vcol)}] = plot_df[{_q(vcol)}].rolling({w}, min_periods=1).mean()")
                    notes.append(f"Smoothed with a {w}-point rolling mean.")
                func = "line"
            else:
                func = "area"
        if y:
            kw["labels"] = {vcol: f"{agg}({y})"}
        kw["category_orders"] = {**_category_orders(data, color, facet), **kw.get("category_orders", {})}

    elif t == "Heatmap (pivot)":
        piv_val = z
        if piv_val:
            pv = df.pivot_table(index=y, columns=x, values=piv_val, aggfunc=agg, observed=True)
            pre.append(f"plot_df = df.pivot_table(index={_q(y)}, columns={_q(x)}, values={_q(piv_val)}, aggfunc={_q(agg)}, observed=True)")
        else:
            pv = pd.crosstab(df[y], df[x])
            pre.append(f"plot_df = pd.crosstab(df[{_q(y)}], df[{_q(x)}])")
        if pv.shape[0] > 60 or pv.shape[1] > 60:
            pv = pv.iloc[:60, :60]
            notes.append("Showing the first 60 Ã— 60 cells.")
        fig = px.imshow(pv, text_auto=".3s" if pv.size <= 400 else False, aspect="auto",
                        color_continuous_scale=scale, labels=dict(color=f"{agg}({z})" if z else "count"))
        code = _render_code("imshow", pre, "plot_df", {"text_auto": ".3s", "aspect": "auto"})
        return _finish(fig, spec, code, notes, height)

    elif t in ("Scatter", "Bubble"):
        data = sample_data(df, max_points)
        if len(df) > max_points:
            notes.append(f"Showing a random sample of {max_points:,} of {len(df):,} rows.")
            pre.append(f"plot_df = df.sample({max_points}, random_state=42)")
        if size:
            data = data[data[size].notna()]
            if (data[size] < 0).any():
                data = data.assign(**{size: data[size] - data[size].min()})
                notes.append("Size values shifted to be non-negative.")
        kw.update(x=x, y=y, color=color, size=size, symbol=g("symbol"), facet_col=facet,
                  log_x=bool(o.get("log_x")), log_y=bool(o.get("log_y")), opacity=o.get("opacity", 0.8))
        if anim:
            data = data.sort_values(anim)
            kw.update(animation_frame=anim, animation_group=g("animation_group"))
            if pd.api.types.is_numeric_dtype(data[x]):
                kw["range_x"] = [data[x].min(), data[x].max()]
            if pd.api.types.is_numeric_dtype(data[y]):
                kw["range_y"] = [data[y].min(), data[y].max()]
        if t == "Bubble":
            kw["size_max"] = 50
        trend = o.get("trendline", "None")
        if trend != "None" and pd.api.types.is_numeric_dtype(data[x]) and pd.api.types.is_numeric_dtype(data[y]):
            try:
                import statsmodels  # noqa: F401
                kw["trendline"] = {"OLS (linear)": "ols", "LOWESS (smooth)": "lowess"}.get(trend)
                if o.get("trend_scope") == "Overall":
                    kw["trendline_scope"] = "overall"
                    kw["trendline_color_override"] = INK
            except ImportError:
                notes.append("Install `statsmodels` to enable trendlines.")
        for m in ("marginal_x", "marginal_y"):
            if o.get(m) and o[m] != "None" and not anim and not facet:
                kw[m] = o[m]
        kw["category_orders"] = _category_orders(data, color, facet, g("symbol"))
        func = "scatter"

    elif t == "3D Scatter":
        data = sample_data(df, max_points)
        kw.update(x=x, y=y, z=z, color=color, size=size, opacity=o.get("opacity", 0.8))
        kw["category_orders"] = _category_orders(data, color)
        func = "scatter_3d"

    elif t == "Scatter Matrix":
        dims = list(spec["dimensions"])[:8]
        data = sample_data(df, min(max_points, 5000))
        kw.update(dimensions=dims, color=color, opacity=0.6)
        kw["category_orders"] = _category_orders(data, color)
        func = "scatter_matrix"

    elif t == "Parallel Coordinates":
        data = sample_data(df, min(max_points, 5000))
        kw.update(dimensions=list(spec["dimensions"]), color=color)
        func = "parallel_coordinates"

    elif t == "Parallel Categories":
        data = sample_data(df, min(max_points, 20000))
        kw.update(dimensions=list(spec["dimensions"]), color=color)
        func = "parallel_categories"

    elif t == "Histogram":
        kw.update(x=x, color=color, facet_col=facet, nbins=int(o.get("bins", 40)) or None,
                  barmode=o.get("hist_barmode", "overlay"), opacity=0.75 if color else 0.9,
                  histnorm=None if o.get("histnorm", "count") == "count" else o["histnorm"],
                  cumulative=bool(o.get("cumulative")), log_y=bool(o.get("log_y")))
        if o.get("marginal") and o["marginal"] != "None" and not facet:
            kw["marginal"] = o["marginal"]
        kw["category_orders"] = _category_orders(df, color, facet)
        func = "histogram"

    elif t in ("Box", "Violin", "Strip"):
        point_based = t == "Strip" or o.get("points") == "all"
        data = sample_data(df, max_points if point_based else 200000)
        kw.update(x=x, y=y, color=color)
        if t == "Box":
            kw.update(points=None if o.get("points") == "none" else o.get("points", "outliers"),
                      notched=bool(o.get("notched")))
            if kw["points"] is None:
                kw["points"] = False
            func = "box"
        elif t == "Violin":
            kw.update(box=True, points=False if o.get("points") == "none" else o.get("points", "outliers"))
            func = "violin"
        else:
            func = "strip"
        kw["log_y"] = bool(o.get("log_y"))
        kw["category_orders"] = _category_orders(data, x, color)

    elif t == "ECDF":
        kw.update(x=x, color=color)
        kw["category_orders"] = _category_orders(df, color)
        func = "ecdf"

    elif t == "Density Heatmap":
        data = sample_data(df, 200000)
        kw.update(x=x, y=y, z=z, facet_col=facet, histfunc="sum" if z else "count",
                  nbinsx=int(o.get("bins", 40)), nbinsy=int(o.get("bins", 40)))
        func = "density_heatmap"

    elif t == "Density Contour":
        data = sample_data(df, max_points)
        kw.update(x=x, y=y, color=color)
        kw["category_orders"] = _category_orders(data, color)
        func = "density_contour"

    elif t == "Pie / Donut":
        names = spec["names"]
        data, vcol, code = _aggregate(df, [names], g("values"), agg if g("values") else "count")
        pre.append(code)
        data = _top_n(data, names, vcol, int(o.get("top_n") or TOP_N_DEFAULT), pre)
        kw.update(names=names, values=vcol, hole=0.55 if o.get("donut", True) else 0)
        kw["category_orders"] = _category_orders(data, names)
        func = "pie"

    elif t in ("Sunburst", "Treemap", "Icicle"):
        path = list(spec["path"])
        values = g("values")
        data = df.dropna(subset=path).copy()
        for p in path:
            data[p] = data[p].astype(str)
        if values:
            data = data[data[values] > 0]
        pre.append(f"plot_df = df.dropna(subset={_q(path)}).astype({{c: str for c in {_q(path)}}})")
        kw.update(path=path, values=values, color=color)
        if color and not pd.api.types.is_numeric_dtype(df[color]):
            kw.pop("color_continuous_scale", None)
        func = t.lower()

    elif t == "Map (points)":
        data = sample_data(df, max_points)
        kw.update(lat=g("lat"), lon=g("lon"), color=color, size=size, projection="natural earth")
        func = "scatter_geo"

    elif t == "Choropleth (countries)":
        loc = g("locations")
        keys = [loc, anim]
        data, vcol, code = _aggregate(df, keys, z, agg if z else "count")
        pre.append(code)
        sample_loc = data[loc].dropna().astype(str)
        mode = "ISO-3" if len(sample_loc) and (sample_loc.str.fullmatch(r"[A-Z]{3}").mean() > 0.9) else "country names"
        kw.update(locations=loc, color=vcol, locationmode=mode, projection="natural earth")
        if anim:
            data = data.sort_values(anim)
            kw["animation_frame"] = anim
            kw["range_color"] = [data[vcol].min(), data[vcol].max()]
        func = "choropleth"

    else:
        return ChartResult(None, notes=[f"Unknown chart type: {t}"])

    kw = {k: v for k, v in kw.items() if v is not None and v is not False or k in ("points",)}
    if "points" in kw and kw["points"] is None:
        kw.pop("points")
    fn = getattr(px, func)
    import inspect
    accepted = inspect.signature(fn).parameters
    call_kw = {**{k: v for k, v in common.items() if k in accepted}, **kw}
    if t in ("Sunburst", "Treemap", "Icicle") and color and not pd.api.types.is_numeric_dtype(df[color]):
        call_kw.pop("color_continuous_scale", None)
    fig = fn(data, **call_kw)

    if t in ("Bar", "Histogram"):
        fig.update_traces(marker_line_width=0, selector=dict(type="bar"))
    if t in ("Scatter", "Bubble"):
        fig.update_traces(marker=dict(line=dict(width=1, color="white")), selector=dict(mode="markers"))
    if t == "Line":
        fig.update_traces(line=dict(width=2))
    if t == "Pie / Donut":
        fig.update_traces(textposition="outside", textinfo="label+percent",
                          marker=dict(line=dict(color="white", width=2)))
    if t == "Density Contour":
        fig.update_traces(contours_coloring="lines", line_width=1.5)
    if t == "Parallel Categories" and not color:
        fig.update_traces(line=dict(color=CATEGORICAL[0]))

    data_var = "plot_df" if pre else "df"
    code_kw = {k: v for k, v in kw.items() if k not in ("category_orders", "range_x", "range_y", "range_color")}
    code = _render_code(func, pre, data_var, code_kw)
    return _finish(fig, spec, code, notes, height)


def _render_code(func: str, pre: list[str], data_var: str, kw: dict) -> str:
    args = ", ".join(f"{k}={_q(v)}" for k, v in kw.items())
    lines = ["import pandas as pd", "import plotly.express as px", "",
             "df = pd.read_csv('your_data.csv')  # the dataset you are analysing", ""]
    lines += pre
    lines.append(f"fig = px.{func}({data_var}{', ' + args if args else ''})")
    lines.append("fig.show()")
    return "\n".join(lines)


def _finish(fig, spec, code, notes, height) -> ChartResult:
    title = spec.get("title") or default_title(spec)
    fig.update_layout(title_text=f"<b>{title}</b>", height=height)
    o = spec.get("options", {}) or {}
    if o.get("x_title"):
        fig.update_xaxes(title_text=o["x_title"])
    if o.get("y_title"):
        fig.update_yaxes(title_text=o["y_title"])
    legend = o.get("legend", "Right")
    if spec.get("color") and spec.get("color") in (spec.get("x"), spec.get("names")) and legend != "Hidden":
        legend = "Hidden"  # the axis already labels each colour — a legend would only repeat it
    if legend == "Top":
        fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0))
    elif legend == "Bottom":
        fig.update_layout(legend=dict(orientation="h", yanchor="top", y=-0.15, xanchor="left", x=0))
    elif legend == "Hidden":
        fig.update_layout(showlegend=False)
    if not o.get("grid", True):
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
    fig.for_each_annotation(lambda a: a.update(text=a.text.split("=")[-1]))  # cleaner facet labels
    return ChartResult(fig, code, notes)


def default_title(spec: dict) -> str:
    t = spec.get("type", "")
    x, y, z, c = spec.get("x"), spec.get("y"), spec.get("z"), spec.get("color")
    agg = spec.get("agg") or "sum"
    if t in ("Bar", "Line", "Area", "Funnel"):
        base = f"{agg.title()} of {y}" if y else "Row count"
        return f"{base} by {x}" + (f" and {c}" if c and c != x else "")
    if t == "Heatmap (pivot)":
        return f"{agg.title()} of {z} by {y} Ã— {x}" if z else f"Count by {y} Ã— {x}"
    if t in ("Scatter", "Bubble"):
        return f"{y} vs {x}"
    if t == "3D Scatter":
        return f"{x}, {y} and {z}"
    if t in ("Histogram", "ECDF"):
        return f"Distribution of {x}" + (f" by {c}" if c else "")
    if t in ("Box", "Violin", "Strip"):
        return f"{y}" + (f" by {x}" if x else " distribution")
    if t in ("Sunburst", "Treemap", "Icicle"):
        return " â†’ ".join(spec.get("path") or [])
    if t == "Pie / Donut":
        return f"Share of {spec.get('values') or 'rows'} by {spec.get('names')}"
    if t in ("Scatter Matrix", "Parallel Coordinates", "Parallel Categories"):
        return t
    if t == "Density Heatmap" or t == "Density Contour":
        return f"Density of {y} vs {x}"
    if t == "Choropleth (countries)":
        return f"{agg.title()} of {z} by country" if z else "Rows by country"
    return t


# --------------------------------------------------------------------------
# Analysis helper plots
# --------------------------------------------------------------------------

def plot_histogram(df, x, color=None, bins=40):
    fig = px.histogram(df, x=x, color=color, nbins=bins, marginal="box", opacity=0.9,
                       category_orders=_category_orders(df, color))
    fig.update_traces(marker_line_width=0, selector=dict(type="histogram"))
    return apply_chart_style(fig, title=f"<b>Distribution of {x}</b>", height=420)


def plot_qq(osm, osr, slope, intercept, name):
    fig = go.Figure()
    fig.add_trace(go.Scattergl(x=osm, y=osr, mode="markers", name="Observed",
                               marker=dict(size=6, color=CATEGORICAL[0], opacity=0.6)))
    line_x = np.array([np.min(osm), np.max(osm)])
    fig.add_trace(go.Scatter(x=line_x, y=slope * line_x + intercept, mode="lines", name="Normal reference",
                             line=dict(color=CATEGORICAL[1], width=2)))
    fig.update_layout(xaxis_title="Theoretical quantiles", yaxis_title=f"Observed {name}",
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    return apply_chart_style(fig, title=f"<b>Q-Q plot: {name}</b>", height=420)


def plot_value_counts(vc: pd.DataFrame, name: str):
    d = vc.iloc[::-1]
    fig = px.bar(d, x="Count", y="Value", orientation="h", text="Share %",
                 color_discrete_sequence=[CATEGORICAL[0]])
    fig.update_traces(texttemplate="%{text:.1f}%", textposition="outside", cliponaxis=False)
    fig.update_layout(yaxis_title=None, height=max(260, 28 * len(d) + 100))
    return apply_chart_style(fig, title=f"<b>Top values: {name}</b>")


def plot_heatmap(matrix: pd.DataFrame, title="Correlation matrix", zmin=-1, zmax=1, diverging=True):
    n = len(matrix)
    fig = px.imshow(matrix, text_auto=".2f" if n <= 15 else False, aspect="auto",
                    color_continuous_scale=DIVERGING if diverging else SEQUENTIAL, zmin=zmin, zmax=zmax)
    fig.update_layout(height=max(420, min(900, 34 * n + 160)))
    fig.update_xaxes(side="bottom", tickangle=-40)
    return apply_chart_style(fig, title=f"<b>{title}</b>")


def plot_missing_matrix(df: pd.DataFrame, max_rows: int = 1500):
    miss = df.isna()
    if len(miss) > max_rows:
        miss = miss.iloc[np.linspace(0, len(miss) - 1, max_rows).astype(int)]
    fig = px.imshow(miss.astype(int).T, aspect="auto", color_continuous_scale=MISSING_SCALE, zmin=0, zmax=1,
                    labels=dict(x="Row (in file order)", y="Column", color="Missing"))
    fig.update_coloraxes(showscale=False)
    fig.update_layout(height=max(320, min(800, 22 * df.shape[1] + 140)))
    return apply_chart_style(fig, title="<b>Missing-value map</b> (red = missing)")


def plot_missing_bars(df: pd.DataFrame):
    m = (df.isna().mean() * 100).sort_values()
    m = m[m > 0]
    if m.empty:
        return None
    fig = px.bar(x=m.values, y=m.index, orientation="h", labels={"x": "Missing %", "y": ""},
                 text=[f"{v:.1f}%" for v in m.values], color_discrete_sequence=[CATEGORICAL[7]])
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(height=max(260, 26 * len(m) + 110), xaxis_range=[0, min(100, m.max() * 1.25 + 1)])
    return apply_chart_style(fig, title="<b>Missing values by column</b>")


def plot_small_multiples(df: pd.DataFrame, columns: list[str], cols: int = 4, bins: int = 30):
    """Grid of mini histograms â€” a one-glance view of every numeric column."""
    columns = columns[:24]
    rows = int(np.ceil(len(columns) / cols))
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[str(c) for c in columns],
                        horizontal_spacing=0.06, vertical_spacing=0.12 if rows < 4 else 0.06)
    for i, c in enumerate(columns):
        s = pd.to_numeric(df[c], errors="coerce").dropna()
        s = s.sample(50000, random_state=1) if len(s) > 50000 else s
        fig.add_trace(go.Histogram(x=s, nbinsx=bins, marker_color=CATEGORICAL[0], marker_line_width=0, name=str(c),
                                   showlegend=False, hovertemplate="%{x}<br>%{y} rows<extra></extra>"),
                      row=i // cols + 1, col=i % cols + 1)
    fig.update_annotations(font=dict(size=12, color=INK_2))
    fig.update_layout(height=max(260, 200 * rows), bargap=0.05)
    fig.update_yaxes(showticklabels=False)
    return apply_chart_style(fig)


def plot_score_gauge(score: int):
    color = "#0ca30c" if score >= 95 else "#fab219" if score >= 75 else "#d03b3b"
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=score, number=dict(suffix="/100", font=dict(size=30, color=INK)),
        gauge=dict(axis=dict(range=[0, 100], tickcolor=MUTED, tickwidth=1), bar=dict(color=color, thickness=0.35),
                   bgcolor=GRID, borderwidth=0)))
    fig.update_layout(height=230, margin=dict(t=24, b=8, l=52, r=52))
    return fig


def plot_time_series(ts: pd.DataFrame, date_col: str, value_col: str, group_col=None, rolling: int = 0):
    fig = px.line(ts, x=date_col, y=value_col, color=group_col,
                  category_orders=_category_orders(ts, group_col))
    fig.update_traces(line=dict(width=2))
    if rolling and rolling > 1 and not group_col:
        r = ts[value_col].rolling(rolling, min_periods=1, center=True).mean()
        fig.add_trace(go.Scatter(x=ts[date_col], y=r, mode="lines", name=f"{rolling}-period rolling mean",
                                 line=dict(color=CATEGORICAL[1], width=2)))
        fig.update_traces(selector=dict(name=value_col), line=dict(color=CATEGORICAL[0], width=1.5), opacity=0.5)
    fig.update_layout(hovermode="x unified", legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    return apply_chart_style(fig, title=f"<b>{value_col} over time</b>", height=440)


def plot_decomposition(result, name: str):
    parts = [("Observed", result.observed), ("Trend", result.trend), ("Seasonal", result.seasonal), ("Residual", result.resid)]
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, subplot_titles=[p[0] for p in parts], vertical_spacing=0.06)
    for i, (label, s) in enumerate(parts, 1):
        mode = "markers" if label == "Residual" else "lines"
        fig.add_trace(go.Scatter(x=s.index, y=s.values, mode=mode, name=label, showlegend=False,
                                 line=dict(color=CATEGORICAL[0] if i != 2 else CATEGORICAL[1], width=2),
                                 marker=dict(size=4, color=CATEGORICAL[0])), row=i, col=1)
    fig.update_layout(height=720)
    return apply_chart_style(fig, title=f"<b>Seasonal decomposition: {name}</b>")


def plot_seasonality_heatmap(df: pd.DataFrame, date_col: str, value_col: str | None, agg="mean"):
    d = df[[date_col] + ([value_col] if value_col else [])].dropna(subset=[date_col]).copy()
    d["Weekday"] = d[date_col].dt.day_name().str[:3]
    d["Month"] = d[date_col].dt.month_name().str[:3]
    if value_col:
        pv = d.pivot_table(index="Weekday", columns="Month", values=value_col, aggfunc=agg)
    else:
        pv = pd.crosstab(d["Weekday"], d["Month"])
    months = [m for m in ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"] if m in pv.columns]
    days = [w for w in ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"] if w in pv.index]
    pv = pv.reindex(index=days, columns=months)
    fig = px.imshow(pv, aspect="auto", color_continuous_scale=SEQUENTIAL, text_auto=".3s")
    return apply_chart_style(fig, title=f"<b>Seasonality: {agg} of {value_col or 'rows'} by weekday Ã— month</b>", height=380)


def plot_elbow(diag: pd.DataFrame):
    fig = make_subplots(rows=1, cols=2, subplot_titles=["Elbow (inertia â€” lower is tighter)", "Silhouette (higher is better)"])
    fig.add_trace(go.Scatter(x=diag["k"], y=diag["Inertia"], mode="lines+markers", line=dict(color=CATEGORICAL[0], width=2),
                             marker=dict(size=8), showlegend=False), row=1, col=1)
    fig.add_trace(go.Scatter(x=diag["k"], y=diag["Silhouette"], mode="lines+markers", line=dict(color=CATEGORICAL[1], width=2),
                             marker=dict(size=8), showlegend=False), row=1, col=2)
    best = diag.loc[diag["Silhouette"].idxmax()]
    fig.add_annotation(x=best["k"], y=best["Silhouette"], text=f"best k = {int(best['k'])}", showarrow=True,
                       arrowhead=0, ay=-30, row=1, col=2, font=dict(color=INK))
    fig.update_xaxes(title_text="k (clusters)", dtick=1)
    fig.update_layout(height=340)
    return apply_chart_style(fig)


def plot_scree(explained: np.ndarray):
    k = len(explained)
    labels = [f"PC{i + 1}" for i in range(k)]
    fig = go.Figure()
    fig.add_trace(go.Bar(x=labels, y=explained * 100, name="Individual", marker_color=CATEGORICAL[0], marker_line_width=0))
    fig.add_trace(go.Scatter(x=labels, y=np.cumsum(explained) * 100, name="Cumulative", mode="lines+markers",
                             line=dict(color=CATEGORICAL[1], width=2), marker=dict(size=8)))
    fig.update_layout(yaxis_title="Variance explained (%)", yaxis_range=[0, 105],
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0))
    return apply_chart_style(fig, title="<b>Scree plot</b>", height=360)


def plot_feature_ranking(ranking: pd.DataFrame, target: str):
    d = ranking.head(25).iloc[::-1]
    fig = px.bar(d, x="Strength", y="Feature", orientation="h", color="Method",
                 hover_data={"Mutual info": ":.3f", "Signed r": ":.3f"},
                 category_orders=_category_orders(d, "Method"))
    fig.update_layout(height=max(320, 26 * len(d) + 120), yaxis_title=None, xaxis_range=[0, 1],
                      legend=dict(orientation="h", y=1.02, yanchor="bottom", x=0, title=None))
    return apply_chart_style(fig, title=f"<b>What drives {target}?</b> (association strength, 0â€“1)")


# Back-compat thin wrappers (older imports)
def plot_scatter(df, x, y, color=None, size=None, trendline=False, animation_frame=None, facet_col=None, log_x=False, log_y=False):
    spec = dict(type="Scatter", x=x, y=y, color=color, size=size, animation_frame=animation_frame, facet_col=facet_col,
                options=dict(log_x=log_x, log_y=log_y, trendline="OLS (linear)" if trendline else "None"))
    return build_chart(df, spec).fig


def plot_box(df, x, y, color=None):
    return build_chart(df, dict(type="Box", x=x, y=y, color=color)).fig


def plot_violin(df, x, y, color=None):
    return build_chart(df, dict(type="Violin", x=x, y=y, color=color)).fig


def plot_bar(df, x, y, color=None):
    return build_chart(df, dict(type="Bar", x=x, y=y, color=color)).fig


def plot_line(df, x, y, color=None):
    return build_chart(df, dict(type="Line", x=x, y=y, color=color)).fig


def plot_3d_scatter(df, x, y, z, color=None):
    return build_chart(df, dict(type="3D Scatter", x=x, y=y, z=z, color=color)).fig

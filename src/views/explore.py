"""Explore: the deep-dive analysis workbench."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from src import eda, state, transforms as T
from src.ui_components import (
    fmt_num, insight_cards, page_header, result_card, section, show_chart, show_df, tiles,
)
from src.visuals import (
    CATEGORICAL, DIVERGING, apply_chart_style, build_chart, plot_decomposition, plot_elbow, plot_feature_ranking,
    plot_heatmap, plot_histogram, plot_missing_bars, plot_missing_matrix, plot_qq, plot_scree, plot_seasonality_heatmap,
    plot_time_series, plot_value_counts,
)

NONE = "— none —"
fmt_none = lambda v: NONE if v is None else str(v)  # noqa: E731


def render(ctx):
    page_header("🧭", "Explore", "Statistical deep dive — quality, distributions, relationships, drivers, time and structure")
    tabs = st.tabs(["🏥 Quality", "📊 Distributions", "🔗 Correlations", "🎯 Drivers", "🚩 Outliers",
                    "📈 Time Series", "🧬 PCA", "🧩 Clustering"])
    fns = [_quality, _distributions, _correlations, _drivers, _outliers, _time_series, _pca, _clustering]
    for tab, fn in zip(tabs, fns):
        with tab:
            fn(ctx)


# --------------------------------------------------------------------------
def _quality(ctx):
    df = ctx.df
    if not df.isna().any().any():
        result_card("good", "✓ No missing values", "Every cell in the dataset is populated.")
    else:
        c1, c2 = st.columns([1.4, 1])
        with c1:
            show_chart(plot_missing_matrix(df), key="ex_missmap")
        with c2:
            show_chart(plot_missing_bars(df), key="ex_missbars")
    miss = df.isna()
    if miss.any().sum() >= 2:
        section("Missingness correlation", "Do blanks in one column coincide with blanks in another?")
        mc = miss.loc[:, miss.any()].astype(int).corr()
        show_chart(plot_heatmap(mc, "Missingness correlation"), key="ex_misscorr")
    section("Diagnostics")
    insights = eda.smart_insights(df)
    insight_cards(insights) if insights else st.success("No issues detected.", icon="✅")


# --------------------------------------------------------------------------
def _distributions(ctx):
    df, types = ctx.df, ctx.types
    mode = st.segmented_control("Variable type", ["Numeric", "Categorical"], default="Numeric", key="p_ex_dist_mode") or "Numeric"
    if mode == "Numeric":
        nums = types["numerical"]
        if not nums:
            st.info("No numeric columns.")
            return
        col = state.select("Variable", nums, key="p_ex_dist_col")
        grp = state.select("Compare across groups (optional)", [None] + [g for g in types["groupable"] if g != col],
                           key="p_ex_dist_grp", format_func=fmt_none)
        c1, c2 = st.columns(2)
        with c1:
            show_chart(plot_histogram(df, col, color=grp), key="ex_hist")
        with c2:
            osm, osr, slope, intercept = eda.qq_data(df[col])
            show_chart(plot_qq(osm, osr, slope, intercept, col), key="ex_qq")
        if grp:
            show_chart(build_chart(df, {"type": "Violin", "y": col, "x": grp, "color": grp,
                                        "options": {"points": "none"}, "height": 420}).fig, key="ex_violin")
        section("Normality tests", "p > 0.05 → no evidence against normality")
        nt = eda.normality_tests(df[col])
        if nt.empty:
            st.info("Not enough distinct values to test normality.")
        else:
            show_df(nt.style.format({"Statistic": "{:.4f}", "p-value": "{:.4g}"}), hide_index=True)
            v = pd.to_numeric(df[col], errors="coerce").dropna()
            sk = v.skew()
            normal_votes = (nt["p-value"] > 0.05).sum()
            if normal_votes >= len(nt) / 2:
                result_card("good", "✓ Approximately normal", "Most tests are consistent with a normal distribution — "
                            "parametric methods (t-tests, ANOVA, Pearson) are appropriate.")
            else:
                hint = "a log or square-root transform" if sk > 1 else "a power transform" if abs(sk) > 1 else "non-parametric tests"
                result_card("warning", "▲ Not normally distributed",
                            f"Skew = {sk:.2f}. Prefer non-parametric tests (Mann-Whitney, Kruskal-Wallis, Spearman) or try {hint}.")
    else:
        cats = types["categorical"] + types["boolean"] + types["text"]
        if not cats:
            st.info("No categorical columns.")
            return
        col = state.select("Variable", cats, key="p_ex_cat_col")
        top = st.slider("Show top N values", 5, 50, 15, key="p_ex_cat_top")
        vc = eda.value_counts_table(df[col], top)
        c1, c2 = st.columns([1.6, 1])
        with c1:
            show_chart(plot_value_counts(vc, col), key="ex_vc")
        with c2:
            show_df(vc, hide_index=True, height=480,
                    column_config={"Share %": st.column_config.ProgressColumn(min_value=0, max_value=100, format="%.1f%%")})


# --------------------------------------------------------------------------
def _correlations(ctx):
    df, types = ctx.df, ctx.types
    nums = types["numerical"]
    sub = st.segmented_control("View", ["Numeric correlation", "Categorical association", "Scatter matrix"],
                               default="Numeric correlation", key="p_ex_corr_view") or "Numeric correlation"
    if sub == "Numeric correlation":
        if len(nums) < 2:
            st.info("Need at least 2 numeric columns.")
            return
        c1, c2 = st.columns([1, 2])
        method = c1.selectbox("Method", ["pearson", "spearman", "kendall"], key="p_ex_corr_m",
                              format_func=lambda m: {"pearson": "Pearson (linear)", "spearman": "Spearman (rank / monotonic)",
                                                     "kendall": "Kendall τ (rank, robust)"}[m])
        with c2:
            cols = state.multiselect("Columns", nums, key="p_ex_corr_cols", default=nums[:20])
        if len(cols) < 2:
            st.info("Select at least 2 columns.")
            return
        corr = eda.correlation_matrix(df, cols, method)
        if st.toggle("Cluster similar variables together", True, key="p_ex_corr_cluster") and len(cols) > 2:
            from scipy.cluster.hierarchy import leaves_list, linkage
            from scipy.spatial.distance import squareform
            dist = 1 - corr.abs().fillna(0).values
            np.fill_diagonal(dist, 0)
            order = leaves_list(linkage(squareform(np.clip(dist, 0, None), checks=False), "average"))
            corr = corr.iloc[order, order]
        show_chart(plot_heatmap(corr, f"{method.title()} correlation"), key="ex_corr")
        section("Strongest relationships")
        pairs = eda.top_correlation_pairs(corr, 15)
        show_df(pairs, hide_index=True, column_config={
            "r": st.column_config.NumberColumn(format="%.3f"),
            "|r|": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")})
        if not pairs.empty:
            best = pairs.iloc[0]
            if st.button(f"🔍 Inspect '{best['Feature A']}' vs '{best['Feature B']}'", key="ex_inspect"):
                show_chart(build_chart(df, {"type": "Scatter", "x": best["Feature A"], "y": best["Feature B"],
                                            "options": {"trendline": "OLS (linear)", "opacity": 0.6}}).fig, key="ex_inspect_fig")
    elif sub == "Categorical association":
        cats = [c for c in types["categorical"] + types["boolean"] if df[c].nunique() <= 100]
        if len(cats) < 2:
            st.info("Need at least 2 categorical columns (≤ 100 levels).")
            return
        cols = state.multiselect("Columns", cats, key="p_ex_cv_cols", default=cats[:12])
        if len(cols) >= 2:
            m = eda.cramers_v_matrix(df, cols)
            show_chart(plot_heatmap(m, "Cramér's V (0 = independent, 1 = perfectly associated)", zmin=0, zmax=1, diverging=False),
                       key="ex_cv")
    else:
        if len(nums) < 2:
            st.info("Need at least 2 numeric columns.")
            return
        dims = state.multiselect("Dimensions (2–6)", nums, key="p_ex_sm_dims", default=nums[:4], max_selections=6)
        color = state.select("Colour by", [None] + types["groupable"], key="p_ex_sm_col", format_func=fmt_none)
        if len(dims) >= 2:
            res = build_chart(df, {"type": "Scatter Matrix", "dimensions": dims, "color": color, "height": 160 * len(dims) + 120})
            show_chart(res.fig, key="ex_sm")


# --------------------------------------------------------------------------
def _drivers(ctx):
    df, types = ctx.df, ctx.types
    candidates = types["numerical"] + types["categorical"] + types["boolean"]
    if len(candidates) < 2:
        st.info("Need at least 2 usable columns.")
        return
    target = state.select("🎯 Target variable — what do you want to explain?", candidates, key="p_ex_target")
    ranking = eda.target_analysis(df, target)
    if ranking.empty:
        st.info("No features to rank.")
        return
    top = ranking.iloc[0]
    strong = ranking[ranking["Strength"] >= 0.3]
    tiles([
        {"label": "Top driver", "value": str(top["Feature"])[:22], "sub": f"{top['Method']} = {top['Strength']:.2f}",
         "status": "good" if top["Strength"] >= 0.5 else "warn" if top["Strength"] >= 0.3 else "neutral"},
        {"label": "Moderate+ drivers", "value": len(strong), "sub": "strength ≥ 0.30", "status": "neutral"},
        {"label": "Features ranked", "value": len(ranking), "status": "neutral"},
    ])
    c1, c2 = st.columns([1.5, 1])
    with c1:
        show_chart(plot_feature_ranking(ranking, target), key="ex_rank")
    with c2:
        show_df(ranking, hide_index=True, height=520, column_config={
            "Strength": st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.3f"),
            "Signed r": st.column_config.NumberColumn(format="%.3f"),
            "Mutual info": st.column_config.NumberColumn(format="%.3f")})
    st.caption("Strength scale: < 0.1 negligible · 0.1–0.3 weak · 0.3–0.5 moderate · > 0.5 strong. "
               "Mutual information also captures non-linear relationships.")
    section("Relationship with the top driver")
    f = top["Feature"]
    t_num = types["kinds"][target] == "numerical"
    f_num = types["kinds"][f] == "numerical"
    if t_num and f_num:
        spec = {"type": "Scatter", "x": f, "y": target, "options": {"trendline": "OLS (linear)", "opacity": 0.6}}
    elif t_num:
        spec = {"type": "Box", "x": f, "y": target, "color": f}
    elif f_num:
        spec = {"type": "Box", "x": target, "y": f, "color": target}
    else:
        spec = {"type": "Bar", "x": f, "color": target, "options": {"barmode": "stack"}}
    show_chart(build_chart(df, spec).fig, key="ex_driver_fig")


# --------------------------------------------------------------------------
def _outliers(ctx):
    df, types = ctx.df, ctx.types
    nums = types["numerical"]
    if not nums:
        st.info("No numeric columns.")
        return
    c1, c2, c3 = st.columns(3)
    method = c1.selectbox("Method", ["IQR", "Z-score", "Modified Z (MAD)", "Isolation Forest (multivariate)"], key="p_ex_out_m")
    if method.startswith("Isolation"):
        cont = c2.slider("Expected anomaly share", 0.005, 0.2, 0.02, 0.005, key="p_ex_out_c")
        cols = state.multiselect("Features", nums, key="p_ex_out_cols", default=nums[:8])
        if len(cols) < 1:
            return
        flags = eda.isolation_forest(df, cols, cont)
        n = int(flags.sum())
        tiles([{"label": "Anomalies", "value": f"{n:,}", "sub": f"{n / len(df):.1%} of rows", "status": "warn" if n else "good"}])
        if len(cols) >= 2:
            plot_df = df.assign(_status=np.where(flags, "Anomaly", "Normal"))
            fig = px.scatter(plot_df.sample(min(len(plot_df), 8000), random_state=1), x=cols[0], y=cols[1], color="_status",
                             color_discrete_map={"Normal": "#b9c9de", "Anomaly": "#d03b3b"}, opacity=0.75,
                             labels={"_status": "Status"})
            show_chart(apply_chart_style(fig, f"<b>Isolation Forest anomalies</b> ({cols[0]} vs {cols[1]})", 460), key="ex_iso")
        show_df(df[flags].head(500), height=320)
        return
    default = {"IQR": 1.5, "Z-score": 3.0, "Modified Z (MAD)": 3.5}[method]
    thr = c2.slider("Threshold", 0.5, 6.0, default, 0.1, key=f"p_ex_out_t_{method}")
    summary = eda.outlier_summary(df, nums, method, thr)
    with c3:
        st.metric("Columns with outliers", f"{(summary['Outliers'] > 0).sum()} / {len(nums)}")
    c1, c2 = st.columns([1, 1.4])
    with c1:
        show_df(summary, hide_index=True, height=440, column_config={
            "Outlier %": st.column_config.ProgressColumn(min_value=0, max_value=max(5.0, float(summary["Outlier %"].max())), format="%.2f%%")})
    with c2:
        col = state.select("Inspect column", summary["Column"].tolist(), key="p_ex_out_col")
        mask = eda.outlier_mask(df[col], method, thr).fillna(False)
        plot_df = df.assign(_status=np.where(mask, "Outlier", "Typical"))
        fig = px.strip(plot_df.sample(min(len(plot_df), 8000), random_state=1), x=col, color="_status",
                       color_discrete_map={"Typical": "#9ec5f4", "Outlier": "#d03b3b"}, labels={"_status": ""})
        fig.update_traces(jitter=0.6, marker=dict(size=6, opacity=0.7))
        show_chart(apply_chart_style(fig, f"<b>{col}</b> — {int(mask.sum()):,} outliers", 300), key="ex_out_strip")
        st.caption("Fix them in **Clean & Transform → Rows → Handle outliers**.")


# --------------------------------------------------------------------------
def _time_series(ctx):
    df, types = ctx.df, ctx.types
    dates = types["datetime"]
    if not dates:
        result_card("info", "No date columns detected",
                    "Convert a column with <b>Clean & Transform → Columns → Change type → date</b> to unlock time-series analysis.")
        return
    c1, c2, c3, c4 = st.columns(4)
    date_col = c1.selectbox("Date column", dates, key="p_ex_ts_date")
    val = c2.selectbox("Value", [None] + types["numerical"], key="p_ex_ts_val", format_func=lambda v: "Row count" if v is None else v)
    freq_label = c3.selectbox("Granularity", list(eda.FREQ_OPTIONS), key="p_ex_ts_freq")
    agg = c4.selectbox("Aggregation", ["mean", "sum", "median", "min", "max", "count"], key="p_ex_ts_agg", disabled=val is None)
    c5, c6 = st.columns(2)
    grp = c5.selectbox("Split into series by", [None] + [g for g in types["groupable"] if df[g].nunique() <= 8],
                       format_func=fmt_none, key="p_ex_ts_grp")
    roll = c6.slider("Rolling mean window (periods)", 0, 52, 0, key="p_ex_ts_roll", disabled=grp is not None)
    freq = eda.FREQ_OPTIONS[freq_label] or eda.auto_frequency(df[date_col])
    ts = eda.resample_series(df, date_col, val, freq, agg, grp)
    vcol = val or "Count"
    if ts.empty:
        st.info("No data after resampling.")
        return
    if not grp:
        series = ts.set_index(date_col)[vcol]
        last, first = series.dropna().iloc[-1], series.dropna().iloc[0]
        change = (last - first) / abs(first) * 100 if first else np.nan
        peak_t = series.idxmax()
        tiles([
            {"label": "Periods", "value": f"{len(series):,}", "sub": f"granularity: {freq_label if freq_label != 'Auto' else freq}"},
            {"label": "Latest", "value": fmt_num(last), "sub": f"{series.index[-1]:%Y-%m-%d}"},
            {"label": "Change (first → last)", "value": f"{change:+.1f}%" if pd.notna(change) else "–",
             "sub": "▲ growth" if change > 0 else "▼ decline", "status": "good" if change > 0 else "crit"},
            {"label": "Peak", "value": fmt_num(series.max()), "sub": f"{peak_t:%Y-%m-%d}"},
        ])
    show_chart(plot_time_series(ts, date_col, vcol, grp, roll), key="ex_ts")

    if not grp:
        c1, c2 = st.columns(2)
        with c1:
            section("Period-over-period change")
            pct = ts.set_index(date_col)[vcol].pct_change() * 100
            fig = px.bar(x=pct.index, y=pct.values, labels={"x": "", "y": "% change"})
            fig.update_traces(marker_color=np.where(pct.values >= 0, CATEGORICAL[0], CATEGORICAL[7]), marker_line_width=0)
            show_chart(apply_chart_style(fig, "<b>% change vs previous period</b>", 340), key="ex_pct")
        with c2:
            section("Seasonality")
            show_chart(plot_seasonality_heatmap(df, date_col, val, agg if val else "count"), key="ex_season")
        section("Seasonal decomposition", "Splits the series into trend + seasonal pattern + residual noise")
        periods = {"h": 24, "D": 7, "W": 52, "MS": 12, "QS": 4, "YS": 0}
        period = st.number_input("Season length (periods)", 0, 400, periods.get(freq, 0), key=f"p_ex_ts_period_{freq}",
                                 help="E.g. 7 for daily data with a weekly cycle, 12 for monthly data with a yearly cycle.")
        s = ts.set_index(date_col)[vcol]
        if period >= 2 and len(s.dropna()) >= 2 * period:
            try:
                res = eda.seasonal_decomposition(s, int(period))
                show_chart(plot_decomposition(res, vcol), key="ex_decomp")
                denom = np.nanvar(res.seasonal + res.resid)
                strength = max(0.0, 1 - np.nanvar(res.resid) / denom) if denom > 0 else 0.0
                result_card("info", f"Seasonal strength: {strength:.0%}",
                            "Share of the de-trended variation explained by the repeating seasonal pattern "
                            "(> 60% = strong seasonality).")
            except ImportError:
                st.info("Install `statsmodels` for seasonal decomposition.")
            except Exception as e:
                st.warning(f"Decomposition failed: {e}")
        else:
            st.caption(f"Need at least {max(2, int(period)) * 2} periods for a season length of {period}.")


# --------------------------------------------------------------------------
def _pca(ctx):
    df, types = ctx.df, ctx.types
    nums = types["numerical"]
    if len(nums) < 2:
        st.info("PCA needs at least 2 numeric columns.")
        return
    cols = state.multiselect("Features", nums, key="p_ex_pca_cols", default=nums[:15])
    c1, c2 = st.columns(2)
    color = c1.selectbox("Colour points by", [None] + types["groupable"] + nums, key="p_ex_pca_color", format_func=fmt_none)
    dims = c2.radio("Projection", ["2D", "3D"], horizontal=True, key="p_ex_pca_dims")
    scores, explained, loadings = eda.run_pca(df, cols, 3)
    if scores is None:
        st.info(explained)
        return
    plot_df = scores.join(df[[color]]) if color else scores
    k = len(explained)
    tiles([{"label": f"PC{i + 1}", "value": f"{explained[i]:.1%}", "sub": "variance explained"} for i in range(min(3, k))]
          + [{"label": "Cumulative (top 3)", "value": f"{explained[:3].sum():.1%}",
              "status": "good" if explained[:3].sum() > 0.7 else "warn"}])
    c1, c2 = st.columns([1.5, 1])
    with c1:
        if dims == "3D" and "PC3" in scores:
            fig = px.scatter_3d(plot_df, x="PC1", y="PC2", z="PC3", color=color, opacity=0.75)
            fig.update_traces(marker=dict(size=3))
        else:
            fig = px.scatter(plot_df, x="PC1", y="PC2", color=color, opacity=0.7)
            fig.update_traces(marker=dict(size=7, line=dict(width=1, color="white")))
        show_chart(apply_chart_style(fig, "<b>PCA projection</b>", 520), key="ex_pca")
    with c2:
        show_chart(plot_scree(explained[:min(len(explained), 12)]), key="ex_scree")
    section("Loadings", "How much each original feature contributes to each component")
    fig = px.imshow(loadings, text_auto=".2f", aspect="auto", color_continuous_scale=DIVERGING, zmin=-1, zmax=1)
    show_chart(apply_chart_style(fig, "<b>Component loadings</b>", max(300, 30 * len(loadings) + 120)), key="ex_load")


# --------------------------------------------------------------------------
def _clustering(ctx):
    df, types = ctx.df, ctx.types
    nums = types["numerical"]
    if len(nums) < 2:
        st.info("Clustering needs at least 2 numeric columns.")
        return
    cols = state.multiselect("Features to cluster on", nums, key="p_ex_cl_cols", default=nums[:6])
    if len(cols) < 2:
        st.info("Select at least 2 features.")
        return
    with st.expander("📐 How many clusters? (elbow & silhouette)", expanded=False):
        diag = eda.kmeans_diagnostics(df, cols, 10)
        show_chart(plot_elbow(diag), key="ex_elbow")
        best_k = int(diag.loc[diag["Silhouette"].idxmax(), "k"])
        st.caption(f"Suggested k = **{best_k}** (highest silhouette score).")
    k = st.slider("Number of clusters (k)", 2, 12, 3, key="p_ex_cl_k")
    labels, profile = eda.run_kmeans(df, cols, k)
    c1, c2 = st.columns([1.4, 1])
    with c1:
        plot_df = df[cols].assign(cluster=labels).sample(min(len(df), 8000), random_state=1)
        scores, _exp, _ = eda.run_pca(df, cols, 2)
        if scores is not None:
            plot_df = scores.join(labels)
            fig = px.scatter(plot_df, x="PC1", y="PC2", color="cluster", opacity=0.75,
                             category_orders={"cluster": sorted(labels.unique(), key=lambda s: int(s.split()[-1]))})
            fig.update_traces(marker=dict(size=7, line=dict(width=1, color="white")))
            show_chart(apply_chart_style(fig, "<b>Clusters</b> (projected onto the first 2 principal components)", 480), key="ex_cl")
    with c2:
        sizes = labels.value_counts().sort_index(key=lambda idx: idx.str.split().str[-1].astype(int))
        fig = px.bar(x=sizes.index, y=sizes.values, labels={"x": "", "y": "Rows"}, color=sizes.index,
                     color_discrete_sequence=CATEGORICAL)
        fig.update_layout(showlegend=False)
        show_chart(apply_chart_style(fig, "<b>Cluster sizes</b>", 480), key="ex_cl_sizes")
    section("Cluster profiles", "Average feature value per cluster — colour shows relative standing (z-score) across clusters")
    z = (profile[cols] - profile[cols].mean()) / profile[cols].std().replace(0, 1)
    fig = px.imshow(z.T, text_auto=False, aspect="auto", color_continuous_scale=DIVERGING, zmin=-2, zmax=2,
                    labels=dict(color="z-score"))
    fig.update_traces(text=profile[cols].T.map(fmt_num).values, texttemplate="%{text}")
    show_chart(apply_chart_style(fig, "<b>Cluster fingerprint</b>", max(300, 34 * len(cols) + 140)), key="ex_cl_heat")
    show_df(profile.style.format(precision=3), height=min(40 * len(profile) + 40, 400))
    name = st.text_input("Save cluster labels as column", "cluster", key="p_ex_cl_name")
    if st.button("➕ Add cluster column to dataset", type="primary", key="ex_cl_add"):
        if ctx.filtered:
            st.warning("Clear global filters first — labels are computed on the filtered view.")
        else:
            new_df, label, code = T.add_column(st.session_state.df, name, labels.reindex(st.session_state.df.index))
            state.apply_step(new_df, f"Added K-Means (k={k}) labels as '{name}'", code)
            st.toast(f"✅ Added '{name}'")
            st.rerun()

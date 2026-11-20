"""Stats Lab: guided hypothesis testing with plain-English conclusions."""
from __future__ import annotations

import plotly.express as px
import streamlit as st

from src import stats_tests as S
from src.ui_components import html_block, page_header, result_card, section, show_chart, show_df
from src.visuals import DIVERGING, apply_chart_style, build_chart

QUESTIONS = {
    "Do two groups differ?": "Compare a numeric value between two groups (t-test / Mann-Whitney).",
    "Do several groups differ?": "Compare a numeric value across 3+ groups (ANOVA / Kruskal-Wallis + Tukey post-hoc).",
    "Are two categories related?": "Test whether two categorical variables are independent (χ² + Cramér's V).",
    "Are two numbers correlated?": "Measure and test a numeric relationship (Pearson / Spearman / Kendall).",
    "Does a mean equal a target?": "Compare a column's mean with a reference value (one-sample t / Wilcoxon).",
    "Did values change (before/after)?": "Paired comparison of two numeric columns (paired t / Wilcoxon).",
}


def _show(res: dict, alpha: float):
    level = "good" if res["verdict"] else "info"
    head = "✓ Statistically significant" if res["verdict"] else "○ Not statistically significant"
    result_card(level, f"{head} · {res['title']}", _md(res["summary"]))
    html_block(f'<span class="dv-badge info">Effect size · {res["effect"]}</span> '
               f'<span class="dv-badge neutral">α = {alpha}</span><div style="height:.6rem"></div>')
    show_df(res["table"].style.format({"Statistic": "{:.4f}", "p-value": "{:.4g}"}).map(
        lambda v: "color:#006300;font-weight:600" if isinstance(v, float) and v < alpha else "", subset=["p-value"]),
        hide_index=True)


def _md(text: str) -> str:
    parts = text.split("**")
    return "".join(f"<b>{p}</b>" if i % 2 else p for i, p in enumerate(parts))


def render(ctx):
    df, types = ctx.df, ctx.types
    page_header("🧪", "Stats Lab", "Pick a question — DataViz chooses the right test and explains the answer")
    nums, groups = types["numerical"], types["groupable"]
    cats = types["categorical"] + types["boolean"]

    c1, c2 = st.columns([3, 1])
    q = c1.selectbox("What do you want to know?", list(QUESTIONS), key="p_sl_q")
    alpha = c2.select_slider("Significance level α", [0.01, 0.05, 0.10], value=0.05, key="p_sl_alpha")
    st.caption(QUESTIONS[q])

    try:
        if q == "Do two groups differ?":
            if not nums or not groups:
                st.info("Needs a numeric column and a grouping column.")
                return
            c1, c2 = st.columns(2)
            val = c1.selectbox("Numeric value", nums, key="p_sl_2g_val")
            grp = c2.selectbox("Grouping column", [g for g in groups if g != val], key="p_sl_2g_grp")
            levels = df[grp].dropna().astype(str).value_counts().index.tolist()
            if len(levels) < 2:
                st.info("The grouping column needs at least 2 values.")
                return
            c3, c4 = st.columns(2)
            a = c3.selectbox("Group A", levels, index=0, key=f"p_sl_2g_a_{grp}")
            b = c4.selectbox("Group B", [lv for lv in levels if lv != a], index=0, key=f"p_sl_2g_b_{grp}")
            res = S.two_groups(df, val, grp, a, b, alpha)
            _show(res, alpha)
            sub = df[df[grp].astype(str).isin([a, b])].assign(**{grp: lambda d: d[grp].astype(str)})
            c1, c2 = st.columns([1.3, 1])
            with c1:
                show_chart(build_chart(sub, {"type": "Box", "x": grp, "y": val, "color": grp, "options": {"points": "outliers"},
                                             "height": 420}).fig, key="sl_2g_box")
            with c2:
                show_df(res["extra"]["groups"].style.format(precision=3), hide_index=True)

        elif q == "Do several groups differ?":
            if not nums or not groups:
                st.info("Needs a numeric column and a grouping column.")
                return
            c1, c2 = st.columns(2)
            val = c1.selectbox("Numeric value", nums, key="p_sl_mg_val")
            grp = c2.selectbox("Grouping column", [g for g in groups if g != val], key="p_sl_mg_grp")
            res = S.many_groups(df, val, grp, alpha)
            _show(res, alpha)
            c1, c2 = st.columns([1.3, 1])
            with c1:
                order = res["extra"]["groups"]["Group"].tolist()
                d = df.assign(**{grp: df[grp].astype(str)})
                d = d[d[grp].isin(order)]
                fig = px.box(d, x=grp, y=val, color=grp, category_orders={grp: order}, points=False)
                fig.update_layout(showlegend=False)
                show_chart(apply_chart_style(fig, f"<b>{val} by {grp}</b> (sorted by mean)", 440), key="sl_mg_box")
            with c2:
                show_df(res["extra"]["groups"].style.format(precision=3), hide_index=True, height=440)
            if "posthoc" in res["extra"]:
                section("Post-hoc comparisons (Tukey HSD)", "Which specific pairs differ?")
                show_df(res["extra"]["posthoc"].style.format({"Mean diff": "{:.4g}", "p-value (Tukey)": "{:.4g}"}), hide_index=True)

        elif q == "Are two categories related?":
            if len(cats) < 2:
                st.info("Needs two categorical columns.")
                return
            c1, c2 = st.columns(2)
            a = c1.selectbox("Variable A", cats, key="p_sl_chi_a")
            b = c2.selectbox("Variable B", [c for c in cats if c != a], key="p_sl_chi_b")
            res = S.chi_square(df, a, b, alpha)
            _show(res, alpha)
            c1, c2 = st.columns(2)
            with c1:
                fig = px.imshow(res["extra"]["contingency"], text_auto=True, aspect="auto",
                                color_continuous_scale=["#f3f7fc", "#2a78d6"])
                show_chart(apply_chart_style(fig, "<b>Observed counts</b>", 420), key="sl_chi_obs")
            with c2:
                r = res["extra"]["residuals"]
                lim = max(2.0, float(abs(r.values).max()))
                fig = px.imshow(r, text_auto=".1f", aspect="auto", color_continuous_scale=DIVERGING,
                                zmin=-lim, zmax=lim)
                show_chart(apply_chart_style(fig, "<b>Standardised residuals</b> (blue = more than expected)", 420), key="sl_chi_res")

        elif q == "Are two numbers correlated?":
            if len(nums) < 2:
                st.info("Needs two numeric columns.")
                return
            c1, c2 = st.columns(2)
            x = c1.selectbox("Variable X", nums, key="p_sl_cor_x")
            y = c2.selectbox("Variable Y", [n for n in nums if n != x], key="p_sl_cor_y")
            res = S.correlation_test(df, x, y, alpha)
            _show(res, alpha)
            show_chart(build_chart(df, {"type": "Scatter", "x": x, "y": y,
                                        "options": {"trendline": "OLS (linear)", "opacity": 0.6,
                                                    "marginal_x": "histogram", "marginal_y": "histogram"},
                                        "height": 520}).fig, key="sl_cor_sc")

        elif q == "Does a mean equal a target?":
            if not nums:
                st.info("Needs a numeric column.")
                return
            c1, c2 = st.columns(2)
            val = c1.selectbox("Numeric value", nums, key="p_sl_1s_val")
            mu = c2.number_input("Reference value", value=float(round(df[val].median(), 2)), key=f"p_sl_1s_mu_{val}")
            res = S.one_sample(df, val, mu, alpha)
            _show(res, alpha)
            fig = build_chart(df, {"type": "Histogram", "x": val, "height": 400}).fig
            fig.add_vline(x=mu, line_width=2, line_color="#eb6834", annotation_text=f"reference {mu:g}")
            fig.add_vline(x=df[val].mean(), line_width=2, line_color="#0b0b0b", annotation_text="mean",
                          annotation_position="bottom right")
            show_chart(fig, key="sl_1s_hist")

        elif q == "Did values change (before/after)?":
            if len(nums) < 2:
                st.info("Needs two numeric columns measured on the same rows.")
                return
            c1, c2 = st.columns(2)
            b = c1.selectbox("Before", nums, key="p_sl_p_b")
            a = c2.selectbox("After", [n for n in nums if n != b], key="p_sl_p_a")
            res = S.paired(df, b, a, alpha)
            _show(res, alpha)
            d = (df[a] - df[b]).dropna().to_frame("difference")
            fig = px.histogram(d, x="difference", nbins=40)
            fig.add_vline(x=0, line_width=2, line_color="#0b0b0b")
            show_chart(apply_chart_style(fig, f"<b>Distribution of differences</b> ({a} − {b})", 380), key="sl_p_hist")
    except ValueError as e:
        st.warning(str(e), icon="⚠️")

    with st.expander("📘 How to read these results"):
        st.markdown(
            "- **p-value** — the probability of seeing a difference at least this large if there were truly no effect. "
            "Below α ⇒ *statistically significant*.\n"
            "- **Effect size** — *how big* the difference is. With large samples, tiny effects can be significant; "
            "always check both.\n"
            "- **Parametric vs non-parametric** — t-test / ANOVA / Pearson assume roughly normal data; Mann-Whitney / "
            "Kruskal-Wallis / Spearman don't. When they disagree, trust the non-parametric result for skewed data.\n"
            "- Significance ≠ causation."
        )

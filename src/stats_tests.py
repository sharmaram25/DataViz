"""
Hypothesis-testing lab. Each function returns a dict:

    {"title", "verdict" (bool: significant?), "summary" (plain English),
     "table" (DataFrame of test statistics), "effect" (label), "extra" {...}}
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from scipy import stats

from src.eda import cramers_v


def _num(s):
    return pd.to_numeric(s, errors="coerce")


def _fmt_p(p: float) -> str:
    return "< 0.0001" if p < 1e-4 else f"{p:.4f}"


def effect_label(value: float, thresholds: tuple, labels=("negligible", "small", "medium", "large")) -> str:
    v = abs(value)
    for t, lab in zip(thresholds, labels):
        if v < t:
            return lab
    return labels[-1]


def two_groups(df, value, group, a, b, alpha=0.05, paired=False):
    x = _num(df.loc[df[group].astype(str) == str(a), value]).dropna()
    y = _num(df.loc[df[group].astype(str) == str(b), value]).dropna()
    if len(x) < 3 or len(y) < 3:
        raise ValueError("Each group needs at least 3 observations.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t = stats.ttest_ind(x, y, equal_var=False)
        u = stats.mannwhitneyu(x, y, alternative="two-sided")
        lev = stats.levene(x, y)
    pooled = np.sqrt(((len(x) - 1) * x.var() + (len(y) - 1) * y.var()) / (len(x) + len(y) - 2))
    d = (x.mean() - y.mean()) / pooled if pooled else 0.0
    eff = effect_label(d, (0.2, 0.5, 0.8))
    table = pd.DataFrame([
        ["Welch's t-test (means)", t.statistic, t.pvalue],
        ["Mann-Whitney U (distributions)", u.statistic, u.pvalue],
        ["Levene (equal variances)", lev.statistic, lev.pvalue],
    ], columns=["Test", "Statistic", "p-value"])
    groups = pd.DataFrame({
        "Group": [str(a), str(b)], "n": [len(x), len(y)], "Mean": [x.mean(), y.mean()],
        "Median": [x.median(), y.median()], "Std": [x.std(), y.std()],
    })
    sig = t.pvalue < alpha
    diff = x.mean() - y.mean()
    direction = "higher" if diff > 0 else "lower"
    summary = (
        f"The mean **{value}** for **{a}** is {abs(diff):,.4g} {direction} than for **{b}** "
        f"(Welch p = {_fmt_p(t.pvalue)}). "
        + (f"This difference is **statistically significant** at α = {alpha}. " if sig else
           f"This difference is **not statistically significant** at α = {alpha} — it could plausibly be noise. ")
        + f"Effect size (Cohen's d = {d:.2f}) is **{eff}**."
    )
    if (u.pvalue < alpha) != sig:
        summary += " Note: the rank-based Mann-Whitney test disagrees, so check the distributions for skew/outliers."
    return {"title": f"{value}: {a} vs {b}", "verdict": sig, "summary": summary, "table": table,
            "effect": f"Cohen's d = {d:.2f} ({eff})", "extra": {"groups": groups}}


def many_groups(df, value, group, alpha=0.05, max_groups=20):
    d = df[[value, group]].dropna()
    d = d.assign(**{value: _num(d[value])}).dropna()
    top = d[group].astype(str).value_counts().head(max_groups).index
    d = d[d[group].astype(str).isin(top)]
    samples = [g[value].values for _, g in d.groupby(d[group].astype(str)) if len(g) >= 2]
    names = [n for n, g in d.groupby(d[group].astype(str)) if len(g) >= 2]
    if len(samples) < 2:
        raise ValueError("Need at least two groups with ≥ 2 observations.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        f = stats.f_oneway(*samples)
        k = stats.kruskal(*samples)
    grand = d[value].mean()
    ss_b = sum(len(s) * (s.mean() - grand) ** 2 for s in samples)
    ss_t = ((d[value] - grand) ** 2).sum()
    eta2 = ss_b / ss_t if ss_t else 0
    eff = effect_label(eta2, (0.01, 0.06, 0.14))
    table = pd.DataFrame([["One-way ANOVA (means)", f.statistic, f.pvalue],
                          ["Kruskal-Wallis H (distributions)", k.statistic, k.pvalue]],
                         columns=["Test", "Statistic", "p-value"])
    groups = d.groupby(d[group].astype(str))[value].agg(["count", "mean", "median", "std"]).reset_index()
    groups.columns = ["Group", "n", "Mean", "Median", "Std"]
    sig = f.pvalue < alpha
    summary = (
        f"Across **{len(samples)}** groups of **{group}**, mean **{value}** "
        + ("**differs significantly**" if sig else "does **not differ significantly**")
        + f" (ANOVA p = {_fmt_p(f.pvalue)}, α = {alpha}). "
        f"η² = {eta2:.3f}: **{group}** explains {eta2:.1%} of the variance in {value} (**{eff}** effect)."
    )
    extra = {"groups": groups.sort_values("Mean", ascending=False)}
    if sig and len(samples) <= 12:
        try:
            res = stats.tukey_hsd(*samples)
            rows = []
            for i in range(len(names)):
                for j in range(i + 1, len(names)):
                    rows.append([names[i], names[j], samples[i].mean() - samples[j].mean(), res.pvalue[i, j]])
            ph = pd.DataFrame(rows, columns=["Group A", "Group B", "Mean diff", "p-value (Tukey)"])
            ph["Significant"] = np.where(ph["p-value (Tukey)"] < alpha, "✅ yes", "— no")
            extra["posthoc"] = ph.sort_values("p-value (Tukey)")
        except Exception:
            pass
    return {"title": f"{value} across {group}", "verdict": sig, "summary": summary, "table": table,
            "effect": f"η² = {eta2:.3f} ({eff})", "extra": extra}


def chi_square(df, a, b, alpha=0.05, max_levels=30):
    d = df[[a, b]].dropna().astype(str)
    for c in (a, b):
        top = d[c].value_counts().head(max_levels).index
        d = d[d[c].isin(top)]
    table = pd.crosstab(d[a], d[b])
    if table.shape[0] < 2 or table.shape[1] < 2:
        raise ValueError("Both variables need at least 2 categories.")
    chi2, p, dof, expected = stats.chi2_contingency(table)
    v = cramers_v(d[a], d[b])
    eff = effect_label(v, (0.1, 0.3, 0.5))
    low_exp = (expected < 5).mean()
    sig = p < alpha
    summary = (
        f"**{a}** and **{b}** are " + ("**associated**" if sig else "**not significantly associated**")
        + f" (χ² = {chi2:,.2f}, df = {dof}, p = {_fmt_p(p)}, α = {alpha}). "
        f"Cramér's V = {v:.3f} (**{eff}** association)."
    )
    if low_exp > 0.2:
        summary += f" ⚠️ {low_exp:.0%} of expected counts are below 5 — the χ² approximation may be unreliable."
    stats_t = pd.DataFrame([["Chi-square test of independence", chi2, p]], columns=["Test", "Statistic", "p-value"])
    resid = (table - expected) / np.sqrt(expected)
    return {"title": f"{a} × {b}", "verdict": sig, "summary": summary, "table": stats_t,
            "effect": f"Cramér's V = {v:.3f} ({eff})",
            "extra": {"contingency": table, "residuals": pd.DataFrame(resid, index=table.index, columns=table.columns)}}


def correlation_test(df, x, y, alpha=0.05):
    d = df[[x, y]].apply(_num).dropna()
    if len(d) < 5:
        raise ValueError("Need at least 5 complete pairs.")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pr = stats.pearsonr(d[x], d[y])
        sr = stats.spearmanr(d[x], d[y])
        kt = stats.kendalltau(d[x].head(5000), d[y].head(5000))
        lr = stats.linregress(d[x], d[y])
    try:
        ci = pr.confidence_interval(0.95)
        ci_txt = f" (95% CI {ci.low:.3f} to {ci.high:.3f})"
    except Exception:
        ci_txt = ""
    eff = effect_label(pr.statistic, (0.1, 0.3, 0.5), ("negligible", "weak", "moderate", "strong"))
    sig = pr.pvalue < alpha
    table = pd.DataFrame([["Pearson r (linear)", pr.statistic, pr.pvalue],
                          ["Spearman ρ (monotonic)", sr.statistic, sr.pvalue],
                          ["Kendall τ (rank)", kt.statistic, kt.pvalue]], columns=["Test", "Statistic", "p-value"])
    direction = "positive" if pr.statistic > 0 else "negative"
    summary = (
        f"**{x}** and **{y}** show a **{eff} {direction}** linear relationship: r = {pr.statistic:.3f}{ci_txt}, "
        f"p = {_fmt_p(pr.pvalue)} — " + ("**statistically significant**." if sig else "**not statistically significant**.")
        + f" A linear fit explains R² = {lr.rvalue ** 2:.1%} of the variance "
        f"({y} ≈ {lr.slope:.4g} × {x} + {lr.intercept:.4g})."
    )
    if abs(sr.statistic) - abs(pr.statistic) > 0.15:
        summary += " Spearman is notably stronger than Pearson, suggesting a monotonic but non-linear relationship."
    return {"title": f"{x} ↔ {y}", "verdict": sig, "summary": summary, "table": table,
            "effect": f"r = {pr.statistic:.3f} ({eff})", "extra": {"slope": lr.slope, "intercept": lr.intercept}}


def one_sample(df, value, mu, alpha=0.05):
    x = _num(df[value]).dropna()
    if len(x) < 3:
        raise ValueError("Need at least 3 observations.")
    t = stats.ttest_1samp(x, mu)
    w = stats.wilcoxon(x - mu) if (x - mu).abs().sum() > 0 else None
    d = (x.mean() - mu) / x.std() if x.std() else 0
    eff = effect_label(d, (0.2, 0.5, 0.8))
    rows = [["One-sample t-test", t.statistic, t.pvalue]]
    if w is not None:
        rows.append(["Wilcoxon signed-rank", w.statistic, w.pvalue])
    sig = t.pvalue < alpha
    se = x.std() / np.sqrt(len(x))
    tcrit = stats.t.ppf(1 - alpha / 2, len(x) - 1)
    summary = (
        f"The mean of **{value}** is {x.mean():,.4g} (95% CI {x.mean() - tcrit * se:,.4g} – {x.mean() + tcrit * se:,.4g}), "
        + ("which **differs significantly**" if sig else "which does **not differ significantly**")
        + f" from {mu:,.4g} (p = {_fmt_p(t.pvalue)}). Effect size d = {d:.2f} (**{eff}**)."
    )
    return {"title": f"{value} vs {mu}", "verdict": sig, "summary": summary,
            "table": pd.DataFrame(rows, columns=["Test", "Statistic", "p-value"]), "effect": f"d = {d:.2f} ({eff})", "extra": {}}


def paired(df, before, after, alpha=0.05):
    d = df[[before, after]].apply(_num).dropna()
    if len(d) < 3:
        raise ValueError("Need at least 3 complete pairs.")
    diff = d[after] - d[before]
    t = stats.ttest_rel(d[after], d[before])
    w = stats.wilcoxon(diff) if diff.abs().sum() > 0 else None
    dz = diff.mean() / diff.std() if diff.std() else 0
    eff = effect_label(dz, (0.2, 0.5, 0.8))
    rows = [["Paired t-test", t.statistic, t.pvalue]]
    if w is not None:
        rows.append(["Wilcoxon signed-rank", w.statistic, w.pvalue])
    sig = t.pvalue < alpha
    summary = (
        f"On average **{after}** is {abs(diff.mean()):,.4g} {'higher' if diff.mean() > 0 else 'lower'} than **{before}** "
        f"across {len(d):,} pairs — " + ("**statistically significant**" if sig else "**not statistically significant**")
        + f" (p = {_fmt_p(t.pvalue)}). Effect size d_z = {dz:.2f} (**{eff}**)."
    )
    return {"title": f"{before} → {after}", "verdict": sig, "summary": summary,
            "table": pd.DataFrame(rows, columns=["Test", "Statistic", "p-value"]), "effect": f"d_z = {dz:.2f} ({eff})", "extra": {}}

"""
Statistical / ML-lite analysis engine.

Heavy functions are wrapped in ``st.cache_data`` so they are computed once
per (data, parameters) combination. Every function also works outside a
Streamlit runtime (tests, notebooks).
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import streamlit as st
from scipy import stats

from src.preprocess import classify_columns, safe_duplicated_count, safe_nunique

SAMPLE_CAP = 5000


def _sample(df: pd.DataFrame, n: int = SAMPLE_CAP, seed: int = 42) -> pd.DataFrame:
    return df.sample(n, random_state=seed) if len(df) > n else df


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


# --------------------------------------------------------------------------
# Descriptive statistics
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def numeric_profile(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    n = max(len(df), 1)
    for col in columns:
        s = _num(df[col])
        v = s.dropna()
        if v.empty:
            continue
        q1, med, q3 = v.quantile([0.25, 0.5, 0.75])
        iqr = q3 - q1
        outliers = int(((v < q1 - 1.5 * iqr) | (v > q3 + 1.5 * iqr)).sum()) if iqr > 0 else 0
        rows.append({
            "Column": col, "Count": int(v.count()), "Missing %": round(s.isna().sum() / n * 100, 2),
            "Mean": v.mean(), "Std": v.std(), "Min": v.min(), "Q1": q1, "Median": med, "Q3": q3,
            "Max": v.max(), "Skew": v.skew() if len(v) > 2 else np.nan,
            "Kurtosis": v.kurtosis() if len(v) > 3 else np.nan,
            "Zeros %": round((v == 0).sum() / n * 100, 2), "Outliers (IQR)": outliers,
        })
    return pd.DataFrame(rows).set_index("Column") if rows else pd.DataFrame()


@st.cache_data(show_spinner=False)
def categorical_profile(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows = []
    n = max(len(df), 1)
    for col in columns:
        s = df[col]
        vc = s.astype(str).where(s.notna()).value_counts()
        rows.append({
            "Column": col, "Count": int(s.notna().sum()), "Missing %": round(s.isna().sum() / n * 100, 2),
            "Unique": safe_nunique(s), "Top value": vc.index[0] if len(vc) else None,
            "Top freq": int(vc.iloc[0]) if len(vc) else 0,
            "Top share %": round(vc.iloc[0] / max(vc.sum(), 1) * 100, 2) if len(vc) else 0,
        })
    return pd.DataFrame(rows).set_index("Column") if rows else pd.DataFrame()


def value_counts_table(series: pd.Series, top: int = 20) -> pd.DataFrame:
    vc = series.astype(str).where(series.notna(), "(missing)").value_counts()
    total = vc.sum()
    out = vc.head(top).rename_axis("Value").reset_index(name="Count")
    if len(vc) > top:
        out = pd.concat([out, pd.DataFrame({"Value": [f"Other ({len(vc) - top} values)"],
                                            "Count": [int(vc.iloc[top:].sum())]})], ignore_index=True)
    out["Share %"] = (out["Count"] / max(total, 1) * 100).round(2)
    return out


# --------------------------------------------------------------------------
# Normality
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def normality_tests(series: pd.Series, alpha: float = 0.05) -> pd.DataFrame:
    data = _num(series).dropna()
    rows = []
    if len(data) < 8:
        return pd.DataFrame()
    data = data.sample(SAMPLE_CAP, random_state=42) if len(data) > SAMPLE_CAP else data
    if data.nunique() < 3:
        return pd.DataFrame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            k2, p = stats.normaltest(data)
            rows.append(("D'Agostino-Pearson K²", k2, p))
        except Exception:
            pass
        try:
            w, p = stats.shapiro(data)
            rows.append(("Shapiro-Wilk", w, p))
        except Exception:
            pass
        try:
            ks = stats.kstest((data - data.mean()) / data.std(ddof=1), "norm")
            rows.append(("Kolmogorov-Smirnov", ks.statistic, ks.pvalue))
        except Exception:
            pass
        try:
            jb = stats.jarque_bera(data)
            rows.append(("Jarque-Bera", jb.statistic, jb.pvalue))
        except Exception:
            pass
    out = pd.DataFrame(rows, columns=["Test", "Statistic", "p-value"])
    out["Normal?"] = np.where(out["p-value"] > alpha, "✅ Consistent with normal", "❌ Not normal")
    return out


def test_normality(df, column):
    """Back-compat: (statistic, p_value, is_normal)."""
    res = normality_tests(df[column]) if column in df.columns else pd.DataFrame()
    if res.empty:
        return None, None, False
    row = res.iloc[0]
    return row["Statistic"], row["p-value"], row["p-value"] > 0.05


@st.cache_data(show_spinner=False)
def qq_data(series: pd.Series) -> tuple[np.ndarray, np.ndarray, float, float]:
    data = _num(series).dropna()
    data = data.sample(SAMPLE_CAP, random_state=42) if len(data) > SAMPLE_CAP else data
    (osm, osr), (slope, intercept, _r) = stats.probplot(data, dist="norm")
    return osm, osr, slope, intercept


# --------------------------------------------------------------------------
# Correlation & association
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def correlation_matrix(df: pd.DataFrame, columns: list[str] | None = None, method: str = "pearson"):
    cols = columns or classify_columns(df)["numerical"]
    if len(cols) < 2:
        return None
    data = df[cols].apply(_num)
    if method == "kendall":
        data = _sample(data, 3000)
    return data.corr(method=method)


def top_correlation_pairs(corr: pd.DataFrame, n: int = 15) -> pd.DataFrame:
    if corr is None or corr.empty:
        return pd.DataFrame()
    mask = np.triu(np.ones(corr.shape, dtype=bool), k=1)
    pairs = corr.where(mask).stack().rename("r").reset_index()
    pairs.columns = ["Feature A", "Feature B", "r"]
    pairs["|r|"] = pairs["r"].abs()
    pairs["Strength"] = pd.cut(pairs["|r|"], [-0.01, 0.1, 0.3, 0.5, 0.7, 1.01],
                               labels=["Negligible", "Weak", "Moderate", "Strong", "Very strong"])
    return pairs.sort_values("|r|", ascending=False).head(n).reset_index(drop=True)


def cramers_v(x: pd.Series, y: pd.Series) -> float:
    """Bias-corrected Cramér's V between two categorical series."""
    table = pd.crosstab(x, y)
    if table.shape[0] < 2 or table.shape[1] < 2:
        return 0.0
    chi2 = stats.chi2_contingency(table, correction=False)[0]
    n = table.values.sum()
    phi2 = chi2 / n
    r, k = table.shape
    phi2corr = max(0, phi2 - (k - 1) * (r - 1) / max(n - 1, 1))
    rcorr = r - (r - 1) ** 2 / max(n - 1, 1)
    kcorr = k - (k - 1) ** 2 / max(n - 1, 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else 0.0


def correlation_ratio(categories: pd.Series, values: pd.Series) -> float:
    """Eta (η): association between a categorical and a numeric variable, 0..1."""
    d = pd.DataFrame({"c": categories, "v": _num(values)}).dropna()
    if d.empty or d["c"].nunique() < 2:
        return 0.0
    grand = d["v"].mean()
    groups = d.groupby("c", observed=True)["v"]
    ss_between = (groups.count() * (groups.mean() - grand) ** 2).sum()
    ss_total = ((d["v"] - grand) ** 2).sum()
    return float(np.sqrt(ss_between / ss_total)) if ss_total > 0 else 0.0


@st.cache_data(show_spinner=False)
def cramers_v_matrix(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    data = _sample(df[columns])
    m = pd.DataFrame(np.eye(len(columns)), index=columns, columns=columns)
    for i, a in enumerate(columns):
        for b in columns[i + 1:]:
            v = cramers_v(data[a], data[b])
            m.loc[a, b] = m.loc[b, a] = v
    return m


# --------------------------------------------------------------------------
# Target / driver analysis
# --------------------------------------------------------------------------

@st.cache_data(show_spinner="Ranking features…")
def target_analysis(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """
    Rank every feature by its association with ``target``. Uses
    Pearson |r| (numeric↔numeric), η correlation ratio (categorical↔numeric),
    Cramér's V (categorical↔categorical) plus mutual information.
    """
    types = classify_columns(df)
    kinds = types["kinds"]
    data = _sample(df.dropna(subset=[target]))
    t_kind = kinds[target]
    t_is_num = t_kind == "numerical" and safe_nunique(data[target]) > 10
    rows = []
    for col in df.columns:
        if col == target or kinds[col] in ("text", "datetime"):
            continue
        f_is_num = kinds[col] == "numerical"
        try:
            if t_is_num and f_is_num:
                r = data[[col, target]].apply(_num).corr().iloc[0, 1]
                rows.append((col, "numeric", "Pearson r", abs(r), r))
            elif t_is_num and not f_is_num:
                rows.append((col, "categorical", "η (correlation ratio)", correlation_ratio(data[col], data[target]), np.nan))
            elif not t_is_num and f_is_num:
                rows.append((col, "numeric", "η (correlation ratio)", correlation_ratio(data[target], data[col]), np.nan))
            else:
                rows.append((col, "categorical", "Cramér's V", cramers_v(data[col], data[target]), np.nan))
        except Exception:
            continue
    out = pd.DataFrame(rows, columns=["Feature", "Feature type", "Method", "Strength", "Signed r"])
    if out.empty:
        return out

    # Mutual information (captures non-linear relationships)
    try:
        from sklearn.feature_selection import mutual_info_classif, mutual_info_regression

        feats = out["Feature"].tolist()
        X = pd.DataFrame(index=data.index)
        discrete = []
        for f in feats:
            if kinds[f] == "numerical":
                X[f] = _num(data[f]).fillna(_num(data[f]).median())
                discrete.append(False)
            else:
                X[f] = data[f].astype(str).factorize()[0]
                discrete.append(True)
        if t_is_num:
            y = _num(data[target])
            mi = mutual_info_regression(X, y, discrete_features=discrete, random_state=0)
        else:
            y = data[target].astype(str).factorize()[0]
            mi = mutual_info_classif(X, y, discrete_features=discrete, random_state=0)
        out["Mutual info"] = mi
    except Exception:
        out["Mutual info"] = np.nan
    return out.sort_values("Strength", ascending=False).reset_index(drop=True)


def compute_target_correlation(df, target_col):
    """Back-compat: numeric correlations vs target, excluding the target itself."""
    nums = classify_columns(df)["numerical"]
    if target_col not in nums:
        return pd.DataFrame()
    corrs = df[nums].corrwith(df[target_col]).drop(target_col, errors="ignore").reset_index()
    corrs.columns = ["Feature", "Correlation"]
    corrs["AbsCorrelation"] = corrs["Correlation"].abs()
    return corrs.sort_values("AbsCorrelation", ascending=False).reset_index(drop=True)


# --------------------------------------------------------------------------
# Outliers
# --------------------------------------------------------------------------

def outlier_mask(series: pd.Series, method: str = "IQR", threshold: float = 1.5) -> pd.Series:
    s = _num(series)
    if method == "IQR":
        q1, q3 = s.quantile([0.25, 0.75])
        iqr = q3 - q1
        return (s < q1 - threshold * iqr) | (s > q3 + threshold * iqr)
    if method == "Z-score":
        z = (s - s.mean()) / s.std(ddof=0)
        return z.abs() > threshold
    if method == "Modified Z (MAD)":
        med = s.median()
        mad = (s - med).abs().median()
        if mad == 0:
            return pd.Series(False, index=s.index)
        return (0.6745 * (s - med) / mad).abs() > threshold
    raise ValueError(method)


@st.cache_data(show_spinner=False)
def outlier_summary(df: pd.DataFrame, columns: list[str], method: str, threshold: float) -> pd.DataFrame:
    rows = []
    for col in columns:
        s = _num(df[col])
        m = outlier_mask(s, method, threshold)
        n_out = int(m.sum())
        rows.append({"Column": col, "Outliers": n_out, "Outlier %": round(n_out / max(s.notna().sum(), 1) * 100, 2),
                     "Min outlier": s[m].min() if n_out else np.nan, "Max outlier": s[m].max() if n_out else np.nan})
    return pd.DataFrame(rows).sort_values("Outliers", ascending=False).reset_index(drop=True)


@st.cache_data(show_spinner="Running Isolation Forest…")
def isolation_forest(df: pd.DataFrame, columns: list[str], contamination: float = 0.02) -> pd.Series:
    from sklearn.ensemble import IsolationForest

    X = df[columns].apply(_num)
    X = X.fillna(X.median())
    model = IsolationForest(contamination=contamination, random_state=42, n_estimators=150)
    labels = model.fit_predict(X)
    return pd.Series(labels == -1, index=df.index, name="is_anomaly")


def detect_outliers_iqr(df, column):
    if column not in df.columns:
        return []
    return df.index[outlier_mask(df[column]).fillna(False)].tolist()


# --------------------------------------------------------------------------
# Time series
# --------------------------------------------------------------------------

FREQ_OPTIONS = {"Auto": None, "Hour": "h", "Day": "D", "Week": "W", "Month": "MS", "Quarter": "QS", "Year": "YS"}


def auto_frequency(dates: pd.Series) -> str:
    d = dates.dropna()
    if d.empty:
        return "D"
    span = (d.max() - d.min()).days
    if span <= 3:
        return "h"
    if span <= 180:
        return "D"
    if span <= 3 * 365:
        return "W"
    if span <= 25 * 365:
        return "MS"
    return "YS"


@st.cache_data(show_spinner=False)
def resample_series(df: pd.DataFrame, date_col: str, value_col: str | None, freq: str | None,
                    agg: str = "mean", group_col: str | None = None) -> pd.DataFrame:
    d = df[[c for c in {date_col, value_col, group_col} if c]].copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col])
    freq = freq or auto_frequency(d[date_col])
    if value_col is None:
        d["_count"] = 1
        value_col, agg = "_count", "sum"
    keys = [pd.Grouper(key=date_col, freq=freq)] + ([group_col] if group_col else [])
    out = d.groupby(keys, observed=True)[value_col].agg(agg).reset_index()
    if value_col == "_count":
        out = out.rename(columns={"_count": "Count"})
    return out


@st.cache_data(show_spinner=False)
def seasonal_decomposition(series: pd.Series, period: int, model: str = "additive"):
    from statsmodels.tsa.seasonal import seasonal_decompose

    s = series.interpolate(limit_direction="both")
    return seasonal_decompose(s, model=model, period=period)


# --------------------------------------------------------------------------
# PCA & clustering
# --------------------------------------------------------------------------

def _prepare_matrix(df: pd.DataFrame, columns: list[str]):
    from sklearn.preprocessing import StandardScaler

    X = df[columns].apply(_num)
    X = X.loc[:, X.std() > 0]
    X = X.fillna(X.median())
    return X, StandardScaler().fit_transform(X)


@st.cache_data(show_spinner="Running PCA…")
def run_pca(df: pd.DataFrame, columns: list[str], n_components: int = 3):
    """Returns (scores DataFrame indexed like df sample, explained ratio, loadings) or (None, msg, None)."""
    from sklearn.decomposition import PCA

    if len(columns) < 2:
        return None, "Select at least 2 numeric columns.", None
    data = _sample(df)
    X, Xs = _prepare_matrix(data, columns)
    if X.shape[1] < 2 or X.shape[0] < 5:
        return None, "Not enough variable numeric data for PCA.", None
    k = min(n_components, X.shape[1], X.shape[0])
    pca = PCA(n_components=k).fit(Xs)
    scores = pd.DataFrame(pca.transform(Xs), columns=[f"PC{i + 1}" for i in range(k)], index=data.index)
    loadings = pd.DataFrame(pca.components_.T, index=X.columns, columns=scores.columns)
    full = PCA().fit(Xs).explained_variance_ratio_
    return scores, full, loadings


def run_pca_analysis(df):
    """Back-compat wrapper (previous API)."""
    cols = classify_columns(df)["numerical"]
    scores, var, _ = run_pca(df, cols, 3)
    if scores is None:
        return None, var
    scores["Color_Dimension"] = "Sample"
    return scores, var


@st.cache_data(show_spinner="Evaluating cluster counts…")
def kmeans_diagnostics(df: pd.DataFrame, columns: list[str], k_max: int = 10) -> pd.DataFrame:
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score

    data = _sample(df, 4000)
    _X, Xs = _prepare_matrix(data, columns)
    rows = []
    for k in range(2, min(k_max, len(Xs) - 1) + 1):
        km = KMeans(n_clusters=k, n_init=5, random_state=42).fit(Xs)
        sil = silhouette_score(Xs, km.labels_, sample_size=min(2000, len(Xs)), random_state=42)
        rows.append({"k": k, "Inertia": km.inertia_, "Silhouette": sil})
    return pd.DataFrame(rows)


@st.cache_data(show_spinner="Clustering…")
def run_kmeans(df: pd.DataFrame, columns: list[str], k: int) -> tuple[pd.Series, pd.DataFrame]:
    """Cluster every row. Returns (labels aligned to df.index, cluster profile table)."""
    from sklearn.cluster import KMeans

    X, Xs = _prepare_matrix(df, columns)
    km = KMeans(n_clusters=k, n_init=10, random_state=42).fit(Xs)
    labels = pd.Series([f"Cluster {i + 1}" for i in km.labels_], index=df.index, name="cluster")
    profile = X.assign(cluster=labels.values).groupby("cluster").mean()
    profile.insert(0, "Size", labels.value_counts().reindex(profile.index))
    return labels, profile


# --------------------------------------------------------------------------
# Smart insights
# --------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def smart_insights(df: pd.DataFrame) -> list[dict]:
    """
    Automated diagnostics. Each insight: {level, title, detail, column}
    level ∈ critical | warning | info | good.
    """
    out: list[dict] = []
    if df.empty:
        return [{"level": "critical", "title": "Empty dataset", "detail": "No rows to analyse.", "column": None}]
    n = len(df)
    types = classify_columns(df)

    def add(level, title, detail, column=None):
        out.append({"level": level, "title": title, "detail": detail, "column": column})

    dups = safe_duplicated_count(df)
    if dups:
        add("warning", f"{dups:,} duplicate rows", f"{dups / n:.1%} of rows are exact duplicates. Use **Clean → Remove duplicates**.")

    miss = df.isna().mean()
    for col, frac in miss.items():
        if frac == 1:
            add("critical", f"'{col}' is completely empty", "Drop it — it carries no information.", col)
        elif frac > 0.5:
            add("critical", f"'{col}' is {frac:.0%} missing", "Consider dropping it or imputing carefully.", col)
        elif frac > 0.05:
            add("warning", f"'{col}' is {frac:.1%} missing", "Impute (median / mode) or drop affected rows.", col)

    for col in df.columns:
        u = safe_nunique(df[col])
        if u == 1 and miss[col] < 1:
            add("warning", f"'{col}' is constant", "Only one distinct value — it adds no signal.", col)

    for col in types["text"] + types["categorical"]:
        s = df[col].dropna()
        u = safe_nunique(s)
        if u > 0.95 * n and n > 20:
            add("info", f"'{col}' looks like an identifier", f"{u:,} unique values ({u / n:.0%}). Exclude it from modelling.", col)
            continue
        if col in types["categorical"] and len(s):
            top_share = s.astype(str).value_counts(normalize=True).iloc[0]
            if top_share > 0.9 and u > 1:
                add("info", f"'{col}' is highly imbalanced", f"The most common value covers {top_share:.0%} of rows.", col)
        sample = s.astype(str).head(5000)
        if len(sample):
            stripped = sample.str.strip()
            if (stripped != sample).mean() > 0.001:
                add("warning", f"'{col}' has stray whitespace", "Leading/trailing spaces split identical categories. Use **Clean → Text cleanup**.", col)
            if col in types["categorical"]:
                lowered = stripped.str.lower()
                if lowered.nunique() < stripped.nunique():
                    add("warning", f"'{col}' has inconsistent casing",
                        f"{stripped.nunique()} variants collapse to {lowered.nunique()} when case is ignored.", col)

    for col in types["numerical"]:
        s = _num(df[col]).dropna()
        if len(s) < 10:
            continue
        sk = s.skew()
        if abs(sk) > 2:
            add("info", f"'{col}' is heavily {'right' if sk > 0 else 'left'}-skewed",
                f"Skew = {sk:.2f}. A log/sqrt transform may help models and charts.", col)
        frac_out = outlier_mask(s).mean()
        if frac_out > 0.05:
            add("warning", f"'{col}' has many outliers", f"{frac_out:.1%} of values fall outside 1.5×IQR.", col)

    if len(types["numerical"]) >= 2:
        corr = correlation_matrix(df, types["numerical"][:40])
        pairs = top_correlation_pairs(corr, 50) if corr is not None else pd.DataFrame()
        for _, p in pairs[pairs["|r|"] > 0.95].iterrows() if not pairs.empty else []:
            add("info", f"'{p['Feature A']}' ≈ '{p['Feature B']}'",
                f"Correlation r = {p['r']:.3f}. These carry near-identical information (multicollinearity).")

    for col in types["datetime"]:
        s = df[col].dropna()
        if len(s):
            add("good", f"'{col}' spans {s.min():%Y-%m-%d} → {s.max():%Y-%m-%d}",
                "Time-series analysis is available in **Explore → Time Series**.", col)

    order = {"critical": 0, "warning": 1, "info": 2, "good": 3}
    return sorted(out, key=lambda i: order[i["level"]])


# --------------------------------------------------------------------------
# Back-compat class API
# --------------------------------------------------------------------------

class EDAAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df

    def get_descriptive_stats(self):
        return numeric_profile(self.df, classify_columns(self.df)["numerical"])

    def compute_correlation_matrix(self, method="pearson"):
        return correlation_matrix(self.df, None, method)

    def detect_outliers_iqr(self, column):
        return detect_outliers_iqr(self.df, column)

    def get_smart_insights(self):
        return [f"{i['title']}: {i['detail']}" for i in smart_insights(self.df)]

    def run_pca_analysis(self):
        return run_pca_analysis(self.df)

    def test_normality(self, column):
        return test_normality(self.df, column)

    def compute_target_correlation(self, target_col):
        return compute_target_correlation(self.df, target_col)


# Names kept for the stress test / old imports
_compute_correlation_matrix = correlation_matrix
_run_pca_analysis = run_pca_analysis
_detect_outliers_iqr = detect_outliers_iqr

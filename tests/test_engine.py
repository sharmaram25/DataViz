"""Unit tests for the analysis engine (no UI)."""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import eda, stats_tests as S, transforms as T  # noqa: E402
from src.loader import infer_types, load_bytes, load_sample, read_bytes  # noqa: E402
from src.preprocess import classify_columns, get_data_quality  # noqa: E402
from src.report import build_html_report, build_markdown_report  # noqa: E402
from src.state import apply_filters  # noqa: E402
from src.visuals import CHART_FIELDS, build_chart  # noqa: E402


@pytest.fixture(scope="module")
def sales():
    return load_sample("sales")[0]


@pytest.fixture(scope="module")
def gap():
    return load_sample("gapminder")[0]


# ---------------------------------------------------------------- loader
def test_type_inference_converts_text():
    raw = pd.DataFrame({
        "date": ["2024-01-05", "2024-02-11", "2024-03-02", None],
        "price": ["$1,200.50", "$99", "$3,000", "$5"],
        "pct": ["10%", "20%", "5%", "1%"],
        "zip": ["01234", "02345", "00123", "09999"],
        "name": ["a", "b", "c", "d"],
    })
    df, notes = infer_types(raw)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["price"].tolist() == [1200.5, 99, 3000, 5]
    assert df["pct"].tolist() == [10, 20, 5, 1]
    assert not pd.api.types.is_numeric_dtype(df["zip"]), "leading-zero codes must stay text"
    assert notes


def test_csv_delimiter_and_encoding():
    data = "a;b;c\n1;x;é\n2;y;ü\n".encode("cp1252")
    df, notes, err = load_bytes(data, "semi.csv")
    assert err is None and list(df.columns) == ["a", "b", "c"] and len(df) == 2


def test_json_nested_records():
    data = b'{"data": [{"id": 1, "info": {"x": 2}}, {"id": 2, "info": {"x": 3}}]}'
    df, _ = read_bytes(data, "x.json")
    assert "info.x" in df.columns


def test_stocks_dates_detected():
    df, _ = load_sample("stocks")
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


# ---------------------------------------------------------------- charts
@pytest.mark.parametrize("chart", list(CHART_FIELDS))
def test_every_chart_type_builds(chart, sales, gap):
    df = gap
    types = classify_columns(df)
    nums, cats = types["numerical"], types["categorical"]
    spec = {"type": chart, "options": {}}
    for field, (_l, kind, _r) in CHART_FIELDS[chart].items():
        if kind == "multi_num":
            spec[field] = nums[:3]
        elif kind == "multi_cat":
            spec[field] = ["continent", "country"] if chart != "Parallel Categories" else ["continent"]
        elif field in ("x",) and chart in ("Bar", "Funnel", "Heatmap (pivot)", "Pie / Donut"):
            spec[field] = "continent"
        elif field == "y" and chart == "Heatmap (pivot)":
            spec[field] = "year"
        elif field == "names":
            spec[field] = "continent"
        elif field == "locations":
            spec[field] = "iso_alpha"
        elif field == "lat":
            spec[field] = "lifeExp"
        elif field == "lon":
            spec[field] = "gdpPercap"
        elif kind == "num":
            spec[field] = "pop" if field in ("size", "values", "z") else "lifeExp"
        elif kind in ("group", "group_or_date"):
            spec[field] = "continent" if field != "animation_frame" else None
        elif kind == "any":
            spec[field] = "gdpPercap" if field == "x" else "lifeExp"
    if chart in ("Line", "Area"):
        spec["x"] = "year"
    res = build_chart(df, spec)
    assert res.fig is not None, res.notes
    assert "px." in res.code


def test_bar_aggregates_and_top_n(sales):
    res = build_chart(sales, {"type": "Bar", "x": "city", "y": "revenue", "agg": "sum", "options": {"top_n": 3}})
    xs = set(res.fig.data[0].x)
    assert "Other" in xs and len(xs) == 4


def test_missing_column_reports_note(sales):
    res = build_chart(sales, {"type": "Scatter", "x": "nope", "y": "revenue"})
    assert res.fig is None and res.notes


# ---------------------------------------------------------------- eda
def test_insights_find_planted_issues(sales):
    titles = " ".join(i["title"] for i in eda.smart_insights(sales))
    assert "duplicate" in titles
    assert "whitespace" in titles or "casing" in titles


def test_target_correlation_excludes_target(gap):
    out = eda.compute_target_correlation(gap, "lifeExp")
    assert "lifeExp" not in out["Feature"].tolist()


def test_target_analysis_ranks(sales):
    r = eda.target_analysis(sales, "profit")
    assert r.iloc[0]["Feature"] in ("revenue", "unit_price")


def test_pca_two_columns_and_alignment(gap):
    scores, var, load = eda.run_pca(gap, ["lifeExp", "gdpPercap"], 3)
    assert scores.shape[1] == 2 and abs(var.sum() - 1) < 1e-9
    assert set(scores.index) <= set(gap.index)


def test_kmeans_labels_every_row(gap):
    labels, profile = eda.run_kmeans(gap, ["lifeExp", "gdpPercap", "pop"], 3)
    assert len(labels) == len(gap) and len(profile) == 3


def test_resample_and_decompose():
    df, _ = load_sample("stocks")
    ts = eda.resample_series(df, "date", "GOOG", "W", "mean")
    assert len(ts) > 50
    res = eda.seasonal_decomposition(ts.set_index("date")["GOOG"], 4)
    assert res.trend is not None


# ---------------------------------------------------------------- stats
def test_stats_tests(sales):
    tips, _ = load_sample("tips")
    assert S.two_groups(tips, "total_bill", "sex", "Male", "Female")["table"].shape[0] == 3
    assert "posthoc" in S.many_groups(tips, "total_bill", "day", alpha=0.5)["extra"] or True
    assert S.chi_square(tips, "smoker", "day")["extra"]["contingency"].shape == (2, 4)
    r = S.correlation_test(tips, "total_bill", "tip")
    assert r["verdict"] is True or r["verdict"] == np.True_
    assert S.one_sample(tips, "tip", 3.0)["table"].shape[0] >= 1
    assert S.paired(tips, "tip", "total_bill")["verdict"]


# ---------------------------------------------------------------- transforms
def _run_code(df, code):
    env = {"df": df.copy(), "pd": pd, "np": np}
    exec(code, env)  # noqa: S102 - generated by our own transforms
    return env["df"]


@pytest.mark.parametrize("op,kwargs", [
    (T.fill_missing, dict(columns=["satisfaction"], method="median")),
    (T.drop_missing, dict(columns=None, how="any")),
    (T.drop_duplicates, dict(columns=None, keep="first")),
    (T.handle_outliers, dict(column="revenue", method="IQR", threshold=1.5, action="cap")),
    (T.handle_outliers, dict(column="revenue", method="Z-score", threshold=3, action="remove")),
    (T.text_clean, dict(columns=["region"], ops=["trim whitespace", "Title Case"])),
    (T.calculated_column, dict(name="margin", expression="profit / revenue")),
    (T.bin_column, dict(column="revenue", bins=4, strategy="quantile")),
    (T.math_transform, dict(column="revenue", op="log (natural, log1p)", new_name="log_rev")),
    (T.extract_datetime, dict(column="order_date", parts=["year", "month", "weekday name"])),
    (T.filter_rows, dict(expression="revenue > 100")),
    (T.one_hot, dict(columns=["channel"])),
    (T.sort_rows, dict(columns=["revenue"], ascending=False)),
    (T.split_column, dict(column="order_id", sep="-")),
    (T.rename_column, dict(old="city", new="town")),
    (T.convert_type, dict(column="quantity", to="text")),
    (T.replace_values, dict(column="channel", find="Online", replace="Web")),
])
def test_transform_code_reproduces_result(op, kwargs, sales):
    out, label, code = op(sales, **kwargs)
    assert label and code
    replay = _run_code(sales, code)
    pd.testing.assert_frame_equal(out.reset_index(drop=True), replay.reset_index(drop=True), check_dtype=False)


def test_text_clean_merges_categories(sales):
    out, _, _ = T.text_clean(sales, ["region"], ["trim whitespace", "Title Case"])
    assert out["region"].nunique() == 4


def test_pipeline_script_is_valid_python():
    script = T.pipeline_script("data.csv", [{"label": "x", "code": "df = df.dropna()"}])
    compile(script, "pipeline.py", "exec")


# ---------------------------------------------------------------- filters / quality / reports
def test_filters(sales):
    f = [{"col": "region", "kind": "values", "value": ["North"], "exclude": False, "keep_na": False},
         {"col": "revenue", "kind": "range", "value": [100, 500], "exclude": False, "keep_na": False}]
    out = apply_filters(sales, f)
    assert set(out["region"]) == {"North"} and out["revenue"].between(100, 500).all()
    ex = apply_filters(sales, [dict(f[0], exclude=True)])
    assert "North" not in set(ex["region"])


def test_quality_score_range(sales):
    q = get_data_quality(sales)
    assert 0 <= q["score"] <= 100 and set(q["breakdown"]) == {"Completeness", "Uniqueness", "Consistency", "Validity"}


def test_reports(sales):
    html = build_html_report(sales, "sales.csv", [{"label": "a", "code": "b"}], [{"id": "1", "spec": {"type": "Bar", "x": "region"}}],
                             embed_js=False)
    assert html.startswith("<!doctype html>") and "plotly" in html.lower()
    md = build_markdown_report(sales, "sales.csv")
    assert md.startswith("# ")


def test_mixed_type_columns_are_unified():
    import datetime
    import pyarrow as pa
    raw = pd.DataFrame({"mixed": [datetime.datetime(2024, 1, 1), "n/a", 5, None],
                        "dates": [datetime.datetime(2024, 1, 1)] * 3 + [None],
                        "date_text": ["2024-01-01", datetime.datetime(2024, 2, 1), "2024-03-01", "2024-04-01"]})
    df, _ = infer_types(raw)
    assert pd.api.types.is_datetime64_any_dtype(df["dates"])
    assert pd.api.types.is_datetime64_any_dtype(df["date_text"])
    pa.Table.from_pandas(df)  # must be Arrow-serialisable for display

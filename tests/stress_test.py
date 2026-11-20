"""
Performance benchmark: times the heavy engine operations on a large synthetic dataset.

Run:  python tests/stress_test.py [rows]
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src import eda  # noqa: E402
from src.preprocess import classify_columns, get_data_quality  # noqa: E402
from src.visuals import MAX_POINTS, build_chart, sample_data  # noqa: E402


def generate_large_dataset(rows=100_000):
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "id": range(rows),
        "category": rng.choice(["A", "B", "C", "D"], rows),
        "value1": rng.standard_normal(rows),
        "value2": rng.random(rows) * 100,
        "value3": rng.integers(0, 1000, rows),
        "date": pd.date_range("2020-01-01", periods=rows, freq="min"),
    })


def timed(label, fn):
    t0 = time.perf_counter()
    out = fn()
    print(f"  {time.perf_counter() - t0:8.3f}s  {label}")
    return out


def run_stress_test(rows=100_000):
    print(f"Benchmark on {rows:,} rows\n")
    t_all = time.perf_counter()
    df = timed("generate data", lambda: generate_large_dataset(rows))
    sampled = timed("visual sampling", lambda: sample_data(df))
    assert len(sampled) <= MAX_POINTS
    types = timed("classify columns", lambda: classify_columns(df))
    timed("quality score", lambda: get_data_quality(df, types))
    timed("smart insights", lambda: eda.smart_insights(df))
    timed("correlation matrix", lambda: eda.correlation_matrix(df, types["numerical"]))
    timed("PCA", lambda: eda.run_pca(df, types["numerical"], 3))
    timed("target analysis", lambda: eda.target_analysis(df, "value2"))
    timed("k-means (k=4)", lambda: eda.run_kmeans(df, ["value1", "value2", "value3"], 4))
    timed("weekly resample", lambda: eda.resample_series(df, "date", "value2", "W", "mean"))
    timed("bar chart (aggregated)", lambda: build_chart(df, {"type": "Bar", "x": "category", "y": "value2", "agg": "mean"}))
    timed("scatter chart (sampled)", lambda: build_chart(df, {"type": "Scatter", "x": "value1", "y": "value2", "color": "category"}))
    print(f"\nTotal: {time.perf_counter() - t_all:.2f}s")


if __name__ == "__main__":
    run_stress_test(int(sys.argv[1]) if len(sys.argv) > 1 else 100_000)

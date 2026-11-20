"""
Headless UI smoke tests: render every page against every sample dataset and
assert that no exception is raised.

Run:  python -m pytest tests -q
"""
from __future__ import annotations

import os
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from streamlit.testing.v1 import AppTest  # noqa: E402

from src.loader import SAMPLE_DATASETS, load_sample  # noqa: E402

PAGES = ["Overview", "Profiler", "Clean & Transform", "Explore", "Stats Lab", "Pivot & Group",
         "SQL Lab", "Chart Studio", "Dashboard", "Export & Report"]


def _app_with(dataset: str, page: str) -> AppTest:
    os.chdir(ROOT)
    df, _ = load_sample(dataset)
    at = AppTest.from_file(os.path.join(ROOT, "main.py"), default_timeout=120)
    at.session_state["raw_df"] = df
    at.session_state["df"] = df
    at.session_state["file_name"] = f"{dataset}.csv"
    at.session_state["nav"] = page
    return at


def test_landing_page():
    os.chdir(ROOT)
    at = AppTest.from_file(os.path.join(ROOT, "main.py"), default_timeout=60).run()
    assert not at.exception, at.exception


@pytest.mark.parametrize("dataset", list(SAMPLE_DATASETS))
@pytest.mark.parametrize("page", PAGES)
def test_page_renders(dataset, page):
    at = _app_with(dataset, page).run()
    assert not at.exception, [e.value for e in at.exception]
    assert not at.error, [e.value for e in at.error]

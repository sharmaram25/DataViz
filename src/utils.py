"""Backward-compatible helpers (kept for older imports)."""
from __future__ import annotations

import pandas as pd


def convert_df_to_csv(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def generate_eda_report(df, eda_analyzer=None, preprocessor=None, name: str = "dataset") -> str:
    from src.report import build_markdown_report

    return build_markdown_report(df, name)

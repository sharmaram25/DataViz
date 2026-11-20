"""Landing page: upload / URL / sample datasets."""
from __future__ import annotations

import streamlit as st

from src import state
from src.loader import SAMPLE_DATASETS, sample_name, SUPPORTED_EXTENSIONS, fetch_url, list_excel_sheets, load_bytes, load_sample
from src.ui_components import html_block

FEATURES = [
    ("🧹", "Clean & Transform", "19 no-code operations with undo, and an exportable pandas pipeline."),
    ("🧭", "Deep Exploration", "Quality diagnostics, distributions, correlations, drivers, outliers, time series."),
    ("🧪", "Stats Lab", "t-tests, ANOVA, χ², correlation tests — with plain-English verdicts."),
    ("🧬", "PCA & Clustering", "Project high-dimensional data and segment it with K-Means."),
    ("🎨", "Chart Studio", "24 chart types, smart aggregation, Python code export."),
    ("📌", "Live Dashboard", "Pin charts; they update with your global filters."),
    ("🗄️", "SQL Lab", "Query your data with DuckDB SQL right in the browser."),
    ("📤", "Reports", "One-click HTML / Markdown reports and multi-format export."),
]


def _finish_load(df, notes, err, name):
    if err:
        st.error(err, icon="🚫")
        return
    state.set_dataset(df, name, notes)
    st.rerun()


def render():
    html_block("""
    <div class="dv-hero">
      <div class="dv-hero-eyebrow">✦ Analytics Studio · v3</div>
      <div class="dv-hero-title">Turn raw data into <span>clear answers.</span></div>
      <p>Load a dataset and DataViz profiles, cleans, tests and visualises it — from first look to
         a shareable report in minutes. Everything runs locally on your machine.</p>
      <div class="dv-hero-stats">
        <div class="dv-hero-stat"><b>24</b><span>chart types</span></div>
        <div class="dv-hero-stat"><b>19</b><span>cleaning operations</span></div>
        <div class="dv-hero-stat"><b>12+</b><span>statistical tests</span></div>
        <div class="dv-hero-stat"><b>9</b><span>file formats</span></div>
      </div>
    </div>""")

    left, right = st.columns([1.35, 1], gap="large")
    with left:
        with st.container(border=True):
            st.markdown("#### 📂 Load your data")
            tab_file, tab_url = st.tabs(["Upload a file", "From a URL"])
            with tab_file:
                up = st.file_uploader(
                    "Drop a file here", type=SUPPORTED_EXTENSIONS, label_visibility="collapsed",
                    help="CSV, TSV, TXT, Excel, JSON / JSON Lines, Parquet — up to 500 MB",
                )
                with st.expander("⚙️ Import options"):
                    c1, c2 = st.columns(2)
                    sep_label = c1.selectbox("Delimiter", ["Auto-detect", "Comma ,", "Semicolon ;", "Tab", "Pipe |"])
                    enc_label = c2.selectbox("Encoding", ["Auto-detect", "utf-8", "cp1252", "latin-1"])
                    auto_types = st.toggle("Auto-detect numbers & dates stored as text", value=True)
                sep = {"Auto-detect": None, "Comma ,": ",", "Semicolon ;": ";", "Tab": "\t", "Pipe |": "|"}[sep_label]
                enc = None if enc_label == "Auto-detect" else enc_label
                if up is not None:
                    data = up.getvalue()
                    sheet = None
                    if up.name.lower().endswith((".xlsx", ".xlsm", ".xls")):
                        sheets = list_excel_sheets(data)
                        if len(sheets) > 1:
                            sheet = st.selectbox("Worksheet", sheets)
                    if st.button("🚀 Analyse this file", type="primary", width="stretch"):
                        with st.spinner("Reading and profiling…"):
                            df, notes, err = load_bytes(data, up.name, sheet=sheet, sep=sep, encoding=enc, auto_types=auto_types)
                        _finish_load(df, notes, err, up.name)
            with tab_url:
                url = st.text_input("File URL", placeholder="https://example.com/data.csv  ·  GitHub file links work too")
                if st.button("⬇️ Fetch & analyse", type="primary", disabled=not url, width="stretch"):
                    try:
                        with st.spinner("Downloading…"):
                            data, name = fetch_url(url)
                            df, notes, err = load_bytes(data, name)
                        _finish_load(df, notes, err, name)
                    except Exception as e:
                        st.error(f"Could not fetch the URL: {e}", icon="🌐")

    with right:
        with st.container(border=True):
            st.markdown("#### ✨ Or try a sample dataset")
            for key, (title, desc) in SAMPLE_DATASETS.items():
                c1, c2 = st.columns([3.2, 1], vertical_alignment="center")
                c1.markdown(f"**{title}**  \n<span style='color:#52514e;font-size:.84rem'>{desc}</span>", unsafe_allow_html=True)
                if c2.button("Open", key=f"sample_{key}", width="stretch"):
                    with st.spinner("Loading sample…"):
                        df, notes = load_sample(key)
                    _finish_load(df, notes, None, sample_name(key))

    st.markdown("#### What you can do")
    html_block('<div class="dv-feature-grid">' + "".join(
        f'<div class="dv-feature"><div class="dv-feature-icon">{i}</div>'
        f'<div class="dv-feature-title">{t}</div><div class="dv-feature-text">{d}</div></div>'
        for i, t, d in FEATURES) + "</div>")

"""SQL Lab: query the dataset with DuckDB (falls back to pandas query syntax)."""
from __future__ import annotations

import time

import streamlit as st

from src import state
from src.ui_components import page_header, result_card, section, show_chart, show_df
from src.visuals import build_chart

try:
    import duckdb
    HAS_DUCKDB = True
except ImportError:  # pragma: no cover
    HAS_DUCKDB = False


def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _examples(types):
    nums, groups = types["numerical"], types["groupable"]
    ex = {"Preview": "SELECT *\nFROM data\nLIMIT 100"}
    if groups and nums:
        g, n = _q(groups[0]), _q(nums[0])
        ex["Group & aggregate"] = (f"SELECT {g},\n       COUNT(*) AS rows,\n       ROUND(AVG({n}), 2) AS avg_{nums[0]},\n"
                                   f"       ROUND(SUM({n}), 2) AS total_{nums[0]}\nFROM data\nGROUP BY {g}\nORDER BY rows DESC")
        ex["Top 10"] = f"SELECT *\nFROM data\nORDER BY {n} DESC\nLIMIT 10"
    if nums:
        n = _q(nums[0])
        ex["Percentiles"] = (f"SELECT\n  MIN({n}) AS min,\n  QUANTILE_CONT({n}, 0.25) AS q1,\n  MEDIAN({n}) AS median,\n"
                             f"  QUANTILE_CONT({n}, 0.75) AS q3,\n  MAX({n}) AS max\nFROM data")
    if types["datetime"] and nums:
        d, n = _q(types["datetime"][0]), _q(nums[0])
        ex["Monthly trend"] = (f"SELECT DATE_TRUNC('month', {d}) AS month,\n       SUM({n}) AS total\n"
                               f"FROM data\nGROUP BY 1\nORDER BY 1")
    if groups and nums:
        g, n = _q(groups[0]), _q(nums[0])
        ex["Window: rank within group"] = (f"SELECT *,\n  RANK() OVER (PARTITION BY {g} ORDER BY {n} DESC) AS rank_in_group\n"
                                           f"FROM data\nQUALIFY rank_in_group <= 3")
    return ex


def render(ctx):
    df, types = ctx.df, ctx.types
    page_header("🗄️", "SQL Lab", "Query your data with SQL — the table is called <code>data</code>")
    if not HAS_DUCKDB:
        st.warning("DuckDB is not installed (`pip install duckdb`). Using pandas `query` expressions instead.")
        ex = st.text_input("Filter expression", placeholder="revenue > 100 and region == 'North'")
        if ex:
            try:
                show_df(df.query(ex, engine="python"))
            except Exception as e:
                st.error(str(e))
        return

    examples = _examples(types)
    c1, c2 = st.columns([3, 1])
    with c2:
        st.markdown("**Examples**")
        for name, sql in examples.items():
            if st.button(name, key=f"sql_ex_{name}", width="stretch"):
                st.session_state["p_sql_text"] = sql
        with st.expander("Columns"):
            st.code("\n".join(f"{_q(c)}  -- {df[c].dtype}" for c in df.columns), language="sql")
    with c1:
        if "p_sql_text" not in st.session_state:
            st.session_state["p_sql_text"] = list(examples.values())[min(1, len(examples) - 1)]
        sql = st.text_area("SQL", key="p_sql_text", height=230, label_visibility="collapsed")
        run = st.button("▶ Run query", type="primary", key="sql_run", shortcut="ctrl+enter")
        st.caption("Ctrl + Enter to run · DuckDB SQL dialect · read-only (your data isn't modified)")

    if run or st.session_state.get("sql_last") == sql:
        try:
            t0 = time.perf_counter()
            con = duckdb.connect()
            con.register("data", df)
            result = con.execute(sql).df()
            con.close()
            ms = (time.perf_counter() - t0) * 1000
            st.session_state["sql_last"] = sql
        except Exception as e:
            result_card("crit", "Query failed", str(e).replace("<", "&lt;"))
            return
        section("Result", f"{len(result):,} rows × {result.shape[1]} columns · {ms:,.0f} ms")
        show_df(result.head(5000), height=420)
        c1, c2, c3 = st.columns(3)
        c1.download_button("⬇️ Download CSV", result.to_csv(index=False).encode("utf-8"), "query_result.csv", "text/csv",
                           width="stretch", key="sql_dl")
        if c2.button("📥 Use result as working dataset", width="stretch", key="sql_use",
                     help="Replaces the working dataset (undoable)."):
            state.apply_step(result, "Replaced data with SQL query result",
                             f"import duckdb\ndf = duckdb.query_df(df, 'data', {sql!r}).df()")
            st.toast("✅ Working dataset replaced (undo available)")
            st.rerun()
        quick = c3.toggle("Quick chart", value=False, key="sql_chart")
        if quick and result.shape[1] >= 2:
            x, y = result.columns[0], result.columns[1]
            kind = "Line" if str(result[x].dtype).startswith("datetime") else "Bar"
            if not str(result[y].dtype).startswith(("int", "float")):
                y = None
            show_chart(build_chart(result, {"type": kind, "x": x, "y": y, "agg": "sum",
                                            "options": {"sort": "None"}}).fig, key="sql_fig")

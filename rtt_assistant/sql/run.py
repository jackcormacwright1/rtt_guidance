"""Running checked SQL against the RTT database: read-only, no external access, time- and row-limited."""

from __future__ import annotations

import threading
from pathlib import Path

import duckdb
import pandas as pd

from .guard import check_sql


class QueryTimeout(RuntimeError):
    pass


class QueryError(RuntimeError):
    """DuckDB rejected the query (e.g. a misspelt column). The message is fed back to the model."""


def connect_readonly(db_path: str | Path) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect(str(db_path), read_only=True, config={"enable_external_access": False})
    try:
        con.execute("SET lock_configuration = true")  # queries can't switch external access back on
    except duckdb.InvalidInputException:
        pass  # another connection to the same file in this process has already locked it
    return con


def run_query(con: duckdb.DuckDBPyConnection, sql: str, allowed_tables: set[str],
              timeout_s: float = 10.0, max_rows: int = 500) -> pd.DataFrame:
    """Check ``sql``, run it on a cursor of ``con``, and return at most ``max_rows`` rows."""
    checked = check_sql(sql, allowed_tables)
    cur = con.cursor()
    result: dict = {}

    def work():
        try:
            rel = cur.execute(checked)
            cols = [d[0] for d in rel.description]
            rows = rel.fetchmany(max_rows + 1)
            result["df"] = pd.DataFrame(rows, columns=cols)
        except Exception as e:  # noqa: BLE001 - reported back as QueryError
            result["error"] = e

    t = threading.Thread(target=work, daemon=True)
    t.start()
    t.join(timeout_s)
    if t.is_alive():
        cur.interrupt()
        t.join(2)
        raise QueryTimeout(f"The query took longer than {timeout_s:g} seconds and was stopped.")
    if "error" in result:
        raise QueryError(str(result["error"]).split("\n")[0][:500])
    df = result["df"]
    df.attrs["truncated"] = len(df) > max_rows
    return df.head(max_rows)

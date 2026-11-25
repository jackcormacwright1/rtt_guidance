"""Checks that a model-written SQL query is safe to run.

The model's SQL is treated as untrusted input. Three layers stand between it and the data:

1. This module parses the query (sqlglot, DuckDB dialect) and rejects anything that is not a
   single read-only SELECT over the known tables: no second statement, no DDL/DML, no ATTACH,
   COPY, SET, PRAGMA or INSTALL, no table functions that read files or URLs, and no tables
   other than the ones in the schema (plus the query's own CTEs).
2. ``run.py`` opens the database read-only, with DuckDB's external access switched off and its
   configuration locked, so even a query that slipped past (1) can't read files or the network.
3. ``run.py`` also stops long-running queries and caps the rows returned.
"""

from __future__ import annotations

import sqlglot
from sqlglot import exp

# Functions that read files, URLs or the environment. External access is also disabled on the
# connection; this list makes the rejection explicit and the error message useful.
FORBIDDEN_FUNCTIONS = {
    "read_csv", "read_csv_auto", "read_parquet", "parquet_scan", "read_json", "read_json_auto",
    "read_json_objects", "read_ndjson", "read_text", "read_blob", "glob", "getenv", "sniff_csv",
    "query", "query_table", "sqlite_scan", "postgres_scan", "mysql_scan", "iceberg_scan", "delta_scan",
}
READ_ONLY_ROOTS = (exp.Select, exp.Union, exp.Intersect, exp.Except)


class UnsafeSQL(ValueError):
    """The query was rejected before reaching the database."""


def check_sql(sql: str, allowed_tables: set[str]) -> str:
    """Return the query (trailing semicolons removed) if it passes, else raise ``UnsafeSQL``."""
    text = (sql or "").strip().rstrip(";").strip()
    if not text:
        raise UnsafeSQL("The query is empty.")
    try:
        statements = [s for s in sqlglot.parse(text, read="duckdb") if s is not None]
    except sqlglot.errors.ParseError as e:
        raise UnsafeSQL(f"The query could not be parsed: {e}") from e
    if len(statements) != 1:
        raise UnsafeSQL("Only a single statement is allowed.")
    stmt = statements[0]
    if not isinstance(stmt, READ_ONLY_ROOTS):
        raise UnsafeSQL(f"Only SELECT queries are allowed (got {type(stmt).__name__.upper()}).")

    allowed = {t.lower() for t in allowed_tables}
    cte_names = {cte.alias_or_name.lower() for cte in stmt.find_all(exp.CTE)}
    for table in stmt.find_all(exp.Table):
        name = table.name.lower()
        if not name or not isinstance(table.this, exp.Identifier):
            raise UnsafeSQL("Table functions (such as reading files) are not allowed.")
        if table.args.get("db") or table.args.get("catalog"):
            raise UnsafeSQL("Qualified table names (schema.table) are not allowed.")
        if name not in allowed and name not in cte_names:
            raise UnsafeSQL(f"Unknown table '{table.name}'. Allowed: {', '.join(sorted(allowed))}.")

    for node in stmt.find_all(exp.Func):
        fname = (node.sql_name() if not isinstance(node, exp.Anonymous) else str(node.this)).lower()
        if fname in FORBIDDEN_FUNCTIONS:
            raise UnsafeSQL(f"The function {fname}() is not allowed.")
    for node in stmt.find_all(exp.Anonymous):
        if str(node.this).lower() in FORBIDDEN_FUNCTIONS:
            raise UnsafeSQL(f"The function {node.this}() is not allowed.")
    return text

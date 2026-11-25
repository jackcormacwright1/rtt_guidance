import duckdb
import pandas as pd
import pytest

from rtt_assistant.sql.guard import UnsafeSQL, check_sql
from rtt_assistant.sql.run import QueryError, connect_readonly, run_query

ALLOWED = {"waiting_list", "new_periods"}


@pytest.mark.parametrize("sql", [
    "SELECT * FROM waiting_list",
    "select count(*) from waiting_list;",
    "WITH t AS (SELECT period, SUM(total_waiting) AS n FROM waiting_list GROUP BY 1) SELECT * FROM t ORDER BY 1",
    "SELECT period FROM waiting_list UNION ALL SELECT period FROM new_periods",
    "SELECT w.period FROM waiting_list w JOIN new_periods n USING (period)",
])
def test_reasonable_queries_pass(sql):
    assert check_sql(sql, ALLOWED)


@pytest.mark.parametrize("sql, why", [
    ("SELECT 1; DROP TABLE waiting_list", "single statement"),
    ("DROP TABLE waiting_list", "Only SELECT"),
    ("DELETE FROM waiting_list", "Only SELECT"),
    ("INSERT INTO waiting_list VALUES (1)", "Only SELECT"),
    ("ATTACH 'other.db'", "Only SELECT"),
    ("COPY waiting_list TO 'out.csv'", "Only SELECT"),
    ("SET enable_external_access = true", "Only SELECT"),
    ("PRAGMA database_list", "Only SELECT"),
    ("INSTALL httpfs", "Only SELECT"),
    ("SELECT * FROM read_csv('/etc/passwd')", "Table functions"),
    ("SELECT * FROM '/etc/passwd'", "Unknown table"),
    ("SELECT * FROM secret_table", "Unknown table"),
    ("SELECT * FROM main.waiting_list", "Qualified"),
    ("SELECT getenv('HOME')", "getenv"),
    ("", "empty"),
])
def test_unsafe_queries_are_rejected(sql, why):
    with pytest.raises(UnsafeSQL, match=why):
        check_sql(sql, ALLOWED)


@pytest.fixture()
def db(tmp_path):
    path = tmp_path / "t.duckdb"
    con = duckdb.connect(str(path))
    con.execute("CREATE TABLE waiting_list AS SELECT range AS period, range * 10 AS total_waiting FROM range(1000)")
    con.close()
    return connect_readonly(path)


def test_runs_read_only_and_caps_rows(db):
    df = run_query(db, "SELECT * FROM waiting_list ORDER BY period", {"waiting_list"}, max_rows=50)
    assert len(df) == 50 and df.attrs["truncated"]
    with pytest.raises(Exception):
        db.execute("CREATE TABLE x (a INT)")  # connection itself is read-only


def test_database_errors_come_back_as_query_errors(db):
    with pytest.raises(QueryError, match="no_such_column"):
        run_query(db, "SELECT no_such_column FROM waiting_list", {"waiting_list"})


def test_external_access_is_off_even_if_the_guard_were_bypassed(db):
    with pytest.raises(Exception):
        db.execute("SELECT * FROM read_csv('/etc/hostname')").fetchall()
    with pytest.raises(Exception):
        db.execute("SET enable_external_access = true")

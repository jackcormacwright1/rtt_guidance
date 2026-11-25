"""Answer a question about the RTT data: question -> SQL -> checked query -> result -> summary.

    1. The model writes SQL from the schema, facts about the loaded months and a few examples.
    2. The SQL is checked (sql/guard.py) and run read-only (sql/run.py).
    3. If the check or the database rejects it, the error goes back to the model for one more attempt.
    4. The model summarises the result for a non-technical reader.
    5. Every number in the summary is checked against the result; unmatched numbers are flagged.
    6. If the result covers months in which a provider is missing from the data, a caveat is attached.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date

import duckdb
import pandas as pd

from .data.schema import TABLE_NAMES
from .llm import ChatModel, LLMError
from .sql import generate, summarise
from .sql.guard import UnsafeSQL
from .sql.run import QueryError, QueryTimeout, run_query


def coverage_gaps(con: duckdb.DuckDBPyConnection) -> dict[str, set[date]]:
    """Provider name -> the months it is missing from the data (from the coverage table)."""
    gaps: dict[str, set[date]] = {}
    for name, missing in con.execute(
            "SELECT provider_name, missing_months FROM coverage WHERE missing_months <> ''").fetchall():
        months: set[date] = set()
        for part in missing.split(";"):
            ends = [p.strip() for p in part.split(" to ")]
            a = date(int(ends[0][:4]), int(ends[0][5:7]), 1)
            b = date(int(ends[-1][:4]), int(ends[-1][5:7]), 1)
            while a <= b:
                months.add(a)
                a = date(a.year + (a.month == 12), a.month % 12 + 1, 1)
        gaps[name] = months
    return gaps


@dataclass
class DataAnswer:
    question: str
    answer: str = ""
    sql: str | None = None
    result: pd.DataFrame | None = None
    assumptions: str = ""
    cannot_answer: bool = False
    ungrounded: list[str] = field(default_factory=list)
    caveat: str = ""
    attempts: list[dict] = field(default_factory=list)  # each: sql, error
    error: str | None = None
    llm_calls: int = 0


def answer_question(question: str, llm: ChatModel, con: duckdb.DuckDBPyConnection, facts: str,
                    max_attempts: int = 2, summarise_result: bool = True) -> DataAnswer:
    out = DataAnswer(question=question)
    previous: list[tuple[str, str]] = []
    for _ in range(max_attempts):
        try:
            reply = llm.complete(generate.SYSTEM, generate.build_prompt(question, facts, previous), json_mode=True)
            out.llm_calls += 1
            draft = generate.parse_reply(reply)
        except (LLMError, ValueError) as e:
            out.error = f"Couldn't get a query from the model: {e}"
            return out
        if draft.cannot_answer:
            out.cannot_answer = True
            out.answer = f"I can't answer that from the RTT data. {draft.reason}".strip()
            return out
        out.sql, out.assumptions = draft.sql, draft.assumptions
        try:
            out.result = run_query(con, draft.sql, TABLE_NAMES)
            out.attempts.append({"sql": draft.sql, "error": None})
            break
        except (UnsafeSQL, QueryError, QueryTimeout) as e:
            out.attempts.append({"sql": draft.sql, "error": str(e)})
            previous.append((draft.sql, str(e)))
    else:
        out.error = f"The query failed after {max_attempts} attempts: {out.attempts[-1]['error']}"
        return out

    out.caveat = summarise.coverage_warning(out.result, out.sql, coverage_gaps(con))
    if not summarise_result:
        return out
    try:
        out.answer = llm.complete(summarise.SYSTEM, summarise.build_prompt(
            question, out.sql, out.result, out.assumptions, out.caveat)).strip()
        out.llm_calls += 1
    except LLMError as e:
        out.error = f"The query ran, but the summary failed: {e}"
        return out
    out.ungrounded = summarise.ungrounded_numbers(out.answer, out.result)
    return out

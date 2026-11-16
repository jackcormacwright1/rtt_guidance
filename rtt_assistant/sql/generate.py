"""Natural-language question -> DuckDB SQL, written by the model from the schema and a few examples."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import duckdb

from ..data.schema import describe_for_prompt

SYSTEM = (
    "You write DuckDB SQL that answers questions about NHS England consultant-led Referral to Treatment (RTT) "
    "waiting times statistics. You can only use the tables described. You always reply with a single JSON object."
)

RULES = """\
Rules:
1. Reply with JSON: {"sql": "<one DuckDB SELECT>", "assumptions": "<anything you assumed, or empty>", "cannot_answer": false}
   If the tables can't answer the question (for example it is about cancer, A&E, diagnostics, individual patients,
   or months that aren't in the data), reply {"sql": null, "cannot_answer": true, "reason": "<short reason>"}.
2. One SELECT statement only (CTEs are fine). Use only the tables and columns listed.
3. England or national figures are sums over all providers. Compute percentages from sums, never by averaging
   providers' percentages: ROUND(100.0 * SUM(within_18_weeks) / SUM(total_waiting), 1).
4. "Latest month" means (SELECT MAX(period) FROM <table>). "Over the last N months" means compare the latest month
   with the month N months earlier; return both months' values and the change in one or two rows.
5. Always include the period (month) in the output, so the answer can say which months it refers to.
6. Match providers with provider_name ILIKE '%<words>%' and specialties with treatment_function_name ILIKE '%<word>%'.
   Use the treatment function names listed below to choose the right search word.
7. Return at most 50 rows, ordered sensibly. Use clear snake_case column aliases.
"""

EXAMPLES = """\
Examples:

Q: How has the percentage of patients waiting under 18 weeks in England changed over the last three months?
{"sql": "WITH m AS (SELECT period, SUM(within_18_weeks) AS within_18, SUM(total_waiting) AS waiting FROM waiting_list GROUP BY period), latest AS (SELECT MAX(period) AS p FROM m) SELECT period, waiting AS total_waiting, ROUND(100.0 * within_18 / waiting, 1) AS pct_within_18_weeks FROM m, latest WHERE period >= p - INTERVAL 3 MONTH ORDER BY period", "assumptions": "Read 'over the last three months' as the latest month and the three months before it.", "cannot_answer": false}

Q: Which five trusts had the most patients waiting over 52 weeks in the latest month?
{"sql": "SELECT period, provider_name, SUM(over_52_weeks) AS over_52_weeks FROM waiting_list WHERE period = (SELECT MAX(period) FROM waiting_list) GROUP BY period, provider_name ORDER BY over_52_weeks DESC LIMIT 5", "assumptions": "", "cannot_answer": false}

Q: Is demand for trauma and orthopaedics rising?
{"sql": "SELECT period, SUM(new_clock_starts) AS new_clock_starts FROM new_periods WHERE treatment_function_name ILIKE '%orthopaedic%' GROUP BY period ORDER BY period", "assumptions": "Used new RTT clock starts as the measure of demand.", "cannot_answer": false}

Q: What is the 62-day cancer performance?
{"sql": null, "cannot_answer": true, "reason": "The database holds RTT waiting times only; cancer waiting times are a separate dataset."}
"""


@dataclass
class SQLDraft:
    sql: str | None
    assumptions: str = ""
    cannot_answer: bool = False
    reason: str = ""


def database_facts(con: duckdb.DuckDBPyConnection) -> str:
    """Facts about the loaded data that help the model write correct filters."""
    periods = [str(r[0])[:7] for r in con.execute("SELECT DISTINCT period FROM waiting_list ORDER BY 1").fetchall()]
    tfs = [r[0] for r in con.execute(
        "SELECT DISTINCT treatment_function_name FROM waiting_list ORDER BY 1").fetchall()]
    n_prov = con.execute("SELECT COUNT(DISTINCT provider_code) FROM waiting_list").fetchone()[0]
    return (f"Months available: {', '.join(periods)} (latest {periods[-1] if periods else 'none'}).\n"
            f"Providers: {n_prov}.\n"
            f"Treatment function names: {'; '.join(tfs)}.")


def build_prompt(question: str, facts: str, previous: list[tuple[str, str]] | None = None) -> str:
    parts = [describe_for_prompt(), "", facts, "", RULES, EXAMPLES]
    for sql, error in previous or []:
        parts.append(f"Your previous query failed.\nQuery: {sql}\nError: {error}\nWrite a corrected query.\n")
    parts.append(f"Q: {question}")
    return "\n".join(parts)


def parse_reply(text: str) -> SQLDraft:
    """Parse the model's JSON. Tolerates code fences and stray text around the object."""
    t = text.strip()
    t = re.sub(r"^```(?:json)?\s*|\s*```$", "", t)
    m = re.search(r"\{.*\}", t, re.S)
    if not m:
        raise ValueError("The model's reply wasn't JSON.")
    obj = json.loads(m.group(0))
    if obj.get("cannot_answer"):
        return SQLDraft(sql=None, cannot_answer=True, reason=str(obj.get("reason", "")))
    sql = obj.get("sql")
    if not sql or not isinstance(sql, str):
        raise ValueError("The model's reply had no SQL.")
    return SQLDraft(sql=sql.strip(), assumptions=str(obj.get("assumptions", "") or ""))

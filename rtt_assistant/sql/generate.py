"""Natural-language question -> DuckDB SQL, written by the model from the schema and a few examples."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass

import duckdb

from ..data.schema import describe_for_prompt

SYSTEM = (
    "You write DuckDB SQL that answers questions about NHS England Referral to Treatment (RTT) waiting-times "
    "statistics for London providers, April 2018 to April 2023. You can only use the tables described. "
    "You always reply with a single JSON object."
)

RULES = """\
Rules:
1. Reply with JSON: {"sql": "<one DuckDB SELECT>", "assumptions": "<anything you assumed, or empty>", "cannot_answer": false}
   If the tables can't answer the question, reply {"sql": null, "cannot_answer": true, "reason": "<short reason>"}.
   Examples: cancer, A&E or diagnostic waits; individual patients or consultants; England or regions outside
   London; months outside the data; waits over 65, 78 or 104 weeks (the longest band is 'over 52 weeks');
   a single trust outside North West London other than Royal Free (they are only available as area groups).
2. One SELECT statement only (CTEs are fine). Use only the tables and columns listed.
3. London totals are sums over every provider_name. A provider's whole waiting list is the sum over both
   commissioner_group values and all treatment functions. Compute percentages from sums, never by averaging
   percentages: ROUND(100.0 * SUM(within_18_weeks) / SUM(total_waiting), 1).
4. "Latest month" means (SELECT MAX(period) FROM <table>). "Over the last N months" means compare the latest month
   with the month N months earlier; return both months' values and the change.
5. Always include the period (or financial_year) in the output, so the answer can say which months it refers to.
6. Match trusts with provider_name ILIKE '%<distinctive word>%' (e.g. '%imperial%', '%hillingdon%',
   '%north west university%') and specialties with treatment_function_name ILIKE '%<word>%'. "North West London
   trusts" means provider_area = 'North West London'. Use the names listed below to choose the search words.
7. Royal Free London is missing from February 2019 to February 2021 (see the coverage table), so London and
   North Central London totals for those months are too low. When a question compares totals across those months,
   still answer it, but say so in "assumptions".
8. Return at most 50 rows, ordered sensibly. Use clear snake_case column aliases.
"""

EXAMPLES = """\
Examples:

Q: How has the percentage of London patients waiting under 18 weeks changed over the last three months?
{"sql": "WITH m AS (SELECT period, SUM(within_18_weeks) AS within_18, SUM(total_waiting) AS waiting FROM waiting_list GROUP BY period), latest AS (SELECT MAX(period) AS p FROM m) SELECT period, waiting AS total_waiting, ROUND(100.0 * within_18 / waiting, 1) AS pct_within_18_weeks FROM m, latest WHERE period >= p - INTERVAL 3 MONTH ORDER BY period", "assumptions": "Read 'over the last three months' as the latest month and the three months before it.", "cannot_answer": false}

Q: Which North West London trust had the most patients waiting over 52 weeks in the latest month?
{"sql": "SELECT period, provider_name, SUM(over_52_weeks) AS over_52_weeks FROM waiting_list WHERE provider_area = 'North West London' AND period = (SELECT MAX(period) FROM waiting_list) GROUP BY period, provider_name ORDER BY over_52_weeks DESC", "assumptions": "", "cannot_answer": false}

Q: How many North West London residents were waiting at hospitals outside North West London in March 2022?
{"sql": "SELECT period, provider_area, SUM(total_waiting) AS total_waiting FROM waiting_list WHERE commissioner_group = 'North West London' AND provider_area <> 'North West London' AND period = DATE '2022-03-01' GROUP BY period, provider_area ORDER BY total_waiting DESC", "assumptions": "Counted patients whose care is commissioned in North West London.", "cannot_answer": false}

Q: Is demand for trauma and orthopaedics at Imperial rising?
{"sql": "SELECT financial_year, SUM(new_clock_starts) AS new_clock_starts, COUNT(DISTINCT period) AS months FROM new_periods WHERE provider_name ILIKE '%imperial%' AND treatment_function_name ILIKE '%orthopaedic%' GROUP BY financial_year ORDER BY financial_year", "assumptions": "Used new RTT clock starts per financial year as the measure of demand; 2023-24 has one month only.", "cannot_answer": false}

Q: How many patients are waiting over 104 weeks at Imperial?
{"sql": null, "cannot_answer": true, "reason": "The data only records waits up to 'over 52 weeks'; longer bands aren't available."}
"""


@dataclass
class SQLDraft:
    sql: str | None
    assumptions: str = ""
    cannot_answer: bool = False
    reason: str = ""


def database_facts(con: duckdb.DuckDBPyConnection) -> str:
    """Facts about the loaded data that help the model write correct filters."""
    first, last, n = con.execute("SELECT MIN(period), MAX(period), COUNT(DISTINCT period) FROM waiting_list").fetchone()
    providers = con.execute("SELECT provider_area, provider_name, missing_months FROM coverage ORDER BY 1, 2").fetchall()
    tfs = [r[0] for r in con.execute("SELECT DISTINCT treatment_function_name FROM waiting_list ORDER BY 1").fetchall()]
    lines = [f"Months available: {first:%Y-%m} to {last:%Y-%m} ({n} months; latest {last:%B %Y}).", "Providers:"]
    for area, name, missing in providers:
        lines.append(f"  - {name} ({area})" + (f": no data {missing}" if missing else ""))
    lines.append(f"Treatment function names: {'; '.join(tfs)}.")
    return "\n".join(lines)


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

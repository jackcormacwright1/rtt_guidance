"""Query result -> a short answer for an NHS manager, plus a check that its numbers come from the result."""

from __future__ import annotations

import itertools
import re
from datetime import date

import numpy as np
import pandas as pd

SYSTEM = (
    "You explain NHS waiting-time statistics to a busy NHS manager. You use only the numbers in the "
    "query result you are given, and you write in UK English."
)

DATA_NOTE = ("NHS England's published RTT statistics for London providers, April 2018 to April 2023, as submitted "
             "by trusts. Trusts outside North West London, apart from Royal Free London, are grouped by area.")


def coverage_warning(result: pd.DataFrame | None, sql: str | None, gaps: dict[str, set[date]]) -> str:
    """A caveat when the result covers months in which a provider is missing from the data.

    ``gaps`` maps provider name -> months with no data. The caveat is skipped when the query is limited to
    other providers (it filters on provider_name or provider_area without mentioning the missing one).
    """
    if result is None or not len(result) or not gaps or not sql:
        return ""
    months: set[date] = set()
    if "period" in result.columns:
        months |= {d.date().replace(day=1) for d in pd.to_datetime(result["period"], errors="coerce").dropna()}
    if "financial_year" in result.columns:
        for fy in result["financial_year"].dropna().astype(str):
            if len(fy) >= 4 and fy[:4].isdigit():
                y = int(fy[:4])
                months |= {date(y + (m < 4), m, 1) for m in range(1, 13)}
    s = sql.lower()
    notes = []
    for provider, missing in gaps.items():
        hit = sorted(months & missing)
        if not hit:
            continue
        words = [w for w in provider.lower().replace("nhs", "").split() if len(w) > 3][:2]
        mentions_it = any(w in s for w in words) or "north central" in s
        limited_elsewhere = ("provider_name" in s or "provider_area" in s) and not mentions_it
        if not limited_elsewhere:
            span = f"{hit[0]:%B %Y}" + (f" to {hit[-1]:%B %Y}" if len(hit) > 1 else "")
            notes.append(f"{provider} has no data for {span}, so totals that should include it "
                         "(London, North Central London, or the trust itself) are too low for those months.")
    return " ".join(notes)


def build_prompt(question: str, sql: str, result: pd.DataFrame, assumptions: str = "", caveat: str = "") -> str:
    table = result.head(60).to_csv(index=False)
    more = f"\n(Showing 60 of {len(result)} rows.)" if len(result) > 60 else ""
    if result.attrs.get("truncated"):
        more += "\n(The query returned more rows than shown.)"
    return f"""Question: {question}

SQL that was run:
{sql}

Assumptions made in writing the SQL: {assumptions or 'none'}

Result (CSV):
{table}{more}

Note on the data: {DATA_NOTE}
{("Coverage caveat (mention this in one sentence): " + caveat) if caveat else ""}

Write the answer:
- Two to four sentences. Lead with the direct answer.
- Say which month or months the figures are for (e.g. "April 2023").
- Use only numbers that appear in the result, or simple differences between them. Round sensibly
  (e.g. "1.16 million", "63.2%"). Describe changes in percentages in percentage points.
- If an assumption in the SQL matters to the answer, mention it briefly.
- Don't speculate about causes. If the result is empty, say that no matching data was found.
"""


NUMBER = re.compile(r"(?<![\w.])(-?\d[\d,]*\.?\d*)\s*(million|thousand|k\b|%)?(\s*-?\s*weeks?\b)?", re.I)
# Waiting-time thresholds and standards that answers mention without them being in the result.
STANDARD_WEEKS = {6, 18, 52, 65, 78, 104}
STANDARD_PERCENTAGES = {92.0}


def _numbers_in(text: str) -> list[tuple[str, float, int, str]]:
    """(as written, value, decimals, unit) for every number in ``text``."""
    out = []
    for m in NUMBER.finditer(text):
        raw, unit = m.group(1), (m.group(2) or "").lower()
        clean = raw.replace(",", "").rstrip(".")
        if not clean or clean in "-":
            continue
        val = float(clean)
        dec = len(clean.split(".")[1]) if "." in clean else 0
        if m.group(3) and dec == 0 and int(val) in STANDARD_WEEKS:
            continue  # "18 weeks", "52-week waits"
        if unit == "%" and val in STANDARD_PERCENTAGES:
            continue  # the 92% constitutional standard
        out.append((m.group(0).strip(), val, dec, unit))
    return out


def ungrounded_numbers(answer: str, result: pd.DataFrame) -> list[str]:
    """Numbers in ``answer`` that can't be matched to the result.

    A number counts as matched if, to the precision it is written, it equals a value in the result,
    that value in thousands or millions, or the difference between two values in the same column.
    Years and small counting words (e.g. 'five trusts', 'last 2 months') are ignored.
    """
    cols = [pd.to_numeric(result[c], errors="coerce").dropna().to_numpy(dtype=float)
            for c in result.columns]
    cols = [c for c in cols if len(c)]
    candidates = [np.concatenate(cols)] if cols else []
    for c in cols:
        if len(c) <= 60:
            diffs = np.array([a - b for a, b in itertools.permutations(c, 2)])
            if len(diffs):
                candidates.append(np.abs(diffs))
    pool = np.concatenate(candidates) if candidates else np.array([])
    if len(pool) == 0:
        return [raw for raw, val, _, _ in _numbers_in(answer) if not (1900 <= val <= 2100 or abs(val) <= 12)]
    pool = np.abs(pool)

    flagged = []
    for raw, val, dec, unit in _numbers_in(answer):
        if (1900 <= val <= 2100 and dec == 0 and unit == "") or (abs(val) <= 12 and dec == 0 and unit == ""):
            continue
        scale = {"million": 1e6, "thousand": 1e3, "k": 1e3}.get(unit, 1.0)
        tol = 0.5 * 10 ** (-dec) * scale + 1e-9
        target = abs(val) * scale
        if not np.any(np.abs(pool - target) <= tol):
            flagged.append(raw)
    return flagged

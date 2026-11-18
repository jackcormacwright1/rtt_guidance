"""The RTT database schema. One definition serves both the loader (column order and types)
and the text-to-SQL prompt (what each table and column means), so the two can't drift apart."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Column:
    name: str
    type: str
    description: str


@dataclass(frozen=True)
class Table:
    name: str
    description: str
    columns: tuple[Column, ...]


AREAS = {
    "NCL": "North Central London",
    "NEL": "North East London",
    "NWL": "North West London",
    "SEL": "South East London",
    "SWL": "South West London",
}

_KEYS = (
    Column("period", "DATE", "First day of the reporting month, e.g. 2023-04-01 is April 2023."),
    Column("financial_year", "VARCHAR", "NHS financial year (April to March), e.g. '2022-23'."),
    Column("provider_name", "VARCHAR",
           "A named North West London trust (plus Royal Free London), e.g. 'Imperial College Healthcare NHS Trust', "
           "or a group of the other trusts in one London area, e.g. 'South East London providers (grouped)'. "
           "'Other North Central London providers (grouped)' excludes Royal Free, which is listed separately. "
           "Trusts outside North West London other than Royal Free can't be identified individually."),
    Column("provider_is_group", "BOOLEAN", "True when provider_name is a group of trusts rather than one trust."),
    Column("provider_area", "VARCHAR",
           "The London area (integrated care system) the provider is in: 'North Central London', "
           "'North East London', 'North West London', 'South East London' or 'South West London'."),
    Column("commissioner_group", "VARCHAR",
           "Where the patients' care is commissioned: 'North West London' (patients registered in North West "
           "London) or 'Outside North West London' (everyone else). Sum over both for a provider's whole list."),
    Column("treatment_function_name", "VARCHAR",
           "Specialty (treatment function), e.g. 'Trauma and Orthopaedic Service'. Smaller specialties are "
           "combined in 'Other'. There is no all-specialties row: sum over treatment functions for totals."),
)

_BANDS = (
    Column("within_18_weeks", "BIGINT", "Waited 18 weeks or less."),
    Column("over_18_weeks", "BIGINT", "Waited more than 18 weeks (the sum of the four bands below)."),
    Column("weeks_18_to_26", "BIGINT", "Waited more than 18 and up to 26 weeks."),
    Column("weeks_26_to_40", "BIGINT", "Waited more than 26 and up to 40 weeks."),
    Column("weeks_40_to_52", "BIGINT", "Waited more than 40 and up to 52 weeks."),
    Column("over_52_weeks", "BIGINT", "Waited more than 52 weeks. Longer bands (65, 78, 104 weeks) are not "
           "available in this data."),
)

WAITING_LIST = Table(
    "waiting_list",
    "Patients still waiting to start treatment (incomplete RTT pathways) at the end of each month. This is "
    "the waiting list; the 18-week standard is the percentage of it waiting 18 weeks or less.",
    _KEYS + (Column("total_waiting", "BIGINT", "Patients waiting. within_18_weeks + over_18_weeks = total_waiting."),)
    + _BANDS,
)

WAITING_LIST_DTA = Table(
    "waiting_list_dta",
    "The part of the waiting list that has a decision to admit (DTA): patients waiting for treatment as an "
    "inpatient or day case. A subset of waiting_list, so never add the two tables together.",
    _KEYS + (Column("total_waiting", "BIGINT", "Patients waiting with a decision to admit."),) + _BANDS,
)

COMPLETED = Table(
    "completed_pathways",
    "Patients whose RTT pathway ended (they started treatment) during each month, with how long they had waited.",
    _KEYS + (
        Column("pathway_type", "VARCHAR", "'admitted' (treated as an inpatient or day case) or 'non_admitted' "
               "(e.g. treated in outpatients)."),
        Column("total_completed", "BIGINT", "Pathways completed in the month with a known clock start date. "
               "within_18_weeks + over_18_weeks = total_completed, so use it as the denominator for percentages."),
        Column("unknown_clock_start", "BIGINT", "Further pathways completed in the month whose clock start date "
               "wasn't recorded, so their wait is unknown. Not included in total_completed."),
    ) + _BANDS,
)

NEW_PERIODS = Table(
    "new_periods",
    "New RTT pathways (clock starts: new referrals onto consultant-led pathways) in each month. A measure of demand.",
    _KEYS + (Column("new_clock_starts", "BIGINT", "Number of new RTT clock starts in the month."),),
)

COVERAGE = Table(
    "coverage",
    "One row per provider: the months it appears in the data. Use it to check for gaps before comparing totals "
    "across months.",
    (
        Column("provider_name", "VARCHAR", "As in the other tables."),
        Column("provider_area", "VARCHAR", "As in the other tables."),
        Column("first_period", "DATE", "First month with data."),
        Column("last_period", "DATE", "Last month with data."),
        Column("months_reported", "INTEGER", "Number of months with data."),
        Column("missing_months", "VARCHAR", "Months between first_period and last_period with no data, as "
               "'YYYY-MM to YYYY-MM' ranges, or empty."),
    ),
)

DATA_TABLES = (WAITING_LIST, WAITING_LIST_DTA, COMPLETED, NEW_PERIODS)
TABLES = DATA_TABLES + (COVERAGE,)
TABLE_NAMES = {t.name for t in TABLES}


def create_sql(table: Table) -> str:
    cols = ",\n  ".join(f"{c.name} {c.type}" for c in table.columns)
    return f"CREATE TABLE {table.name} (\n  {cols}\n)"


def describe_for_prompt() -> str:
    """Plain-text schema for the model."""
    parts = []
    for t in TABLES:
        parts.append(f"TABLE {t.name}: {t.description}")
        for c in t.columns:
            parts.append(f"  - {c.name} ({c.type}): {c.description}")
        parts.append("")
    return "\n".join(parts).strip()

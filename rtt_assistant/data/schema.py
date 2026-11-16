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


_KEYS = (
    Column("period", "DATE", "First day of the reporting month, e.g. 2026-09-01 is September 2026."),
    Column("provider_code", "VARCHAR", "Organisation code of the provider (NHS trust or independent sector provider)."),
    Column("provider_name", "VARCHAR", "Provider name in capitals, e.g. 'LEWISHAM AND GREENWICH NHS TRUST'."),
    Column("provider_parent_code", "VARCHAR", "Code of the provider's parent organisation (NHS England region or ICB)."),
    Column("provider_parent_name", "VARCHAR", "Name of the provider's parent organisation."),
    Column("treatment_function_code", "VARCHAR", "Treatment function (specialty) code, e.g. 'C_110'."),
    Column("treatment_function_name", "VARCHAR", "Treatment function name, e.g. 'Trauma and Orthopaedic Service'."),
)

WAITING_LIST = Table(
    "waiting_list",
    "Patients still waiting to start treatment (incomplete RTT pathways) at the end of each month, by provider "
    "and treatment function, summed over commissioners. There is no all-specialties row: sum over "
    "treatment functions for totals, and over providers for England.",
    _KEYS + (
        Column("total_waiting", "BIGINT", "Patients waiting with a known clock start date. The denominator for "
               "the 18-week performance measure."),
        Column("within_18_weeks", "BIGINT", "Of total_waiting, those waiting 18 weeks or less."),
        Column("over_18_weeks", "BIGINT", "Of total_waiting, those waiting more than 18 weeks."),
        Column("over_52_weeks", "BIGINT", "Of total_waiting, those waiting more than 52 weeks."),
        Column("over_65_weeks", "BIGINT", "Of total_waiting, those waiting more than 65 weeks."),
        Column("over_78_weeks", "BIGINT", "Of total_waiting, those waiting more than 78 weeks."),
        Column("over_104_weeks", "BIGINT", "Of total_waiting, those waiting more than 104 weeks."),
        Column("unknown_clock_start", "BIGINT", "Patients waiting whose clock start date is unknown (not in total_waiting)."),
    ),
)

COMPLETED = Table(
    "completed_pathways",
    "Patients who started treatment during each month (completed RTT pathways), by provider, treatment function "
    "and pathway type, summed over commissioners.",
    _KEYS + (
        Column("pathway_type", "VARCHAR", "'admitted' (treated as an inpatient or day case) or 'non_admitted' "
               "(e.g. treated in outpatients)."),
        Column("total_completed", "BIGINT", "Pathways completed in the month with a known clock start date."),
        Column("within_18_weeks", "BIGINT", "Of total_completed, those that had waited 18 weeks or less."),
        Column("over_52_weeks", "BIGINT", "Of total_completed, those that had waited more than 52 weeks."),
    ),
)

NEW_PERIODS = Table(
    "new_periods",
    "New RTT pathways (clock starts, i.e. new referrals onto consultant-led pathways) in each month, by provider "
    "and treatment function, summed over commissioners. A measure of demand.",
    _KEYS + (Column("new_clock_starts", "BIGINT", "Number of new RTT clock starts in the month."),),
)

TABLES = (WAITING_LIST, COMPLETED, NEW_PERIODS)
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

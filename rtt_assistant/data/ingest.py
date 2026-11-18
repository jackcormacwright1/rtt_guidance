"""Load the London RTT extract into a DuckDB database.

The extract (data/source/) is NHS England's published Referral to Treatment statistics for London
providers, April 2018 to April 2023, one row per month x provider x commissioner x RTT part x treatment
function. The four North West London trusts (and Royal Free London) appear individually; every other London
trust is grouped by its area ('Other London' with an area code). Commissioners are the North West London
CCGs (which merged and were renamed during the period) or 'Non-NWL'. Waiting times come in five bands.

RTT parts and where they go:

    Part_1A  completed pathways, admitted          -> completed_pathways (pathway_type 'admitted')
    Part_1B  completed pathways, non-admitted      -> completed_pathways (pathway_type 'non_admitted')
    Part_2   incomplete pathways (the waiting list) -> waiting_list
    Part_2A  incomplete pathways with a decision to admit (a subset of Part_2) -> waiting_list_dta
    Part_3   new RTT periods (clock starts)        -> new_periods

The loader groups commissioners into 'North West London' / 'Outside North West London' (so the CCG
renames don't break time series), gives grouped providers readable names, checks that the bands add up to
each row's total, and records which months each provider is present in (the coverage table).
"""

from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from .schema import (AREAS, COMPLETED, COVERAGE, DATA_TABLES, NEW_PERIODS, TABLES, WAITING_LIST,
                     WAITING_LIST_DTA, create_sql)

RAW_COLUMNS = ["year_month", "financial_year", "Provider", "Provider_STP", "Commissioner Org", "RTT Part Type",
               "Treatment Function Name", "Total", "<18wks", "18-26wks", "26-40wks", "40-52wks", ">52wks"]
RAW_BANDS = {"<18wks": "within_18_weeks", "18-26wks": "weeks_18_to_26", "26-40wks": "weeks_26_to_40",
             "40-52wks": "weeks_40_to_52", ">52wks": "over_52_weeks"}
KEYS = ["period", "financial_year", "provider_name", "provider_is_group", "provider_area", "commissioner_group",
        "treatment_function_name"]
NWL_OUTSIDE = "Non-NWL"
SMALL_WORDS = {"and", "of", "the", "for"}


class IngestError(ValueError):
    pass


def read_source(path: str | Path) -> pd.DataFrame:
    """Read the extract from the original workbook (.xlsx, sheet 'Data') or its parquet copy."""
    path = Path(path)
    if path.suffix.lower() == ".parquet":
        raw = pd.read_parquet(path)
    elif path.suffix.lower() in (".xlsx", ".xlsm"):
        raw = pd.read_excel(path, sheet_name="Data", engine="openpyxl")
    else:
        raise IngestError(f"{path.name}: expected .xlsx or .parquet")
    missing = [c for c in RAW_COLUMNS if c not in raw.columns]
    if missing:
        raise IngestError(f"{path.name}: missing columns {missing}")
    return raw


def trust_name(raw: str) -> str:
    """'THE HILLINGDON HOSPITALS NHS FOUNDATION TRUST' -> 'The Hillingdon Hospitals NHS Foundation Trust'."""
    words = []
    for i, w in enumerate(str(raw).strip().split()):
        lw = w.lower()
        if lw == "nhs":
            words.append("NHS")
        elif lw in SMALL_WORDS and i > 0:
            words.append(lw)
        else:
            words.append(lw.capitalize())
    return " ".join(words)


def provider_columns(provider: pd.Series, stp: pd.Series) -> pd.DataFrame:
    area = stp.map(AREAS)
    if area.isna().any():
        raise IngestError(f"Unknown area codes: {sorted(stp[area.isna()].unique())}")
    grouped = provider.str.strip().str.lower() == "other london"
    named = provider.map(trust_name)
    group_name = area + " providers (grouped)"
    # In North Central London, Royal Free is listed separately, so the group holds the other trusts. (Its name
    # avoids the words 'Royal Free' so that a search for Royal Free can't match the group too.)
    group_name = group_name.where(stp != "NCL", "Other North Central London providers (grouped)")
    return pd.DataFrame({"provider_name": named.where(~grouped, group_name),
                         "provider_is_group": grouped, "provider_area": area})


def tidy(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """The raw extract -> the four data tables in schema.py."""
    df = raw[RAW_COLUMNS].copy()
    ym = pd.to_numeric(df["year_month"], errors="raise").astype(int)
    df["period"] = [date(v // 100, v % 100, 1) for v in ym]
    df = pd.concat([df.reset_index(drop=True),
                    provider_columns(df["Provider"].reset_index(drop=True),
                                     df["Provider_STP"].reset_index(drop=True))], axis=1)
    df["commissioner_group"] = (df["Commissioner Org"].str.strip() == NWL_OUTSIDE).map(
        {True: "Outside North West London", False: "North West London"})
    df["treatment_function_name"] = df["Treatment Function Name"].str.strip()
    df["Total"] = pd.to_numeric(df["Total"], errors="raise").astype("int64")
    part = df["RTT Part Type"].str.strip().str.lower()

    def banded(frame: pd.DataFrame, total_name: str, allow_unknown: bool = False) -> pd.DataFrame:
        """Rename the bands and check them against the row total. For completed pathways the total also
        counts pathways with an unknown clock start, which have no band; those are kept separately."""
        frame = frame.rename(columns=RAW_BANDS).copy()
        bands = list(RAW_BANDS.values())
        frame[bands] = frame[bands].apply(pd.to_numeric, errors="raise").fillna(0).astype("int64")
        unknown = frame["Total"] - frame[bands].sum(axis=1)
        bad = (unknown < 0) if allow_unknown else (unknown != 0)
        if bad.any():
            raise IngestError(f"{total_name}: the waiting-time bands don't add up to the total on {int(bad.sum())} rows.")
        frame[total_name] = frame[bands].sum(axis=1)
        if allow_unknown:
            frame["unknown_clock_start"] = unknown
        frame["over_18_weeks"] = frame[bands[1:]].sum(axis=1)
        return frame

    def grouped(frame: pd.DataFrame, table, extra: list[str] | None = None) -> pd.DataFrame:
        cols = [c.name for c in table.columns]
        keys = KEYS + (extra or [])
        out = frame.groupby(keys, as_index=False)[[c for c in cols if c not in keys]].sum()
        return out[cols]

    waiting = grouped(banded(df[part == "part_2"], "total_waiting"), WAITING_LIST)
    dta = grouped(banded(df[part == "part_2a"], "total_waiting"), WAITING_LIST_DTA)
    c = banded(df[part.isin(["part_1a", "part_1b"])], "total_completed", allow_unknown=True)
    c["pathway_type"] = c["RTT Part Type"].str.strip().str.lower().map({"part_1a": "admitted", "part_1b": "non_admitted"})
    completed = grouped(c, COMPLETED, ["pathway_type"])
    n = df[part == "part_3"].copy()
    n["new_clock_starts"] = n["Total"]
    new = grouped(n, NEW_PERIODS)

    unknown = set(part.unique()) - {"part_1a", "part_1b", "part_2", "part_2a", "part_3"}
    if unknown:
        raise IngestError(f"Unknown RTT part types: {sorted(unknown)}")
    return {WAITING_LIST.name: waiting, WAITING_LIST_DTA.name: dta, COMPLETED.name: completed, NEW_PERIODS.name: new}


def _ranges(months: list[date]) -> str:
    """[2019-02, 2019-03, ..., 2021-02] -> '2019-02 to 2021-02'; separate runs joined with '; '."""
    if not months:
        return ""
    runs, start, prev = [], months[0], months[0]
    for m in months[1:]:
        if (m.year * 12 + m.month) - (prev.year * 12 + prev.month) != 1:
            runs.append((start, prev))
            start = m
        prev = m
    runs.append((start, prev))
    return "; ".join(f"{a:%Y-%m}" if a == b else f"{a:%Y-%m} to {b:%Y-%m}" for a, b in runs)


def coverage(waiting: pd.DataFrame) -> pd.DataFrame:
    """Which months each provider is present in, judged from the waiting list."""
    all_months = sorted(waiting["period"].unique())
    rows = []
    for (name, area), g in waiting.groupby(["provider_name", "provider_area"]):
        have = set(g["period"])
        first, last = min(have), max(have)
        gap = [m for m in all_months if first <= m <= last and m not in have]
        rows.append({"provider_name": name, "provider_area": area, "first_period": first, "last_period": last,
                     "months_reported": len(have), "missing_months": _ranges(gap)})
    return pd.DataFrame(rows)[[c.name for c in COVERAGE.columns]]


def build_database(source: str | Path, db_path: str | Path) -> pd.DataFrame:
    """Load the extract into a fresh DuckDB file. Returns one summary row per month (London totals)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    if db_path.exists():
        db_path.unlink()
    tables = tidy(read_source(source))
    tables[COVERAGE.name] = coverage(tables[WAITING_LIST.name])

    con = duckdb.connect(str(db_path))
    try:
        for t in TABLES:
            con.execute(create_sql(t))
            df = tables[t.name]
            if t in DATA_TABLES:
                keys = KEYS + (["pathway_type"] if t is COMPLETED else [])
                dupes = int(df.duplicated(subset=keys).sum())
                if dupes:
                    raise IngestError(f"{t.name}: {dupes} duplicate rows.")
            con.register("incoming", df[[c.name for c in t.columns]])
            con.execute(f"INSERT INTO {t.name} SELECT * FROM incoming")
            con.unregister("incoming")
        con.execute("CREATE TABLE load_log (source VARCHAR, rows_read BIGINT, loaded_at TIMESTAMP)")
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        con.execute("INSERT INTO load_log VALUES (?, ?, ?)", [Path(source).name, int(sum(map(len, tables.values()))), now])
        summary = con.execute(
            "SELECT period, COUNT(DISTINCT provider_name) AS providers, SUM(total_waiting) AS total_waiting, "
            "ROUND(100.0 * SUM(within_18_weeks) / SUM(total_waiting), 1) AS pct_within_18_weeks "
            "FROM waiting_list GROUP BY period ORDER BY period").df()
    finally:
        con.close()
    return summary

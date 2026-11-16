"""Load NHS England's monthly RTT 'Full CSV data file' extracts into a DuckDB database.

Each monthly file has one row per provider x commissioner x treatment function x RTT part, with
patient counts in weekly bands ('Gt 00 To 01 Weeks SUM 1', ..., 'Gt 104 Weeks SUM 1'). The parts are:

    Part_1A  completed pathways, admitted          -> completed_pathways (pathway_type 'admitted')
    Part_1B  completed pathways, non-admitted      -> completed_pathways (pathway_type 'non_admitted')
    Part_2   incomplete pathways (still waiting)   -> waiting_list
    Part_2A  incomplete pathways with a decision to admit (a subset of Part_2; not loaded)
    Part_3   new RTT periods (clock starts)        -> new_periods

The loader sums over commissioners, drops the all-specialties total rows (so totals are always sums),
turns the weekly bands into the cumulative counts in schema.py, and checks totals as it goes.
"""

from __future__ import annotations

import io
import re
import zipfile
from datetime import date, datetime, timezone
from pathlib import Path

import duckdb
import pandas as pd

from .schema import COMPLETED, NEW_PERIODS, TABLES, WAITING_LIST, create_sql

BAND = re.compile(r"^gt_(\d+)_to_(\d+)_weeks_sum_1$")
TOP_BAND = re.compile(r"^gt_(\d+)_weeks_sum_1$")
MONTHS = {m.lower(): i for i, m in enumerate(
    ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
     "November", "December"], start=1)}
KEY_MAP = {
    "provider_org_code": "provider_code", "provider_org_name": "provider_name",
    "provider_parent_org_code": "provider_parent_code", "provider_parent_name": "provider_parent_name",
    "treatment_function_code": "treatment_function_code", "treatment_function_name": "treatment_function_name",
}
KEYS = ["period", "provider_code", "provider_name", "provider_parent_code", "provider_parent_name",
        "treatment_function_code", "treatment_function_name"]


class IngestError(ValueError):
    pass


def _norm(col: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", col.strip().lower()).strip("_")


def parse_period(value: str) -> date:
    """'RTT-September-2026' (or 'September 2026', '2026-09') -> date(2026, 9, 1)."""
    v = str(value).strip()
    m = re.search(r"([A-Za-z]+)[-\s_]+(\d{4})", v)
    if m and m.group(1).lower() in MONTHS:
        return date(int(m.group(2)), MONTHS[m.group(1).lower()], 1)
    m = re.search(r"(\d{4})-(\d{1,2})", v)
    if m:
        return date(int(m.group(1)), int(m.group(2)), 1)
    raise IngestError(f"Unrecognised period: {value!r}")


def read_extract(path: str | Path) -> pd.DataFrame:
    """Read one monthly extract (.zip containing a CSV, or a .csv) with normalised column names."""
    path = Path(path)
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as z:
            csvs = [n for n in z.namelist() if n.lower().endswith(".csv")]
            if len(csvs) != 1:
                raise IngestError(f"{path.name}: expected one CSV inside, found {csvs}")
            raw = pd.read_csv(io.BytesIO(z.read(csvs[0])), dtype=str, low_memory=False)
    else:
        raw = pd.read_csv(path, dtype=str, low_memory=False)
    raw.columns = [_norm(c) for c in raw.columns]
    return raw


def _bands(cols: list[str]) -> list[tuple[str, int, int | None]]:
    """(column, lower weeks, upper weeks or None for the open top band)."""
    out = []
    for c in cols:
        if m := BAND.match(c):
            out.append((c, int(m.group(1)), int(m.group(2))))
        elif m := TOP_BAND.match(c):
            out.append((c, int(m.group(1)), None))
    if not out:
        raise IngestError("No weekly band columns found (expected e.g. 'Gt 00 To 01 Weeks SUM 1').")
    return sorted(out, key=lambda b: b[1])


def tidy(raw: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """One raw monthly extract -> the three tables in schema.py."""
    required = {"period", "rtt_part_type", *KEY_MAP}
    missing = required - set(raw.columns)
    if missing:
        raise IngestError(f"Missing columns: {sorted(missing)}")
    df = raw.rename(columns=KEY_MAP).copy()
    df["period"] = df["period"].map(parse_period)
    bands = _bands(list(df.columns))
    band_cols = [b[0] for b in bands]
    numeric = band_cols + [c for c in ("total", "patients_with_unknown_clock_start_date", "total_all") if c in df]
    for c in numeric:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0).astype("int64")

    # Drop the all-specialties rows so that every total in the database is a sum of its parts.
    is_total_tf = df["treatment_function_code"].str.upper().str.endswith("999")
    df = df[~is_total_tf]
    part = df["rtt_part_type"].str.strip().str.lower()

    def cum(frame: pd.DataFrame, lower_than: int | None = None, over: int | None = None) -> pd.Series:
        if lower_than is not None:  # bands entirely at or below `lower_than` weeks
            cols = [c for c, lo, hi in bands if hi is not None and hi <= lower_than]
        else:  # bands entirely above `over` weeks
            cols = [c for c, lo, hi in bands if lo >= over]
        return frame[cols].sum(axis=1)

    def grouped(frame: pd.DataFrame, extra_keys: list[str] | None = None) -> pd.DataFrame:
        return frame.groupby(KEYS + (extra_keys or []), dropna=False, as_index=False).sum(numeric_only=True)

    # Waiting list (incomplete pathways)
    w = df[part == "part_2"].copy()
    w["total_waiting"] = w[band_cols].sum(axis=1)
    if "total" in w and not (w["total"] == w["total_waiting"]).all():
        bad = int((w["total"] != w["total_waiting"]).sum())
        raise IngestError(f"Waiting list: weekly bands don't add up to 'Total' on {bad} rows.")
    w["within_18_weeks"] = cum(w, lower_than=18)
    w["over_18_weeks"] = cum(w, over=18)
    for k in (52, 65, 78, 104):
        w[f"over_{k}_weeks"] = cum(w, over=k)
    w["unknown_clock_start"] = w.get("patients_with_unknown_clock_start_date", 0)
    waiting = grouped(w[KEYS + [c.name for c in WAITING_LIST.columns[len(KEYS):]]])

    # Completed pathways
    c = df[part.isin(["part_1a", "part_1b"])].copy()
    c["pathway_type"] = part[c.index].map({"part_1a": "admitted", "part_1b": "non_admitted"})
    c["total_completed"] = c[band_cols].sum(axis=1)
    c["within_18_weeks"] = cum(c, lower_than=18)
    c["over_52_weeks"] = cum(c, over=52)
    completed = grouped(c[KEYS + ["pathway_type", "total_completed", "within_18_weeks", "over_52_weeks"]],
                        ["pathway_type"])
    completed = completed[[col.name for col in COMPLETED.columns]]

    # New periods (clock starts): counts are in 'Total' (or the bands, whichever is populated)
    n = df[part == "part_3"].copy()
    n["new_clock_starts"] = n["total"] if "total" in n else n[band_cols].sum(axis=1)
    new = grouped(n[KEYS + ["new_clock_starts"]])

    return {WAITING_LIST.name: waiting, COMPLETED.name: completed, NEW_PERIODS.name: new}


def build_database(sources: list[str | Path], db_path: str | Path) -> pd.DataFrame:
    """Load every extract in ``sources`` into a fresh DuckDB file. Returns one summary row per month."""
    db_path = Path(db_path)
    if db_path.exists():
        db_path.unlink()
    frames: dict[str, list[pd.DataFrame]] = {t.name: [] for t in TABLES}
    loaded = []
    for src in sorted(sources):
        tables = tidy(read_extract(src))
        periods = set(tables[WAITING_LIST.name]["period"])
        loaded.append({"file": Path(src).name, "periods": ", ".join(str(p) for p in sorted(periods))})
        for name, df in tables.items():
            frames[name].append(df)

    con = duckdb.connect(str(db_path))
    for t in TABLES:
        con.execute(create_sql(t))
        df = pd.concat(frames[t.name], ignore_index=True)
        dupes = df.duplicated(subset=[c.name for c in t.columns if c.type == "VARCHAR" or c.name == "period"]).sum()
        if dupes:
            raise IngestError(f"{t.name}: {dupes} duplicate rows - is the same month loaded twice?")
        con.register("incoming", df[[c.name for c in t.columns]])
        con.execute(f"INSERT INTO {t.name} SELECT * FROM incoming")
        con.unregister("incoming")
    con.execute("CREATE TABLE load_log (file VARCHAR, periods VARCHAR, loaded_at TIMESTAMP)")
    now = datetime.now(timezone.utc).replace(tzinfo=None)
    con.executemany("INSERT INTO load_log VALUES (?, ?, ?)", [(r["file"], r["periods"], now) for r in loaded])
    summary = con.execute(
        "SELECT period, COUNT(DISTINCT provider_code) AS providers, SUM(total_waiting) AS total_waiting, "
        "ROUND(100.0 * SUM(within_18_weeks) / SUM(total_waiting), 2) AS pct_within_18_weeks "
        "FROM waiting_list GROUP BY period ORDER BY period").df()
    con.close()
    return summary

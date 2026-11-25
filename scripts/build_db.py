"""Build data/rtt.duckdb from the London RTT extract.

Usage:
    python scripts/build_db.py                                        # from data/source/ (committed)
    python scripts/build_db.py "National RTT Waiting List Data.xlsx"  # from the original workbook

The figures are NHS England's Referral to Treatment waiting times statistics, published under the
Open Government Licence: https://www.england.nhs.uk/statistics/statistical-work-areas/rtt-waiting-times/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtt_assistant.data.ingest import build_database  # noqa: E402

SOURCE = ROOT / "data" / "source" / "london_rtt_2018-04_to_2023-04.parquet"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("source", nargs="?", type=Path, default=SOURCE, help="the extract (.parquet or .xlsx)")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "rtt.duckdb")
    args = ap.parse_args()
    summary = build_database(args.source, args.out)
    print(summary.to_string(index=False))
    print(f"\nWrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) from {args.source.name}.")


if __name__ == "__main__":
    main()

"""Build data/rtt.duckdb from NHS England's monthly RTT 'Full CSV data file' extracts.

Usage:
    python scripts/build_db.py data/raw/*.zip            # from files you've downloaded
    python scripts/build_db.py --download 13             # fetch the latest 13 months first (needs internet)

The extracts are published under the Open Government Licence by NHS England:
https://www.england.nhs.uk/statistics/statistical-work-areas/rtt-waiting-times/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtt_assistant.data.ingest import build_database  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("files", nargs="*", type=Path, help="monthly extracts (.zip or .csv)")
    ap.add_argument("--download", type=int, metavar="MONTHS", help="download the latest MONTHS extracts first")
    ap.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "raw")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "rtt.duckdb")
    args = ap.parse_args()

    files = list(args.files)
    if args.download:
        from rtt_assistant.data.download import download_latest
        files += download_latest(args.download, args.raw_dir)
    if not files:
        files = sorted(args.raw_dir.glob("*.zip"))
    if not files:
        ap.error("No extracts given or found in data/raw/.")
    summary = build_database(sorted(set(files)), args.out)
    print(summary.to_string(index=False))
    print(f"\nWrote {args.out} ({args.out.stat().st_size / 1e6:.1f} MB) from {len(set(files))} file(s).")


if __name__ == "__main__":
    main()

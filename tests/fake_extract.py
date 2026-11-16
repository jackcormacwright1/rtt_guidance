"""A small synthetic extract in NHS England's 'Full CSV data file' layout, for tests."""

from __future__ import annotations

import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = [f"Gt {i:02d} To {i + 1:02d} Weeks SUM 1" for i in range(104)] + ["Gt 104 Weeks SUM 1"]
PROVIDERS = [("RJ2", "LEWISHAM AND GREENWICH NHS TRUST"), ("RJZ", "KING'S COLLEGE HOSPITAL NHS FOUNDATION TRUST")]
COMMISSIONERS = [("QKK", "NHS SOUTH EAST LONDON INTEGRATED CARE BOARD"), ("QWE", "NHS SOUTH WEST LONDON INTEGRATED CARE BOARD")]
TFS = [("C_110", "Trauma and Orthopaedic Service"), ("C_100", "General Surgery Service")]
PARTS = [("Part_1A", "Completed Pathways For Admitted Patients"),
         ("Part_1B", "Completed Pathways For Non-Admitted Patients"),
         ("Part_2", "Incomplete Pathways"), ("Part_2A", "Incomplete Pathways with DTA"),
         ("Part_3", "New RTT Periods - All Patients")]


def make_extract(month: str, year: int, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for pc, pn in PROVIDERS:
        for cc, cn in COMMISSIONERS:
            for part, desc in PARTS:
                tf_rows = []
                for tc, tn in TFS:
                    if part == "Part_3":
                        bands = np.zeros(len(BANDS), dtype=int)
                        total = int(rng.integers(50, 200))
                    else:
                        bands = rng.poisson(np.linspace(30, 0.5, len(BANDS)))
                        total = int(bands.sum())
                    tf_rows.append((tc, tn, bands, total))
                all_bands = sum(r[2] for r in tf_rows)
                tf_rows.append(("C_999", "Total", all_bands, sum(r[3] for r in tf_rows)))
                for tc, tn, bands, total in tf_rows:
                    rows.append({
                        "Period": f"RTT-{month}-{year}", "Provider Parent Org Code": "Y56",
                        "Provider Parent Name": "NHS ENGLAND LONDON", "Provider Org Code": pc,
                        "Provider Org Name": pn, "Commissioner Parent Org Code": "Y56",
                        "Commissioner Parent Name": "NHS ENGLAND LONDON", "Commissioner Org Code": cc,
                        "Commissioner Org Name": cn, "RTT Part Type": part, "RTT Part Description": desc,
                        "Treatment Function Code": tc, "Treatment Function Name": tn,
                        **dict(zip(BANDS, (int(b) for b in bands))),
                        "Total": total, "Patients with unknown clock start date": int(rng.integers(0, 5)),
                        "Total All": total,
                    })
    return pd.DataFrame(rows)


def write_zip(df: pd.DataFrame, path: Path) -> Path:
    csv = df.to_csv(index=False)
    with zipfile.ZipFile(path, "w") as z:
        z.writestr(path.stem + ".csv", csv)
    return path

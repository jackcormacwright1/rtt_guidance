"""A small synthetic extract in the same layout as the London RTT workbook, for tests."""

from __future__ import annotations

import numpy as np
import pandas as pd

PROVIDERS = [("IMPERIAL COLLEGE HEALTHCARE NHS TRUST", "NWL"), ("THE HILLINGDON HOSPITALS NHS FOUNDATION TRUST", "NWL"),
             ("ROYAL FREE LONDON NHS FOUNDATION TRUST", "NCL"), ("Other London", "NCL"), ("Other London", "SEL")]
COMMISSIONERS = ["NHS EALING CCG", "NHS NORTH WEST LONDON CCG", "Non-NWL"]
TFS = ["Trauma and Orthopaedic Service", "General Surgery Service"]
PARTS = [("Part_1A", "Completed Pathways For Admitted Patients"),
         ("Part_1B", "Completed Pathways For Non-Admitted Patients"),
         ("Part_2", "Incomplete Pathways"), ("Part_2A", "Incomplete Pathways with DTA"),
         ("Part_3", "New RTT Periods - All Patients")]
BANDS = ["<18wks", "18-26wks", "26-40wks", "40-52wks", ">52wks"]


def make_source(months: list[int], seed: int = 0, royal_free_missing: tuple[int, ...] = ()) -> pd.DataFrame:
    """Rows for each month (YYYYMM) x provider x commissioner x part x specialty."""
    rng = np.random.default_rng(seed)
    rows = []
    for ym in months:
        fy_start = ym // 100 if ym % 100 >= 4 else ym // 100 - 1
        for prov, stp in PROVIDERS:
            if prov.startswith("ROYAL") and ym in royal_free_missing:
                continue
            for comm in COMMISSIONERS:
                for part, desc in PARTS:
                    for tf in TFS:
                        if part == "Part_3":
                            bands = [np.nan] * 5
                            total = int(rng.integers(20, 80))
                        else:
                            b = rng.poisson([60, 15, 10, 5, 2])
                            bands = [float(x) for x in b]
                            unknown = int(rng.integers(0, 3)) if part in ("Part_1A", "Part_1B") else 0
                            total = int(b.sum()) + unknown
                        rows.append({"year_month": ym, "financial_year": f"{fy_start}-{str(fy_start + 1)[2:]}",
                                     "financial_month": 1, "month": "X", "Provider": prov, "Provider_STP": stp,
                                     "Commissioner Org": comm, "RTT Part Type": part, "RTT Part Description": desc,
                                     "Treatment Function Name": tf, "Total": total, **dict(zip(BANDS, bands))})
    return pd.DataFrame(rows)

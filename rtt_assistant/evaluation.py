"""Scoring for the evaluation sets in eval/.

Data questions: each has a hand-written gold SQL query (or is marked unanswerable). A generated answer
is correct if its result contains the gold result's key values: every number in the gold key columns
(to the gold's precision) and every text value (case-insensitive), wherever they appear in the generated
result. That tolerates different column names, orders and extra columns, but not different numbers.

Rules questions: each lists the source document and page(s) that answer it. Retrieval is scored as
hit@k: whether any of the top k passages comes from one of those pages.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _decimals(x: float) -> int:
    s = f"{x:.6f}".rstrip("0").rstrip(".")
    return len(s.split(".")[1]) if "." in s else 0


def result_contains(generated: pd.DataFrame | None, gold: pd.DataFrame, key_columns: list[str] | None = None) -> bool:
    if generated is None or len(gold) == 0:
        return generated is not None and len(gold) == 0 and len(generated) == 0
    cols = key_columns or list(gold.columns)
    gen_numbers = []
    gen_text = set()
    for c in generated.columns:
        s = generated[c]
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().all() and not pd.api.types.is_datetime64_any_dtype(s):
            gen_numbers.append(num.to_numpy(dtype=float))
        else:
            gen_text |= {str(v).strip().lower() for v in s.dropna()}
            gen_text |= {str(v)[:10] for v in s.dropna()}  # dates as YYYY-MM-DD
    pool = np.concatenate(gen_numbers) if gen_numbers else np.array([])

    for c in cols:
        s = gold[c]
        num = pd.to_numeric(s, errors="coerce")
        if num.notna().all() and not pd.api.types.is_datetime64_any_dtype(s):
            for v in num.to_numpy(dtype=float):
                tol = 0.5 * 10 ** (-_decimals(v)) + 1e-9
                if not len(pool) or not np.any(np.abs(pool - v) <= tol):
                    return False
        else:
            for v in s.dropna():
                if str(v).strip().lower() not in gen_text and str(v)[:10] not in gen_text:
                    return False
    return True


def hit_at_k(retrieved: list[tuple[str, int | None]], expected: list[dict]) -> bool:
    """``retrieved``: (source file name, page) for each passage. ``expected``: [{source, pages}]."""
    for src, page in retrieved:
        for e in expected:
            if src == e["source"] and (not e.get("pages") or page in e["pages"]):
                return True
    return False

"""Evaluate retrieval for the rules questions in eval/rules_questions.yaml.

Reports hit@k: the share of questions where at least one of the top k passages comes from a page
that answers the question. With --answers, also generates an answer for each question (needs
OPENAI_API_KEY) and saves them next to the expected gist for review.

Usage:  python scripts/eval_rules.py [--k 4] [--answers] [--no-govuk]
"""

from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtt_assistant import rag  # noqa: E402
from rtt_assistant.evaluation import hit_at_k  # noqa: E402

SOURCES = {"guidance": rag.PDFS[0].name, "faqs": rag.PDFS[1].name}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--answers", action="store_true", help="also generate answers (uses the OpenAI API)")
    ap.add_argument("--no-govuk", action="store_true", help="don't fetch the GOV.UK rules suite")
    args = ap.parse_args()

    questions = yaml.safe_load((ROOT / "eval" / "rules_questions.yaml").read_text())
    index = rag.Index(rag.build_chunks(include_govuk=not args.no_govuk), rag.SentenceTransformerEmbedder())
    llm = None
    if args.answers:
        from rtt_assistant.llm import OpenAIChat
        llm = OpenAIChat()

    rows = []
    for q in questions:
        expected = [{"source": SOURCES[e["source"]], "pages": e.get("pages")} for e in q["expected"]]
        retrieved = index.search(q["question"], k=args.k)
        hit = hit_at_k([(c.source, c.page) for c, _ in retrieved], expected)
        row = {"id": q["id"], "question": q["question"], f"hit@{args.k}": hit,
               "top_passages": "; ".join(c.citation for c, _ in retrieved), "expected_gist": q["answer"]}
        if llm:
            row["answer"] = llm.complete(rag.SYSTEM, rag.build_prompt(q["question"], retrieved))
        rows.append(row)
        print(f"{q['id']}  {'hit ' if hit else 'MISS'}  {q['question']}")

    df = pd.DataFrame(rows)
    out = ROOT / "eval" / "results" / f"rules_{date.today()}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"\nhit@{args.k}: {df[f'hit@{args.k}'].mean():.0%} of {len(df)} questions. Details: {out}")


if __name__ == "__main__":
    main()

"""Evaluate the text-to-SQL pipeline on eval/data_questions.yaml (needs data/rtt.duckdb and a model; see rtt_assistant/llm.py).

For each question:
  * answerable questions: the generated query's result must contain the gold query's key values
    (see rtt_assistant/evaluation.py), and the summary is checked for numbers not in the result;
  * unanswerable questions: the model must decline rather than write a query.

Usage:  python scripts/eval_data.py [--provider groq] [--model openai/gpt-oss-120b] [--only d01,d02]
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rtt_assistant.data.schema import TABLE_NAMES  # noqa: E402
from rtt_assistant.evaluation import result_contains  # noqa: E402
from rtt_assistant.llm import OpenAICompatibleChat  # noqa: E402
from rtt_assistant.pipeline import answer_question  # noqa: E402
from rtt_assistant.sql.generate import database_facts  # noqa: E402
from rtt_assistant.sql.run import connect_readonly, run_query  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--provider", default=None, help="groq, ollama, huggingface or openai (default: LLM_PROVIDER or groq)")
    ap.add_argument("--model", default=None)
    ap.add_argument("--only", default="", help="comma-separated question ids")
    args = ap.parse_args()

    con = connect_readonly(ROOT / "data" / "rtt.duckdb")
    facts = database_facts(con)
    llm = OpenAICompatibleChat(provider=args.provider, model=args.model)
    if not llm.available:
        sys.exit(f"Set {llm.provider.key_env} first.")
    questions = yaml.safe_load((ROOT / "eval" / "data_questions.yaml").read_text())
    if args.only:
        keep = set(args.only.split(","))
        questions = [q for q in questions if q["id"] in keep]

    rows = []
    for q in questions:
        t0 = time.time()
        ans = answer_question(q["question"], llm, con, facts)
        secs = time.time() - t0
        if q.get("gold_sql"):
            gold = run_query(con, q["gold_sql"], TABLE_NAMES)
            correct = ans.error is None and not ans.cannot_answer and result_contains(ans.result, gold, q.get("key_columns"))
        else:
            correct = ans.cannot_answer
        rows.append({"id": q["id"], "category": q.get("category", ""), "question": q["question"],
                     "answerable": bool(q.get("gold_sql")), "correct": correct,
                     "ran": ans.result is not None, "attempts": len(ans.attempts),
                     "ungrounded_numbers": ", ".join(ans.ungrounded), "seconds": round(secs, 1),
                     "llm_calls": ans.llm_calls, "sql": ans.sql, "answer": ans.answer, "error": ans.error})
        print(f"{q['id']}  {'ok  ' if correct else 'FAIL'}  {q['question']}")

    df = pd.DataFrame(rows)
    out = ROOT / "eval" / "results" / f"data_{date.today()}_{llm.name.replace('/', '-').replace(':', '_')}.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    a = df[df.answerable]
    u = df[~df.answerable]
    print(f"\nModel: {llm.name}")
    print(f"Answerable questions correct:  {a.correct.mean():.0%} ({a.correct.sum()}/{len(a)})")
    print(f"Queries that ran:              {a.ran.mean():.0%}, needing a retry: {(a.attempts > 1).mean():.0%}")
    print(f"Unanswerable correctly refused: {u.correct.mean():.0%} ({u.correct.sum()}/{len(u)})" if len(u) else "")
    print(f"Summaries with unmatched numbers: {(a.ungrounded_numbers != '').mean():.0%}")
    print(f"Median seconds per question: {df.seconds.median():.1f}. Details: {out}")


if __name__ == "__main__":
    main()

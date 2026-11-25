# RTT assistant

Ask questions about NHS Referral to Treatment (RTT) waiting times in plain English, and get answers you can check.

- **Ask the data:** a language model turns the question into SQL over five years of London RTT statistics
  (April 2018 to April 2023), the app checks and runs the query, and the model explains the result. The query,
  the result table and any caveats are shown with every answer.
- **Ask the rules:** questions about how RTT waits are counted (does a cancelled appointment stop the clock?)
  are answered from the published guidance only, with the passages used shown as sources.

Built with Python, DuckDB, sqlglot, sentence-transformers, FAISS and Streamlit. The model is any
OpenAI-compatible endpoint; the default is an open-weight model on Groq's free tier.

## How a data question is answered

```
question ──► model writes SQL (schema, data facts, rules, examples) ──► sqlglot check: one read-only SELECT
                    ▲                                                        │
                    └──────── error fed back, one retry ◄── DuckDB (read-only, no file or network access,
                                                                │           10 s limit, 500 rows)
                                                                ▼
            answer + query + table ◄── number check, coverage caveat ◄── model summarises the result
```

The model's SQL is treated as untrusted. It has to pass a parser check (one `SELECT`, known tables only, no
functions that read files or URLs) and then runs on a read-only connection with external access switched off,
so a query that slipped through still couldn't touch anything else. After the model writes its answer, every
number in it is matched against the query result, and unmatched numbers are flagged. If the result spans months
in which a trust is missing from the data, the answer says so.

[docs/HOW_IT_WORKS.md](docs/HOW_IT_WORKS.md) has the details (it's also the "How it works" tab in the app).

## The data

NHS England's RTT statistics for London providers, published under the Open Government Licence, by month,
provider, specialty and commissioner, April 2018 to April 2023 (`data/source/`, 194,000 rows). The four North
West London trusts and Royal Free London are listed individually; other London trusts are grouped by area.
The loader (`rtt_assistant/data/ingest.py`) checks that every row's waiting-time bands add up to its total,
keeps completed pathways with an unknown clock start separately, and records which months each provider
appears in. One gap matters: Royal Free London is missing from February 2019 to February 2021.

## Evaluation

`eval/data_questions.yaml` has 22 questions: 16 with a hand-written correct query and 6 the data can't answer
(78-week waits, months after April 2023, cancer and A&E waits, individual consultants, a trust only available
within a group). An answer is correct if its result contains the correct query's key values; an unanswerable
question is correct only if the model declines.

```bash
python scripts/eval_data.py --oracle     # the correct queries in place of the model: checks the harness (22/22)
python scripts/eval_data.py              # the configured model
python scripts/eval_rules.py             # retrieval hit@4 on 18 rules questions with known source pages
```

## Run it

```bash
pip install -r requirements.txt
python scripts/build_db.py               # builds data/rtt.duckdb (the app also does this on first run)
export GROQ_API_KEY=...                  # free key from console.groq.com, or LLM_PROVIDER=ollama for a local model
streamlit run app.py
python -m pytest                         # 40 tests; no model or network needed
```

Settings (environment variables or Streamlit secrets): `LLM_PROVIDER` (`groq`, `ollama`, `huggingface`,
`openai`), `LLM_MODEL`, the provider's key, `MAX_QUESTIONS_PER_SESSION` and `MAX_LLM_CALLS_PER_DAY`.

## Layout

```
app.py                       Streamlit app: Ask the data, Ask the rules, How it works
rtt_assistant/
  data/schema.py             tables and columns, used by both the loader and the prompt
  data/ingest.py             extract -> DuckDB, with checks and the coverage table
  sql/generate.py            prompt, rules and examples for writing SQL
  sql/guard.py, sql/run.py   query checks and read-only, time-limited execution
  sql/summarise.py           summary prompt, number check, coverage caveat
  pipeline.py                question -> query -> result -> answer
  rag.py                     retrieval over the RTT guidance
  llm.py                     one client for any OpenAI-compatible provider
eval/                        test questions for both halves
scripts/                     build the database, run the evaluations
```

## Limits

The data ends in April 2023 and covers London only. Waits are banded up to "over 52 weeks", so longer waits
can't be counted. Answers come from a language model: the checks catch unsafe queries and numbers that aren't in
the result, not every misreading of a question, so check the query and table shown with each answer.

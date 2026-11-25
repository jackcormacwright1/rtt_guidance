# How it works

The assistant answers two kinds of question. **Ask the data** turns a question about waiting times into a
database query, runs it and explains the result. **Ask the rules** finds the passages of the published RTT
guidance that answer a question about how waiting times are counted, and answers from those passages only.
Both use a language model, and in both the model only sees what the app chooses to give it.

## Ask the data

### The data

The database holds NHS England's Referral to Treatment (RTT) statistics for London providers from April 2018
to April 2023, which covers the years before COVID, the collapse in elective activity in 2020 and the
recovery that followed. Each month has four measures, each by provider, specialty and where the patients' care
is commissioned:

| Table | What it counts |
|---|---|
| `waiting_list` | Patients still waiting to start treatment at the end of the month (incomplete pathways). |
| `waiting_list_dta` | The part of the waiting list with a decision to admit (waiting for inpatient or day-case treatment). |
| `completed_pathways` | Patients who started treatment in the month, admitted or non-admitted, with how long they had waited. |
| `new_periods` | New RTT clock starts in the month: the demand side. |

Waits are recorded in five bands: up to 18 weeks, 18 to 26, 26 to 40, 40 to 52, and over 52 weeks. The four
North West London trusts and Royal Free London appear individually. Every other London trust is grouped by
area (for example, *South East London providers (grouped)*). Patients are split into those whose care is
commissioned in North West London and everyone else.

A fifth table, `coverage`, records the months each provider appears in. Royal Free London is missing from
February 2019 to February 2021, so any London or North Central London total for those months is too low. When an
answer covers those months, the app says so (see step 6 below). South West London's grouped total also rises by
about 39,000 between December 2018 and January 2019, which looks like a trust starting to report again, so
comparisons across that month need care.

### From question to answer

1. **Write the query.** The model gets the question, a description of every table and column (from
   `rtt_assistant/data/schema.py`, the same definition the loader uses), facts about the loaded data (the months
   available, the provider and specialty names, the coverage gaps), a set of rules and five worked examples.
   It replies with JSON holding one SQL query and any assumptions it made, or with a refusal if the data can't
   answer the question, such as a question about cancer waits or waits over 78 weeks.
2. **Check it.** The query is parsed with [sqlglot](https://github.com/tobymao/sqlglot) and rejected unless it is
   a single read-only `SELECT` over the known tables. That rules out second statements, changes to the data, and
   functions that read files or URLs.
3. **Run it safely.** The database is opened read-only with DuckDB's external access switched off and its
   configuration locked, so even a query that got past step 2 can't reach files or the network. Queries are
   stopped after 10 seconds and results are capped at 500 rows.
4. **Retry once.** If the check or the database rejects the query (a misspelt column, say), the error goes back to
   the model, which gets one more attempt.
5. **Explain it.** The model writes a two-to-four-sentence answer from the query result, told to use only the
   numbers in the result and to say which months they refer to.
6. **Check the answer.** Every number in the answer is compared with the result: it must match a value in the
   result, that value in thousands or millions, or the difference between two values in the same column. Anything
   else is shown as a warning. If the result covers months in which a provider is missing, a coverage caveat is
   attached, unless the query is limited to other providers.

The query, any error sent back to the model, and the assumptions are all shown under each answer, so the answer
can be checked against the table.

### How well it works

`eval/data_questions.yaml` holds 22 test questions: 16 answerable ones, each with a hand-written correct query,
covering lookups, trends, rankings, shares, demand, flows between areas, completed pathways and coverage; and 6
that the data can't answer (waits over 78 weeks, months after April 2023, cancer and A&E waits, individual
consultants, and a trust that is only available as part of a group). An answer counts as correct if its result
contains the values from the correct query's result, wherever they appear, so different column names or extra
columns don't count against it but different numbers do. An unanswerable question counts as correct only if the
model declines. `python scripts/eval_data.py` runs the set against the configured model; `--oracle` runs it with
the correct queries in place of the model, which checks the test harness itself.

## Ask the rules

This is retrieval-augmented generation over the GOV.UK RTT rules suite, NHS England's *Recording and reporting
RTT waiting times guidance* (v5.2, February 2025) and its FAQs, and a local access policy. The documents are
split into overlapping passages of about 1,800 characters, each passage is turned into a vector with the
`all-MiniLM-L6-v2` sentence-transformers model, and the vectors are searched with FAISS by cosine similarity.
The four closest passages go to the model, which is told to answer only from them and to cite them as [S1] to
[S4]. If even the closest passage is only loosely related, the app says the guidance may not cover the question.

`eval/rules_questions.yaml` holds 18 questions, each with the document and page that answers it. Retrieval is
scored as hit@4: whether any of the four passages comes from one of those pages
(`python scripts/eval_rules.py`).

## The model

Every model call goes through one OpenAI-compatible client (`rtt_assistant/llm.py`), so the provider is a setting.
The default is an open-weight model on Groq's free tier; Ollama runs open-weight models locally at no cost.
A public deployment is limited to 15 questions per session and 80 model calls per day.

## Limits

- The data stops in April 2023 and covers London only, with most trusts available only as area groups.
- The longest waiting-time band is "over 52 weeks", so 65-, 78- and 104-week waits can't be counted.
- The answers are generated by a language model. The checks above catch unsafe queries and numbers that aren't
  in the result, but not every misreading of a question, so check the query and the table before relying on an
  answer.

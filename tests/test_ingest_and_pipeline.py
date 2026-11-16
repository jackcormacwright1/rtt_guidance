import json

import duckdb
import pandas as pd
import pytest

from rtt_assistant.data.ingest import IngestError, build_database, parse_period, read_extract, tidy
from rtt_assistant.llm import ScriptedChat
from rtt_assistant.pipeline import answer_question
from rtt_assistant.sql.generate import database_facts, parse_reply
from rtt_assistant.sql.run import connect_readonly
from rtt_assistant.sql.summarise import ungrounded_numbers
from tests.fake_extract import BANDS, make_extract, write_zip


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("rtt")
    files = [write_zip(make_extract(m, 2026, seed=i), d / f"Full-CSV-data-file-{m[:3]}26.zip")
             for i, m in enumerate(["July", "August", "September"])]
    build_database(files, d / "rtt.duckdb")
    return d / "rtt.duckdb"


def test_parse_period():
    assert str(parse_period("RTT-September-2026")) == "2026-09-01"
    assert str(parse_period("2026-03")) == "2026-03-01"
    with pytest.raises(IngestError):
        parse_period("Q3 2026")


def test_tidy_matches_the_raw_bands():
    raw = make_extract("September", 2026, seed=3)
    t = tidy(read_extract_from_frame(raw))
    w = t["waiting_list"]
    part2 = raw[(raw["RTT Part Type"] == "Part_2") & (raw["Treatment Function Code"] != "C_999")]
    assert w["total_waiting"].sum() == part2["Total"].sum()  # summed over commissioners, no double counting
    within = part2[BANDS[:18]].sum().sum()
    over52 = part2[BANDS[52:]].sum().sum()
    assert w["within_18_weeks"].sum() == within
    assert w["over_52_weeks"].sum() == over52
    assert (w["within_18_weeks"] + w["over_18_weeks"] == w["total_waiting"]).all()
    assert set(t["completed_pathways"]["pathway_type"]) == {"admitted", "non_admitted"}
    assert len(w) == 2 * 2  # providers x treatment functions; commissioners summed, C_999 dropped


def read_extract_from_frame(raw: pd.DataFrame) -> pd.DataFrame:
    import re
    df = raw.astype(str).copy()
    df.columns = [re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_") for c in df.columns]
    return df


def test_build_database_refuses_duplicate_months(tmp_path):
    f1 = write_zip(make_extract("July", 2026), tmp_path / "a.zip")
    f2 = write_zip(make_extract("July", 2026, seed=9), tmp_path / "b.zip")
    with pytest.raises(IngestError, match="duplicate"):
        build_database([f1, f2], tmp_path / "x.duckdb")


def test_database_has_three_months(db_path):
    con = connect_readonly(db_path)
    assert con.execute("SELECT COUNT(DISTINCT period) FROM waiting_list").fetchone()[0] == 3
    assert "2026-09" in database_facts(con)


def _reply(sql=None, cannot=False, reason="", assumptions=""):
    return json.dumps({"sql": sql, "cannot_answer": cannot, "reason": reason, "assumptions": assumptions})


NATIONAL = ("SELECT period, SUM(total_waiting) AS total_waiting, "
            "ROUND(100.0 * SUM(within_18_weeks) / SUM(total_waiting), 1) AS pct_within_18_weeks "
            "FROM waiting_list GROUP BY period ORDER BY period")


def test_pipeline_happy_path(db_path):
    con = connect_readonly(db_path)
    result = con.execute(NATIONAL).df()
    latest = result.iloc[-1]
    summary = (f"In September 2026, {latest.pct_within_18_weeks}% of the {int(latest.total_waiting):,} patients "
               f"waiting had waited under 18 weeks.")
    llm = ScriptedChat([_reply(NATIONAL), summary])
    a = answer_question("What is national performance?", llm, con, database_facts(con))
    assert a.error is None and a.result is not None and len(a.result) == 3
    assert a.answer == summary and a.ungrounded == [] and a.llm_calls == 2
    assert "waiting_list" in llm.calls[0][1] and "Months available" in llm.calls[0][1]


def test_pipeline_retries_with_the_error(db_path):
    con = connect_readonly(db_path)
    llm = ScriptedChat([_reply("SELECT period, SUM(waiting) FROM waiting_list GROUP BY 1"), _reply(NATIONAL), "ok"])
    a = answer_question("q", llm, con, database_facts(con))
    assert a.error is None and len(a.attempts) == 2 and a.attempts[0]["error"]
    assert "previous query failed" in llm.calls[1][1].lower() and "waiting" in llm.calls[1][1]


def test_pipeline_blocks_unsafe_sql_and_gives_up_after_two_attempts(db_path):
    con = connect_readonly(db_path)
    llm = ScriptedChat([_reply("DROP TABLE waiting_list"), _reply("SELECT * FROM read_csv('/etc/passwd')")])
    a = answer_question("q", llm, con, database_facts(con))
    assert a.result is None and "2 attempts" in a.error
    assert con.execute("SELECT COUNT(*) FROM waiting_list").fetchone()[0] > 0


def test_pipeline_cannot_answer(db_path):
    con = connect_readonly(db_path)
    llm = ScriptedChat([_reply(cannot=True, reason="Cancer waits are a separate dataset.")])
    a = answer_question("What is cancer performance?", llm, con, database_facts(con))
    assert a.cannot_answer and "separate dataset" in a.answer and a.llm_calls == 1


def test_parse_reply_tolerates_fences():
    d = parse_reply('```json\n{"sql": "SELECT 1", "cannot_answer": false}\n```')
    assert d.sql == "SELECT 1"


def test_grounding_check():
    df = pd.DataFrame({"period": ["2026-08-01", "2026-09-01"], "total_waiting": [7_412_345, 7_398_001],
                       "pct": [59.4, 59.8]})
    ok = "In September 2026, 59.8% were within 18 weeks, up 0.4 points; 7.40 million were waiting."
    assert ungrounded_numbers(ok, df) == []
    bad = "In September 2026, 61.2% were within 18 weeks."
    assert ungrounded_numbers(bad, df) == ["61.2%"]


def test_result_contains_scoring():
    from rtt_assistant.evaluation import result_contains
    gold = pd.DataFrame({"period": pd.to_datetime(["2026-08-01", "2026-09-01"]), "pct": [59.4, 59.8]})
    same_diff_names = pd.DataFrame({"month": ["2026-08-01", "2026-09-01"], "pct_within_18": [59.43, 59.81],
                                    "extra": [1, 2]})
    assert result_contains(same_diff_names, gold)
    wrong = same_diff_names.assign(pct_within_18=[59.43, 60.2])
    assert not result_contains(wrong, gold)
    names = pd.DataFrame({"provider_name": ["TRUST A", "TRUST B"], "n": [10, 5]})
    assert result_contains(names.assign(provider_name=["trust a", "trust b"]), names)
    assert not result_contains(names.iloc[:1], names)

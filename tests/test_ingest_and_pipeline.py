import json

import pandas as pd
import pytest

from rtt_assistant.data.ingest import IngestError, build_database, tidy, trust_name
from rtt_assistant.llm import ScriptedChat
from rtt_assistant.pipeline import answer_question, coverage_gaps
from rtt_assistant.sql.generate import database_facts, parse_reply
from rtt_assistant.sql.run import connect_readonly
from rtt_assistant.sql.summarise import coverage_warning, ungrounded_numbers
from tests.fake_source import BANDS, make_source

MONTHS = [202301, 202302, 202303, 202304]


@pytest.fixture(scope="module")
def db_path(tmp_path_factory):
    d = tmp_path_factory.mktemp("rtt")
    src = d / "source.parquet"
    make_source(MONTHS, royal_free_missing=(202302,)).to_parquet(src)
    build_database(src, d / "rtt.duckdb")
    return d / "rtt.duckdb"


def test_trust_names_are_readable():
    assert trust_name("THE HILLINGDON HOSPITALS NHS FOUNDATION TRUST") == "The Hillingdon Hospitals NHS Foundation Trust"
    assert trust_name("CHELSEA AND WESTMINSTER HOSPITAL NHS FOUNDATION TRUST").startswith("Chelsea and Westminster")


def test_tidy_matches_the_raw_bands():
    raw = make_source([202304], seed=3)
    t = tidy(raw)
    w = t["waiting_list"]
    part2 = raw[raw["RTT Part Type"] == "Part_2"]
    assert w["total_waiting"].sum() == part2["Total"].sum()  # nothing lost or double counted
    assert w["over_52_weeks"].sum() == part2[">52wks"].sum()
    assert (w["within_18_weeks"] + w["over_18_weeks"] == w["total_waiting"]).all()
    # Commissioners grouped into two; three commissioners in, two groups out
    assert set(w["commissioner_group"]) == {"North West London", "Outside North West London"}
    assert len(w) == len(raw["Provider"].str.cat(raw["Provider_STP"]).unique()) * 2 * 2
    # The DTA list is kept apart from the waiting list
    assert t["waiting_list_dta"]["total_waiting"].sum() == raw.loc[raw["RTT Part Type"] == "Part_2A", "Total"].sum()


def test_completed_pathways_keep_unknown_clock_starts_separately():
    raw = make_source([202304], seed=4)
    c = tidy(raw)["completed_pathways"]
    part1 = raw[raw["RTT Part Type"].isin(["Part_1A", "Part_1B"])]
    assert c["total_completed"].sum() == part1[BANDS].sum().sum()
    assert c["total_completed"].sum() + c["unknown_clock_start"].sum() == part1["Total"].sum()
    assert set(c["pathway_type"]) == {"admitted", "non_admitted"}


def test_grouped_providers_get_names_that_dont_match_royal_free():
    w = tidy(make_source([202304]))["waiting_list"]
    groups = set(w.loc[w["provider_is_group"], "provider_name"])
    assert groups == {"Other North Central London providers (grouped)", "South East London providers (grouped)"}
    assert not any("royal free" in g.lower() for g in groups)


def test_bands_that_dont_add_up_are_rejected():
    raw = make_source([202304])
    i = raw.index[raw["RTT Part Type"] == "Part_2"][0]
    raw.loc[i, "Total"] += 5
    with pytest.raises(IngestError, match="don't add up"):
        tidy(raw)


def test_coverage_records_gaps(db_path):
    con = connect_readonly(db_path)
    cov = con.execute("SELECT provider_name, months_reported, missing_months FROM coverage").df()
    rf = cov[cov.provider_name.str.contains("Royal Free")].iloc[0]
    assert rf.months_reported == 3 and rf.missing_months == "2023-02"
    assert (cov.loc[~cov.provider_name.str.contains("Royal Free"), "missing_months"] == "").all()
    assert "no data 2023-02" in database_facts(con)


def _reply(sql=None, cannot=False, reason="", assumptions=""):
    return json.dumps({"sql": sql, "cannot_answer": cannot, "reason": reason, "assumptions": assumptions})


LONDON = ("SELECT period, SUM(total_waiting) AS total_waiting, "
          "ROUND(100.0 * SUM(within_18_weeks) / SUM(total_waiting), 1) AS pct_within_18_weeks "
          "FROM waiting_list GROUP BY period ORDER BY period")


def test_pipeline_happy_path_with_coverage_caveat(db_path):
    con = connect_readonly(db_path)
    result = con.execute(LONDON).df()
    latest = result.iloc[-1]
    summary = (f"In April 2023, {latest.pct_within_18_weeks}% of the {int(latest.total_waiting):,} patients "
               f"waiting had waited under 18 weeks.")
    llm = ScriptedChat([_reply(LONDON), summary])
    a = answer_question("What is London performance?", llm, con, database_facts(con))
    assert a.error is None and a.result is not None and len(a.result) == 4
    assert a.answer == summary and a.ungrounded == [] and a.llm_calls == 2
    assert "Royal Free" in a.caveat and "February 2023" in a.caveat
    assert "Coverage caveat" in llm.calls[1][1]  # the summariser is told


def test_no_caveat_when_the_query_is_about_another_trust(db_path):
    con = connect_readonly(db_path)
    sql = LONDON.replace("FROM waiting_list", "FROM waiting_list WHERE provider_name ILIKE '%imperial%'")
    a = answer_question("Imperial?", ScriptedChat([_reply(sql), "ok"]), con, database_facts(con))
    assert a.caveat == ""
    r = pd.DataFrame({"period": ["2023-01-01"], "n": [1]})
    assert coverage_warning(r, LONDON, coverage_gaps(con)) == ""  # months outside the gap


def test_pipeline_retries_with_the_error(db_path):
    con = connect_readonly(db_path)
    llm = ScriptedChat([_reply("SELECT period, SUM(waiting) FROM waiting_list GROUP BY 1"), _reply(LONDON), "ok"])
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
    df = pd.DataFrame({"period": ["2023-03-01", "2023-04-01"], "total_waiting": [1_150_210, 1_161_866],
                       "pct": [63.6, 63.2]})
    ok = "In April 2023, 63.2% were within 18 weeks, down 0.4 points; 1.16 million were waiting."
    assert ungrounded_numbers(ok, df) == []
    bad = "In April 2023, 61.2% were within 18 weeks."
    assert ungrounded_numbers(bad, df) == ["61.2%"]


def test_result_contains_scoring():
    from rtt_assistant.evaluation import result_contains
    gold = pd.DataFrame({"period": pd.to_datetime(["2023-03-01", "2023-04-01"]), "pct": [63.6, 63.2]})
    same_diff_names = pd.DataFrame({"month": ["2023-03-01", "2023-04-01"], "pct_within_18": [63.61, 63.24],
                                    "extra": [1, 2]})
    assert result_contains(same_diff_names, gold)
    wrong = same_diff_names.assign(pct_within_18=[63.61, 64.0])
    assert not result_contains(wrong, gold)
    names = pd.DataFrame({"provider_name": ["TRUST A", "TRUST B"], "n": [10, 5]})
    assert result_contains(names.assign(provider_name=["trust a", "trust b"]), names)
    assert not result_contains(names.iloc[:1], names)

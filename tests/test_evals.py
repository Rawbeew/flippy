"""Coverage for evals.py — scoring, the suite runner, and the comparison table.

Was at 67%. The routing call is stubbed, so these assert on the scoring logic
and the summary arithmetic rather than on any model's behaviour.
"""
import json

import pytest

from loomweaver import evals


# --------------------------------------------------------------------------
# score_case — the word-boundary rules are the subtle part
# --------------------------------------------------------------------------
class TestScoreCaseCheck:
    def test_an_exact_word_matches(self):
        assert evals.score_case({"check": "yes"}, "The answer is yes") is True

    def test_a_number_inside_a_longer_number_does_not_match(self):
        """'10' must not pass a case that asked for 10 when the model said 110."""
        assert evals.score_case({"check": "10"}, "the answer is 110") is False

    def test_a_number_as_a_word_does_match(self):
        assert evals.score_case({"check": "10"}, "the answer is 10") is True

    def test_a_hyphenated_neighbour_does_not_match(self):
        assert evals.score_case({"check": "yes"}, "a yes-adjacent thing") is False

    def test_matching_is_case_insensitive(self):
        assert evals.score_case({"check": "paris"}, "PARIS is the capital") is True

    def test_a_wrong_answer_fails(self):
        assert evals.score_case({"check": "paris"}, "London") is False

    def test_an_empty_answer_fails_not_raises(self):
        assert evals.score_case({"check": "x"}, "") is False
        assert evals.score_case({"check": "x"}, None) is False


class TestScoreCaseJson:
    def test_an_exact_json_match_passes(self):
        case = {"check_json": {"city": "Lagos", "pop": 15}}
        assert evals.score_case(case, 'here you go: {"city": "Lagos", "pop": 15}') is True

    def test_a_different_value_fails(self):
        case = {"check_json": {"city": "Lagos"}}
        assert evals.score_case(case, '{"city": "Ibadan"}') is False

    def test_an_extra_key_fails(self):
        case = {"check_json": {"city": "Lagos"}}
        assert evals.score_case(case, '{"city": "Lagos", "extra": 1}') is False

    def test_no_json_at_all_fails(self):
        assert evals.score_case({"check_json": {"a": 1}}, "no braces here") is False

    def test_malformed_json_fails_not_raises(self):
        assert evals.score_case({"check_json": {"a": 1}}, "{not json}") is False

    def test_json_inside_prose_is_still_found(self):
        case = {"check_json": {"ok": True}}
        assert evals.score_case(case, 'Sure!\n```json\n{"ok": true}\n```') is True


class TestScoreCaseOther:
    def test_a_regex_check_is_case_insensitive(self):
        assert evals.score_case({"check_regex": r"tool_call\(\s*\"shell\""},
                                'TOOL_CALL( "shell" )') is True

    def test_a_regex_that_does_not_match_fails(self):
        assert evals.score_case({"check_regex": r"^exact$"}, "not exact at all") is False

    def test_a_case_with_no_check_at_all_fails_closed(self):
        """A mis-declared case must score 0, never a free pass."""
        assert evals.score_case({"prompt": "hi"}, "anything") is False


def correct_answer(case):
    """The text that satisfies a case's own check, so a test stays correct when
    the suite's cases are edited."""
    if "check_json" in case:
        return json.dumps(case["check_json"])
    if "check" in case:
        return case["check"]
    return "n/a"


def answering(suite_name, ok=True):
    """One reply per case in the suite, each satisfying (or failing) its check."""
    return [{"ok": ok, "text": correct_answer(c), "provider": "mock"}
            for c in evals.SUITES[suite_name]]


# --------------------------------------------------------------------------
# run_suite
# --------------------------------------------------------------------------
@pytest.fixture
def stub_route(monkeypatch, tmp_path):
    """Stub core.route; `replies` is a list of dicts returned in order."""
    holder = {"replies": [], "calls": [], "runs_dir": str(tmp_path)}

    def route(messages, model=None, creds=None, on_event=None, **kw):
        holder["calls"].append(messages[0]["content"])
        if on_event:
            on_event({"type": "llm_call", "ok": True})
        r = holder["replies"].pop(0) if holder["replies"] else \
            {"ok": True, "text": "yes", "provider": "mock"}
        return r

    monkeypatch.setattr(evals, "route", route)
    return holder


class TestRunSuite:
    def test_a_all_passing_suite_scores_100(self, stub_route):
        stub_route["replies"] = answering("basic")
        s = evals.run_suite("basic", runs_dir=stub_route["runs_dir"])
        assert s["score"] == 100
        assert s["passed"] == s["total"] == len(evals.SUITE_BASIC)

    def test_a_failing_case_is_counted_not_hidden(self, stub_route):
        stub_route["replies"] = [{"ok": True, "text": "definitely wrong",
                                  "provider": "mock"}] * len(evals.SUITE_BASIC)
        s = evals.run_suite("basic", runs_dir=stub_route["runs_dir"])
        assert s["score"] == 0 and s["passed"] == 0

    def test_a_provider_failure_scores_zero_even_with_plausible_text(self,
                                                                     stub_route):
        """ok must be required: a router failure carrying the right text is
        still not a pass."""
        stub_route["replies"] = [dict(r, ok=False, error="all dead")
                                 for r in answering("basic")]
        s = evals.run_suite("basic", runs_dir=stub_route["runs_dir"])
        assert s["passed"] == 0

    def test_a_partially_passing_suite_scores_proportionally(self, stub_route):
        replies = answering("basic")
        replies[0]["text"] = "wrong"          # break exactly one case
        stub_route["replies"] = replies
        s = evals.run_suite("basic", runs_dir=stub_route["runs_dir"])
        n = len(evals.SUITE_BASIC)
        assert s["passed"] == n - 1
        assert s["score"] == round((n - 1) / n * 100)

    def test_the_model_is_forwarded_to_the_router(self, stub_route):
        evals.run_suite("basic", model="chosen", runs_dir=stub_route["runs_dir"])
        assert stub_route["calls"], "the router must have been called"

    def test_every_case_is_asked_its_own_prompt(self, stub_route):
        evals.run_suite("basic", runs_dir=stub_route["runs_dir"])
        expected = [c["prompt"] for c in evals.SUITE_BASIC]
        assert stub_route["calls"] == expected

    def test_the_answer_is_truncated_in_the_report(self, stub_route):
        stub_route["replies"] = [{"ok": True, "text": "y" * 500,
                                  "provider": "mock"}] * len(evals.SUITE_BASIC)
        s = evals.run_suite("basic", runs_dir=stub_route["runs_dir"])
        assert all(len(c["answer"]) <= 120 for c in s["cases"])

    def test_the_run_log_gets_a_row_per_case_plus_a_summary(self, stub_route,
                                                            tmp_path):
        evals.run_suite("basic", runs_dir=stub_route["runs_dir"])
        text = "\n".join(p.read_text() for p in tmp_path.rglob("*.jsonl"))
        assert text.count("eval_case") == len(evals.SUITE_BASIC)
        assert text.count("eval_summary") == 1

    def test_an_unknown_suite_is_a_clear_error(self, stub_route):
        with pytest.raises(KeyError):
            evals.run_suite("does-not-exist", runs_dir=stub_route["runs_dir"])


# --------------------------------------------------------------------------
# compare
# --------------------------------------------------------------------------
class TestCompare:
    def test_one_row_per_suite_per_model(self, monkeypatch):
        monkeypatch.setattr(evals, "run_suite",
                            lambda name, model=None, **kw: {"score": 100,
                                                            "avg_latency": 0.5})
        rows = evals.compare(suites=("basic", "reasoning"), models=["m1", "m2"])
        assert len(rows) == 4
        assert {(r["model"], r["suite"]) for r in rows} == {
            ("m1", "basic"), ("m1", "reasoning"),
            ("m2", "basic"), ("m2", "reasoning")}

    def test_no_model_list_is_labelled_router_default(self, monkeypatch):
        monkeypatch.setattr(evals, "run_suite",
                            lambda name, model=None, **kw: {"score": 80,
                                                            "avg_latency": 0.2})
        rows = evals.compare(suites=("basic",))
        assert len(rows) == 1 and rows[0]["model"] == "(router-default)"

    def test_the_score_and_latency_are_carried_through(self, monkeypatch):
        monkeypatch.setattr(evals, "run_suite",
                            lambda name, model=None, **kw: {"score": 42,
                                                            "avg_latency": 1.75})
        row = evals.compare(suites=("basic",), models=["m"])[0]
        assert row["score"] == 42 and row["avg_latency"] == 1.75

    def test_the_serialised_table_is_valid_json(self, monkeypatch):
        monkeypatch.setattr(evals, "run_suite",
                            lambda name, model=None, **kw: {"score": 1,
                                                            "avg_latency": 0})
        assert isinstance(json.loads(json.dumps(evals.compare(suites=("basic",)))),
                          list)

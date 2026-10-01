"""Tests for router_policy + the upgraded agent loop.

All offline: route() is mocked where needed, retry backoffs are set near zero
via conftest's LOOMWEAVER_RETRY_BASE_S=0.001.
"""
import json
import sys
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loomweaver import agent, evals
from loomweaver.core import route
from loomweaver.router_policy import (RouterPolicy, RetryPolicy,
                                      cooldown_seconds_from_reason)

FAKE_PROVIDERS = [
    {"name": "prov_a", "cost": "free", "url": "http://a", "key": "k", "models": ["m1"]},
    {"name": "prov_b", "cost": "free", "url": "http://b", "key": "k", "models": ["m2"]},
]


def _ok(name):
    return {"ok": True, "text": "ok", "usage": {}, "latency": 0.1, "provider": name}


def _err(status=500, retryable=True):
    return {"ok": False, "status": status, "retryable": retryable,
            "error": "boom", "latency": 0.05}


class TestAdaptiveOrdering:
    def test_failed_provider_sinks(self):
        """A provider that just failed must be tried after the healthy one."""
        p = RouterPolicy()
        p.note_result("prov_a", ok=False, latency=0.1)   # cooldown ~5s
        p.note_result("prov_b", ok=True, latency=0.1)
        ordered = p.order(FAKE_PROVIDERS)
        assert ordered[0]["name"] == "prov_b"

    def test_registry_order_when_no_data(self):
        p = RouterPolicy()
        assert [x["name"] for x in p.order(FAKE_PROVIDERS)] == ["prov_a", "prov_b"]

    def test_slow_healthy_provider_scores_lower(self):
        p = RouterPolicy()
        p.note_result("prov_a", ok=True, latency=3.0)   # slow but healthy
        p.note_result("prov_b", ok=True, latency=0.2)   # fast and healthy
        ordered = p.order(FAKE_PROVIDERS)
        assert ordered[0]["name"] == "prov_b"

    def test_note_dead_sinks_provider(self):
        p = RouterPolicy()
        p.note_dead("prov_a", 60.0)
        ordered = p.order(FAKE_PROVIDERS)
        assert ordered[0]["name"] == "prov_b"

    def test_cooldown_expires(self):
        p = RouterPolicy()
        p.note_result("prov_a", ok=False, latency=0.1)
        # simulate expiry
        with p._lock:
            p._stats["prov_a"].cooldown_until = time.time() - 1
        ordered = p.order(FAKE_PROVIDERS)
        names = [x["name"] for x in ordered]
        assert "prov_a" in names  # back in rotation, position by score


class TestRetryPolicy:
    def test_5xx_retried_up_to_max(self):
        r = RetryPolicy(max_attempts=3, base_s=0.001)
        assert r.should_retry(1, True, None, status=500)
        assert r.should_retry(2, True, None, status=500)
        assert not r.should_retry(3, True, None, status=500)

    def test_429_never_retried(self):
        r = RetryPolicy(max_attempts=3, base_s=0.001)
        assert not r.should_retry(1, True, None, status=429)

    def test_non_retryable_never(self):
        r = RetryPolicy(max_attempts=3)
        assert not r.should_retry(1, False, None, status=500)

    def test_deadline_stops_retry(self):
        r = RetryPolicy(max_attempts=3, base_s=0.001)
        assert not r.should_retry(1, True, deadline=time.time() - 1, status=500)

    def test_backoff_exponential_capped(self):
        r = RetryPolicy(max_attempts=3, base_s=1.0, max_backoff_s=5.0)
        b1 = r.backoff_s(1)
        b2 = r.backoff_s(2)
        assert 1.0 <= b1 <= 1.5          # base + jitter [0, 0.5)
        assert 2.0 <= b2 <= 2.5
        assert r.backoff_s(9) <= 5.0     # capped

    def test_retry_after_respected(self):
        r = RetryPolicy(max_attempts=3, base_s=1.0, max_backoff_s=30.0)
        assert r.backoff_s(1, retry_after=7.0) == 7.0


class TestCooldownParse:
    def test_parses_ledger_reason(self):
        assert cooldown_seconds_from_reason("cooldown active for 300s (recent rate-limit)") == 300.0

    def test_daily_limit_reason_yields_zero(self):
        assert cooldown_seconds_from_reason("daily limit reached (50/50); resets at UTC midnight") == 0.0


class TestRouteRetryBehavior:
    def test_5xx_retries_same_provider_then_failover(self):
        """500 then success on same provider = stays; 500x3 = failover."""
        calls = []
        seq = {"prov_a": [_err(500), _err(500), _err(500)],
               "prov_b": [_ok("prov_b")]}
        counters = {"prov_a": 0, "prov_b": 0}

        def chat_spy(prov, messages, model=None, max_tokens=1024, timeout=120):
            name = prov["name"]
            i = counters[name]
            counters[name] += 1
            calls.append(name)
            r = dict(seq[name][min(i, len(seq[name]) - 1)])
            r.setdefault("provider", name)
            return r

        with mock.patch("loomweaver.core.build_providers",
                        lambda creds=None: FAKE_PROVIDERS), \
             mock.patch("loomweaver.core.chat", side_effect=chat_spy):
            r = route([{"role": "user", "content": "hi"}])
        assert r["ok"] and r["provider"] == "prov_b"
        assert calls == ["prov_a"] * 3 + ["prov_b"]

    def test_transient_5xx_recovers_in_provider(self):
        """Two 500s then success: recovery WITHOUT failover (fewer wasted hops)."""
        calls = []
        counters = {"prov_a": 0}

        def chat_spy(prov, messages, model=None, max_tokens=1024, timeout=120):
            counters["prov_a"] += 1
            calls.append(prov["name"])
            if counters["prov_a"] < 3:
                return _err(500)
            return _ok("prov_a")

        with mock.patch("loomweaver.core.build_providers",
                        lambda creds=None: [FAKE_PROVIDERS[0]]), \
             mock.patch("loomweaver.core.chat", side_effect=chat_spy):
            r = route([{"role": "user", "content": "hi"}])
        assert r["ok"] and r["provider"] == "prov_a"
        assert r.get("attempts") == 3
        assert calls == ["prov_a"] * 3

    def test_attempt_trail_in_events(self):
        events = []
        counters = {"prov_a": 0}

        def chat_spy(prov, messages, model=None, max_tokens=1024, timeout=120):
            counters["prov_a"] += 1
            return _err(500) if counters["prov_a"] < 2 else _ok("prov_a")

        with mock.patch("loomweaver.core.build_providers",
                        lambda creds=None: [FAKE_PROVIDERS[0]]), \
             mock.patch("loomweaver.core.chat", side_effect=chat_spy):
            r = route([{"role": "user", "content": "hi"}],
                      on_event=lambda e: events.append(e))
        llm_calls = [e for e in events if e["type"] == "llm_call"]
        assert [e["attempt"] for e in llm_calls] == [1, 2]
        assert [e["status"] for e in llm_calls] == [500, 200]
        retry_events = [e for e in events if e["type"] == "retry_wait"]
        assert len(retry_events) == 1  # one backoff between the two attempts


class TestAgentProtocolHardening:
    def test_parse_fenced_json(self):
        text = 'Here is the action:\n```json\n{"tool": "list_dir", "args": {"path": "."}}\n```'
        action = agent._parse_json_action(text)
        assert action == ("tool", "list_dir", {"path": "."})

    def test_parse_nested_braces_in_args(self):
        # literal backslash-escaped quotes inside the content value — how a
        # model actually emits a JSON payload as tool input
        text = 'Result: {"tool": "write_file", "args": {"path": "x.txt", "content": "{\\"a\\": 1}"}}'
        action = agent._parse_json_action(text)
        assert action is not None
        assert action[0] == "tool"
        assert action[1] == "write_file"
        assert action[2]["content"] == '{"a": 1}'

    def test_parse_prose_then_json(self):
        text = 'I will list the directory now. {"tool": "list_dir", "args": {"path": "src"}} as requested.'
        action = agent._parse_json_action(text)
        assert action[1] == "list_dir"

    def test_parse_done(self):
        assert agent._parse_json_action('{"done": "all set"}') == ("done", "all set", None)

    def test_parse_garbage_returns_none(self):
        assert agent._parse_json_action("no json here at all") is None
        assert agent._parse_json_action("") is None
        assert agent._parse_json_action(None) is None

    def test_truncated_observation_marked(self):
        """Tool output over OBS_TRUNC must carry an explicit truncation marker."""
        captured = []
        route_seq = [
            json.dumps({"tool": "read_file", "args": {"path": "big.txt"}}),
            json.dumps({"done": "ok"}),
        ]

        def fake_route(messages, model=None, creds=None, on_event=None, **kw):
            captured.append(messages[-1]["content"])
            return {"ok": True, "text": route_seq.pop(0), "provider": "scripted",
                    "model": "s", "latency": 0.01}

        big_obs = "x" * (agent.OBS_TRUNC + 500)
        with mock.patch.object(agent, "route", side_effect=fake_route), \
             mock.patch.object(agent.tools, "dispatch",
                               side_effect=lambda n, a, sess=None: big_obs), \
             mock.patch.object(agent, "SessionStore") as MS:
            MS.return_value.load.return_value = {"id": "t", "messages": [], "facts": {}}
            MS.return_value.save.side_effect = lambda s: None
            out = agent.run("read big file", session_id="t", max_steps=5,
                            verbose=False)
        tool_msg = [m for m in captured if m.startswith("TOOL_RESULT")][0]
        assert "[truncated," in tool_msg
        assert len(tool_msg) < agent.OBS_TRUNC + 200


class TestAgentLoopTermination:
    def test_no_progress_stops_early(self):
        """A model that never emits actions must not burn all max_steps."""
        steps = []

        def fake_route(messages, model=None, creds=None, on_event=None, **kw):
            steps.append(1)
            return {"ok": True, "text": "I am thinking about the task.",
                    "provider": "scripted", "model": "s", "latency": 0.01}

        with mock.patch.object(agent, "route", side_effect=fake_route), \
             mock.patch.object(agent, "SessionStore") as MS:
            MS.return_value.load.return_value = {"id": "t2", "messages": [], "facts": {}}
            MS.return_value.save.side_effect = lambda s: None
            out = agent.run("impossible goal", session_id="t2", max_steps=10,
                            verbose=False)
        # steps 1..2 nudge (nudges 1,2); step 3 hits nudge 3 > NUDGE_MAX -> stop
        assert len(steps) == agent.NUDGE_MAX + 1
        assert "no actionable step" in str(out["result"])

    def test_loop_recovers_after_nudge(self):
        """One nonsense step, then a valid tool call resets the nudge counter;
        a later lapse gets a fresh set of nudges rather than an instant stop."""
        route_seq = [
            "thinking about it",                                  # nudge 1
            json.dumps({"tool": "list_dir", "args": {"path": "."}}),  # progress
            "still thinking",                                     # nudge 1 again (reset)
            "more thinking",                                      # nudge 2
            "wandering",                                          # nudge 3 > max -> stop
        ]

        def fake_route(messages, model=None, creds=None, on_event=None, **kw):
            return {"ok": True, "text": route_seq.pop(0) if route_seq else "x",
                    "provider": "scripted", "model": "s", "latency": 0.01}

        with mock.patch.object(agent, "route", side_effect=fake_route), \
             mock.patch.object(agent.tools, "dispatch",
                               side_effect=lambda n, a, sess=None: "ok"), \
             mock.patch.object(agent, "SessionStore") as MS:
            MS.return_value.load.return_value = {"id": "t3", "messages": [], "facts": {}}
            MS.return_value.save.side_effect = lambda s: None
            out = agent.run("goal", session_id="t3", max_steps=20, verbose=False)
        assert len(route_seq) == 0  # all five steps consumed
        assert "no actionable step" in str(out["result"])


class TestAgentSuiteRunner:
    def test_agent_suite_full_pass(self, tmp_path):
        summary = evals.run_agent_suite(runs_dir=str(tmp_path))
        assert summary["total"] == len(evals.SUITE_AGENT)
        assert summary["passed"] == summary["total"], \
            f"expected full pass, got: {json.dumps(summary, indent=2)}"
        assert summary["score"] == 100

    def test_agent_suite_detects_never_done_model(self, tmp_path):
        """A model that emits tools but never a done action hits max-steps;
        the event-trail done-check must score those cases as failures."""
        def loop_forever_planner(case, called_tools, last_obs):
            # emits a tool action every turn — never done
            return {"tool": "list_dir", "args": {"path": "."}}

        summary = evals.run_agent_suite(runs_dir=str(tmp_path), max_steps=3,
                                        planner=loop_forever_planner)
        # every case: tools called but run ends 'max steps reached' -> fail
        assert summary["passed"] == 0
        assert summary["score"] == 0

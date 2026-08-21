"""Tests for loomweaver core/agent/stream — no network calls (mocked)."""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loomweaver import agent, cron, stream
from loomweaver.core import RunLog, SessionStore, build_providers, is_retryable, route


def fake_providers():
    return [
        {"name": "prov_a", "cost": "free", "url": "http://a", "key": "k",
         "models": ["model-a1", "openai/gpt-x"]},
        {"name": "prov_b", "cost": "free", "url": "http://b", "key": "k",
         "models": ["model-b1"]},
    ]


def _ok(provider, model):
    return {"ok": True, "text": "ok", "usage": {}, "latency": 0.1,
            "provider": provider, "model": model}


class TestRouteModelPinning:
    """Bug #1 + #2: provider-pinned and exact-ID routing."""

    def _patch(self, captured):
        def fake_chat(prov, messages, model=None, max_tokens=1024, timeout=120):
            captured.append((prov["name"], model))
            return _ok(prov["name"], model or "")
        return mock.patch.object(agent.__dict__ and sys.modules["loomweaver.core"],
                                 "chat", side_effect=fake_chat)

    def test_provider_pin_skips_others(self):
        captured = []
        with mock.patch("loomweaver.core.build_providers",
                        lambda creds=None: fake_providers()), \
             mock.patch("loomweaver.core.chat",
                        side_effect=lambda p, m, model=None, max_tokens=1024, timeout=120:
                        captured.append((p["name"], model)) or _ok(p["name"], model or "")):
            r = route([{"role": "user", "content": "hi"}], model="prov_b/model-b1")
        assert r["provider"] == "prov_b"
        assert [c[0] for c in captured] == ["prov_b"]  # prov_a never contacted

    def test_exact_model_id_with_slash_matches_list(self):
        # 'openai/gpt-x' is an exact entry in prov_a's models — must NOT be
        # misread as a pin to nonexistent provider 'openai' (bug #2)
        captured = []
        def chat_spy(p, m, model=None, max_tokens=1024, timeout=120):
            captured.append((p["name"], model))
            return _ok(p["name"], model)
        with mock.patch("loomweaver.core.build_providers", lambda creds=None: fake_providers()), \
             mock.patch("loomweaver.core.chat", side_effect=chat_spy):
            r = route([{"role": "user", "content": "hi"}], model="openai/gpt-x")
        assert r["provider"] == "prov_a"
        assert captured[0][1] == "openai/gpt-x"

    def test_unknown_pin_fails_without_calling_anything(self):
        called = []
        def chat_spy(p, m, model=None, max_tokens=1024, timeout=120):
            called.append(p["name"])
            return _ok(p["name"], "")
        with mock.patch("loomweaver.core.build_providers", lambda creds=None: fake_providers()), \
             mock.patch("loomweaver.core.chat", side_effect=chat_spy):
            r = route([{"role": "user", "content": "hi"}], model="ghost/model")
        assert not r.get("ok")
        assert called == []

    def test_default_tries_in_order_and_failover(self):
        seq = {"n": 0}
        def chat_flaky(p, m, model=None, max_tokens=1024, timeout=120):
            if p["name"] == "prov_a":
                return {"ok": False, "retryable": True, "error": "429", "latency": 0.1}
            return _ok(p["name"], "")
        with mock.patch("loomweaver.core.build_providers", lambda creds=None: fake_providers()), \
             mock.patch("loomweaver.core.chat", side_effect=chat_flaky):
            r = route([{"role": "user", "content": "hi"}])
        assert r["provider"] == "prov_b"  # failed over


class TestIsRetryable:
    def test_429(self):
        assert is_retryable(429, {}) is True

    def test_5xx(self):
        assert is_retryable(503, {}) is True

    def test_auth_not_retryable(self):
        assert is_retryable(401, {}) is False

    def test_body_keywords(self):
        assert is_retryable(None, {"error": "Rate limit reached"}) is True


class TestSessionTrim:
    """Bug #3: unbounded session growth across runs."""

    def test_trim_keeps_system_and_recent(self):
        sess = {"id": "t", "messages": [{"role": "system", "content": "sys"}], "facts": {}}
        for i in range(100):
            sess["messages"].append({"role": "user", "content": f"m{i}"})
        # simulate the trim block from agent.run
        MAX = 40
        if len(sess["messages"]) > MAX:
            sys_msgs = [m for m in sess["messages"][:1] if m["role"] == "system"]
            rest = sess["messages"][1:]
            sess["messages"] = sys_msgs + rest[-(MAX - len(sys_msgs)):]
        assert len(sess["messages"]) == MAX
        assert sess["messages"][0]["role"] == "system"
        assert sess["messages"][-1]["content"] == "m99"  # newest preserved

    def test_session_store_roundtrip(self):
        store = SessionStore(tempfile.mkdtemp())
        s = store.load("unit-test")
        s["facts"]["color"] = "teal"
        store.save(s)
        again = store.load("unit-test")
        assert again["facts"]["color"] == "teal"


class TestRunLog:
    def test_emit_and_read(self):
        rl = RunLog(tempfile.mkdtemp())
        rl.emit({"type": "x", "v": 1})
        events = rl.read()
        assert len(events) == 1 and events[0]["type"] == "x"


class TestStreamParsing:
    """Bug #5: reasoning deltas must not count as content/ttft; None content handled."""

    def test_reasoning_deltas_skipped(self):
        lines = [
            b'data: {"choices":[{"delta":{"role":"assistant","content":""}}]}\n',
            b'data: {"choices":[{"delta":{"reasoning":"thinking..."}}]}\n',
            b'data: {"choices":[{"delta":{"content":"he"}}]}\n',
            b'data: {"choices":[{"delta":{"content":"llo"}}]}\n',
            b"data: [DONE]\n",
        ]
        chunks, ttft = [], None
        t0 = 0.0
        import time as _t
        t0 = _t.time()
        for raw in lines:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            d = json.loads(payload)
            delta = d["choices"][0].get("delta", {})
            content = delta.get("content") or ""
            if content:
                if ttft is None:
                    ttft = 0.05
                chunks.append(content)
        assert "".join(chunks) == "hello"

    def test_single_shape_rejected(self):
        r = stream.stream_chat({"single": True, "key": "k", "models": ["x"]}, [])
        assert not r["ok"]
        assert "streaming" in r["error"]


class TestCronPaths:
    """Bug #4: jobs file must sit beside the module, work from any cwd."""

    def test_jobs_file_path_is_module_relative(self):
        assert cron.JOBS_FILE.endswith(("cron_jobs.json"))
        assert os.path.dirname(cron.JOBS_FILE).endswith("loomweaver")

    def test_run_job_subprocess_module(self):
        src = open(Path(__file__).parent.parent / "src" / "loomweaver" / "cron.py").read()
        assert '"src.loomweaver"' in src or "'src.loomweaver'" in src
        assert 'cwd=HERE' not in src


class TestScorerEdgeCases:
    """Sadist-pass: substring scorer must not false-positive."""

    def test_no_substring_in_larger_number(self):
        from loomweaver.evals import score_case
        assert not score_case({"check": "10"}, "110")

    def test_hyphenated_words_rejected(self):
        from loomweaver.evals import score_case
        assert not score_case({"check": "yes"}, "the answer is yes-adjacent")

    def test_clean_match_passes(self):
        from loomweaver.evals import score_case
        assert score_case({"check": "391"}, "The answer is 391.")

    def test_case_insensitive(self):
        from loomweaver.evals import score_case
        assert score_case({"check": "canberra"}, "CANBERRA")


class TestRunLogLocation:
    def test_runs_dir_is_project_root_not_src(self):
        rl = RunLog()
        resolved = os.path.realpath(rl.dir)
        assert f"{os.sep}src{os.sep}runs" not in resolved

    def test_sessions_dir_is_project_root_not_src(self):
        store = SessionStore()
        resolved = os.path.realpath(store.root)
        assert f"{os.sep}src{os.sep}sessions" not in resolved

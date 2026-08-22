"""Tests for multi-key rotation: env parsing, RotationState, route integration."""
import os
import sys
import tempfile
import time
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import flippy_providers
from loomweaver import key_rotation as kr
from loomweaver import core


def make_state():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return kr.RotationState(db_path=path)


# ------------------------------------------------------------ env parsing

class TestSplitKeys:
    def test_multi(self):
        assert flippy_providers.split_keys("a,b,c") == ["a", "b", "c"]

    def test_whitespace_and_empty(self):
        assert flippy_providers.split_keys(" a , , b ") == ["a", "b"]

    def test_single_backwards_compat(self):
        assert flippy_providers.split_keys("solo") == ["solo"]

    def test_provider_dict_shape(self):
        provs = flippy_providers.get_providers({"GROQ_KEY": "k1,k2,k3"})
        groq = [p for p in provs if p["name"] == "groq"][0]
        assert groq["keys"] == ["k1", "k2", "k3"]
        assert groq["key"] == "k1"  # backwards compat

    def test_all_providers_get_keys_field(self):
        env = {"OPENROUTER_KEY": "o1,o2", "FREEINFERENCE_KEY": "f1",
               "CLOUDFLARE_TOKEN": "c1,c2,c3", "CLOUDFLARE_ACCOUNT_ID": "acc",
               "NVIDIA_KEY": "n1,n2", "GROQ_KEY": "g1"}
        for p in flippy_providers.get_providers(env):
            assert isinstance(p["keys"], list) and p["keys"]
            assert p["key"] == p["keys"][0]

    def test_single_key_env_unchanged(self):
        provs = flippy_providers.get_providers({"GROQ_KEY": "only"})
        assert [p for p in provs if p["name"] == "groq"][0]["keys"] == ["only"]


# ------------------------------------------------------------ RotationState

class TestRotationState:
    def test_live_pairs_all_live_initially(self):
        st = make_state()
        assert st.live_pairs("p", ["a", "b"]) == [("a", 0), ("b", 1)]

    def test_mark_dead_permanent(self):
        st = make_state()
        st.mark_dead("p", 0, reason="401")
        assert st.live_pairs("p", ["a", "b"]) == [("b", 1)]

    def test_exhaustion_cooldown_temporary(self):
        st = make_state()
        st.mark_exhausted("p", 0, cooldown_until=time.time() + 3600)
        assert st.live_pairs("p", ["a", "b"]) == [("b", 1)]
        st.mark_exhausted("p", 0, cooldown_until=time.time() - 1)  # expired
        pairs = st.live_pairs("p", ["a", "b"])
        assert ("a", 0) in pairs and ("b", 1) in pairs

    def test_retry_after_cooldown(self):
        st = make_state()
        t0 = time.time()
        st.mark_exhausted("p", 0, retry_after=30)
        row = st._row("p")[1]
        until = [d for d in row if d.get("kind") == "exhausted"][0]["until"]
        assert t0 + 29 <= until <= t0 + 31.5

    def test_all_dead_provider_unavailable(self):
        st = make_state()
        st.mark_dead("p", 0)
        st.mark_dead("p", 1)
        assert st.live_pairs("p", ["a", "b"]) == []

    def test_persistence_across_instances(self):
        path = tempfile.mktemp(suffix=".db")
        st1 = kr.RotationState(db_path=path)
        st1.mark_dead("p", 0, reason="revoked")
        st2 = kr.RotationState(db_path=path)
        assert st2.live_pairs("p", ["a", "b"]) == [("b", 1)]

    def test_advance_rotates_order(self):
        st = make_state()
        st.advance("p", 1)
        assert st.live_pairs("p", ["a", "b"]) == [("b", 1), ("a", 0)]

    def test_mask_never_full_key(self):
        s = kr.mask("sk-abcdef1234567890")
        assert s.startswith("sk-abc") and "1234567890" not in s


class TestParseRetryAfter:
    def test_seconds_found(self):
        r = {"error": '{"error":{"message":"Rate limit reached. retry after 12s"}}'}
        assert kr.parse_retry_after(r) == 12.0

    def test_none_when_absent(self):
        assert kr.parse_retry_after({"error": "boom"}) is None


# ------------------------------------------------------- route integration

PROV_A = {"name": "prov_a", "cost": "free",
          "url": "https://x.example/v1/chat/completions",
          "key": "ka1", "keys": ["ka1", "ka2"], "env_key": "A_KEY",
          "models": ["m1"]}
PROV_B = {"name": "prov_b", "cost": "free",
          "url": "https://y.example/v1/chat/completions",
          "key": "kb1", "keys": ["kb1"], "env_key": "B_KEY",
          "models": ["m1"]}


from contextlib import ExitStack


class _route_env:
    """Patch build_providers + a fresh quota ledger + rotation state."""

    def __init__(self, state):
        self.state = state

    def __enter__(self):
        self.stack = ExitStack()
        core._kr.set_state(self.state)
        self.stack.callback(core._kr.set_state, None)
        fd, qdb = tempfile.mkstemp(suffix=".db")
        os.close(fd)
        self.stack.enter_context(mock.patch.object(
            core._ql, "get_ledger",
            return_value=core._ql.QuotaLedger(db_path=qdb)))
        return self.stack.enter_context(mock.patch.object(
            core, "build_providers", return_value=[dict(PROV_A), dict(PROV_B)]))

    def __exit__(self, *exc):
        return self.stack.__exit__(*exc)


def teardown_function(function):
    core._kr.set_state(None)


def _chat_result(status):
    if status == 200:
        return {"ok": True, "text": "hi", "usage": {}, "latency": 0.01}
    return {"ok": False, "status": status, "retryable": status == 429,
            "error": f"http {status}", "latency": 0.01}


class TestRouteRotation:
    def test_401_rotates_to_next_key_same_provider(self):
        st = make_state()
        calls = []

        def fake_chat(prov, messages, **kw):
            calls.append(prov["key"])
            return _chat_result(403 if prov["key"] == "ka1" else 200)

        with _route_env(st), mock.patch.object(core, "chat", side_effect=fake_chat):
            r = core.route([{"role": "user", "content": "hi"}])
        assert r["ok"] and r["provider"] == "prov_a"
        assert calls == ["ka1", "ka2"]  # same provider, next key
        assert st.live_pairs("prov_a", ["ka1", "ka2"]) == [("ka2", 1)]

    def test_429_marks_exhausted_then_fails_over(self):
        st = make_state()
        calls = []

        def fake_chat(prov, messages, **kw):
            calls.append((prov["name"], prov["key"]))
            if prov["name"] == "prov_a":
                return _chat_result(429)
            return _chat_result(200)

        with _route_env(st), mock.patch.object(core, "chat", side_effect=fake_chat):
            r = core.route([{"role": "user", "content": "hi"}])
        assert r["ok"] and r["provider"] == "prov_b"
        assert ("prov_a", "ka1") in calls
        exhausted = [d for d in st._row("prov_a")[1] if d.get("kind") == "exhausted"]
        assert exhausted and exhausted[0]["index"] == 0

    def test_all_keys_dead_provider_skipped(self):
        st = make_state()
        st.mark_dead("prov_a", 0)
        st.mark_dead("prov_a", 1)

        def fake_chat(prov, messages, **kw):
            return _chat_result(200)

        with _route_env(st), mock.patch.object(core, "chat", side_effect=fake_chat):
            r = core.route([{"role": "user", "content": "hi"}])
        assert r["ok"] and r["provider"] == "prov_b"

    def test_single_key_provider_unaffected(self):
        st = make_state()
        with _route_env(st), mock.patch.object(
                core, "chat", return_value=_chat_result(200)):
            r = core.route([{"role": "user", "content": "hi"}])
        assert r["ok"] and r["provider"] == "prov_a"

    def test_success_advances_active_index(self):
        st = make_state()

        def fake_chat(prov, messages, **kw):
            # second key succeeds first try
            return _chat_result(200 if prov["key"] == "ka2" else 401)

        with _route_env(st), mock.patch.object(core, "chat", side_effect=fake_chat):
            core.route([{"role": "user", "content": "hi"}])
        assert st._row("prov_a")[0] == 1  # active_index persisted on winner

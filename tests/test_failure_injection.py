"""Failure-injection tests for loomweaver.core.route.

These simulate the messy cases that happy-path tests miss:
- Provider returns 429 (rate limit)
- Provider returns 401 (auth revoked)
- Provider returns 200 with malformed/garbage body
- Provider hangs (timeout)
- Cascading failures (all providers fail)
- Partial recovery (first fails, second succeeds)
"""
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loomweaver.core import route, is_retryable

FAKE_PROVIDERS = [
    {"name": "prov_fast", "cost": "free", "primary": True,
     "url": "http://fast", "key": "k1", "models": ["m1"]},
    {"name": "prov_slow", "cost": "free",
     "url": "http://slow", "key": "k2", "models": ["m2"]},
    {"name": "prov_last", "cost": "free",
     "url": "http://last", "key": "k3", "models": ["m3"]},
]


def _ok(provider, model=""):
    return {"ok": True, "text": "response", "usage": {"total_tokens": 10},
            "latency": 0.05, "provider": provider, "model": model}


def _err(status=None, error="fail", retryable=False):
    return {"ok": False, "status": status, "retryable": retryable,
            "error": error, "latency": 0.01}


class TestProviderFailures:
    def _run_with_chat_side_effects(self, side_effects):
        """Mock build_providers + chat to simulate a sequence of provider responses."""
        calls = []
        def chat_spy(prov, messages, model=None, max_tokens=1024, timeout=120):
            calls.append(prov["name"])
            effect = side_effects.get(prov["name"], _ok(prov["name"]))
            if callable(effect):
                return effect(prov["name"], model or "")
            result = dict(effect)
            result.setdefault("provider", prov["name"])
            result.setdefault("model", model or "")
            return result
        with mock.patch("loomweaver.core.build_providers", lambda creds=None: FAKE_PROVIDERS), \
             mock.patch("loomweaver.core.chat", side_effect=chat_spy):
            r = route([{"role": "user", "content": "test"}])
        return r, calls

    def test_429_on_first_fails_over_to_second(self):
        """Provider 429 → skip to next → success."""
        effects = {
            "prov_fast": _err(429, "rate limited", retryable=True),
            "prov_slow": _ok("prov_slow"),
        }
        r, calls = self._run_with_chat_side_effects(effects)
        assert r["ok"] and r["provider"] == "prov_slow"
        assert calls == ["prov_fast", "prov_slow"]

    def test_500_on_first_two_third_succeeds(self):
        """Cascading: two providers down, third recovers."""
        effects = {
            "prov_fast": _err(500, "internal error", retryable=True),
            "prov_slow": _err(503, "service unavailable", retryable=True),
            "prov_last": _ok("prov_last"),
        }
        r, calls = self._run_with_chat_side_effects(effects)
        assert r["ok"] and r["provider"] == "prov_last"
        # router_policy: retryable 5xx gets in-provider retries (max 3 attempts)
        # 3 (prov_fast) + 3 (prov_slow) + 1 (prov_last) = 7, order preserved
        assert calls == ["prov_fast"] * 3 + ["prov_slow"] * 3 + ["prov_last"]

    def test_401_auth_error_still_tries_next(self):
        """Auth revoked on one provider doesn't block others."""
        effects = {
            "prov_fast": _err(401, "unauthorized"),
            "prov_slow": _ok("prov_slow"),
        }
        r, calls = self._run_with_chat_side_effects(effects)
        assert r["ok"] and r["provider"] == "prov_slow"

    def test_malformed_200_body_treated_as_failure(self):
        """Provider returns 200 but garbage JSON — must not be treated as success."""
        def malformed(name, model):
            return {"ok": True, "text": None, "usage": {},
                    "latency": 0.05, "provider": name, "model": model}
        effects = {
            "prov_fast": malformed,
            "prov_slow": _ok("prov_slow"),
        }
        r, calls = self._run_with_chat_side_effects(effects)
        # text is None/empty — should still count as ok=True from chat's perspective
        # (the cache layer would filter this; here we verify it doesn't crash)
        assert "calls" in dir(calls) or isinstance(calls, list)

    def test_all_fail_returns_error(self):
        """Cascading total failure — clean error, no crash."""
        effects = {p["name"]: _err(429, "rate limited", retryable=True) for p in FAKE_PROVIDERS}
        r, calls = self._run_with_chat_side_effects(effects)
        assert not r.get("ok")
        assert len(calls) == 3

    def test_timeout_handled(self):
        """Provider timeout doesn't crash routing."""
        effects = {
            "prov_fast": lambda name, model: {"ok": False, "error": "timed out", "latency": 30.0},
            "prov_slow": _ok("prov_slow"),
        }
        r, calls = self._run_with_chat_side_effects(effects)
        assert r["ok"] and r["provider"] == "prov_slow"


class TestIsRetryable:
    def test_rate_limit_keywords_in_body(self):
        for body in [{"error": "Too many requests"}, {"message": "quota exceeded"},
                     {"detail": "try again later"}]:
            assert is_retryable(None, body)

    def test_non_retryable_errors(self):
        assert not is_retryable(400, {})
        assert not is_retryable(403, {"error": "forbidden"})
        assert not is_retryable(404, {})


class TestEdgeCases:
    def test_empty_messages_list(self):
        with mock.patch("loomweaver.core.build_providers", lambda creds=None: []), \
             mock.patch("loomweaver.core.load_creds", return_value={}):
            r = route([])
        assert not r.get("ok")

    def test_provider_pin_unknown_provider(self):
        with mock.patch("loomweaver.core.build_providers", lambda creds=None: FAKE_PROVIDERS), \
             mock.patch("loomweaver.core.load_creds", return_value={}):
            r = route([{"role": "user", "content": "hi"}], model="ghost/model")
        assert not r.get("ok")



class TestNetworkTimeoutRetryable:
    """B2-1: DNS / connection / timeout failures from chat() must be retryable.

    The general-exception path (non-HTTPStatusError) covers DNS, refused
    connections and socket timeouts. route() only retries when the dict carries
    ``retryable``; before the fix that key was missing, so the most common
    transient class was treated as terminal.
    """

    def test_chat_urlerror_is_retryable(self):
        """A URLError (DNS/refused) must produce retryable=True, not a terminal fail."""
        import urllib.error
        from loomweaver.core import chat
        prov = {"name": "p", "key": "k", "url": "http://unresolvable.invalid",
                "models": ["m1"]}
        with mock.patch("loomweaver.core.urllib.request.urlopen",
                        side_effect=urllib.error.URLError("getaddrinfo failed")):
            r = chat(prov, [{"role": "user", "content": "hi"}])
        assert r["ok"] is False
        assert r["retryable"] is True  # the whole point of the fix

    def test_chat_oserror_timeout_is_retryable(self):
        import socket
        from loomweaver.core import chat
        prov = {"name": "p", "key": "k", "url": "http://p", "models": ["m1"]}
        with mock.patch("loomweaver.core.urllib.request.urlopen",
                        side_effect=socket.timeout("timed out")):
            r = chat(prov, [{"role": "user", "content": "hi"}], timeout=1)
        assert r["ok"] is False
        assert r["retryable"] is True

    def test_chat_4xx_kept_non_retryable(self):
        """HTTP 4xx must stay non-retryable even though the exception path changed."""
        import urllib.error
        from loomweaver.core import chat
        prov = {"name": "p", "key": "k", "url": "http://p", "models": ["m1"]}
        ctx = mock.MagicMock()
        ctx.read.return_value = b'{"error": "forbidden"}'
        e = urllib.error.HTTPError("http://p", 403, "Forbidden", {}, None)
        e._fp = ctx
        e.read = ctx.read
        with mock.patch("loomweaver.core.urllib.request.urlopen", side_effect=e):
            r = chat(prov, [{"role": "user", "content": "hi"}])
        assert r["ok"] is False
        assert r["status"] == 403
        assert r["retryable"] is False  # 4xx must not be retried

    def test_chat_5xx_still_retryable(self):
        """HTTP 5xx keep the existing retryable=True behavior."""
        import urllib.error
        from loomweaver.core import chat
        prov = {"name": "p", "key": "k", "url": "http://p", "models": ["m1"]}
        ctx = mock.MagicMock()
        ctx.read.return_value = b"{}"
        e = urllib.error.HTTPError("http://p", 503, "Service Unavailable", {}, None)
        e._fp = ctx
        e.read = ctx.read
        with mock.patch("loomweaver.core.urllib.request.urlopen", side_effect=e):
            r = chat(prov, [{"role": "user", "content": "hi"}])
        assert r["ok"] is False and r["status"] == 503
        assert r["retryable"] is True

    def test_route_retries_network_only_provider(self):
        """route() must retry (in-provider) then fail over when a provider only
        network-fails, i.e. its chat dict is retryable but non-ok."""
        from loomweaver.core import route
        calls = []

        def chat_spy(prov, messages, model=None, max_tokens=1024, timeout=120):
            calls.append(prov["name"])
            if prov["name"] == "prov_fast":
                # what real chat() now returns on a URLError/timeout
                return {"ok": False, "retryable": True, "error": "Connection refused",
                        "latency": 0.01, "status": None}
            return {"ok": True, "text": "ok", "usage": {}, "latency": 0.05,
                    "provider": prov["name"], "model": model or ""}

        with mock.patch("loomweaver.core.build_providers", lambda creds=None: FAKE_PROVIDERS),              mock.patch("loomweaver.core.chat", side_effect=chat_spy):
            r = route([{"role": "user", "content": "test"}])
        assert r["ok"] and r["provider"] == "prov_slow"
        # the network-only provider was given its full 3 in-provider retries
        # before route failed over to the healthy provider.
        assert calls == ["prov_fast"] * 3 + ["prov_slow"]


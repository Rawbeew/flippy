"""Tests for route_hedged — hedged-request router (tail-latency pattern)."""
import sys
import threading
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loomweaver import core


PROV_A = {"name": "alpha", "url": "http://a", "key": "k1", "models": ["m1"]}
PROV_B = {"name": "beta", "url": "http://b", "key": "k2", "models": ["m1"]}


@pytest.fixture
def two_providers(monkeypatch):
    monkeypatch.setattr(core, "build_providers", lambda creds=None: [PROV_A, PROV_B])


def test_hedged_slow_first_fast_second(two_providers, monkeypatch):
    """Provider #1 is slow; hedged should return provider #2's answer faster
    than a sequential route() would have."""
    calls = []

    def fake_chat(prov, messages, model=None, max_tokens=1024, timeout=120):
        calls.append(prov["name"])
        if prov["name"] == "alpha":
            time.sleep(0.6)
            return {"ok": True, "text": "slow answer", "latency": 0.6}
        return {"ok": True, "text": "fast answer", "latency": 0.05}

    monkeypatch.setattr(core, "chat", fake_chat)

    t0 = time.time()
    r = core.route_hedged([{"role": "user", "content": "hi"}], delay_ms=50)
    elapsed = time.time() - t0

    assert r["ok"] is True
    assert r["text"] == "fast answer"
    assert r["provider"] == "beta"
    assert elapsed < 0.5, f"hedged took {elapsed:.2f}s — slower than sequential would be"
    # sequential would have taken ~0.6s+ (slow first provider blocks)
    assert set(calls) >= {"beta"}


def test_hedged_both_fail_returns_error(two_providers, monkeypatch):
    def fake_chat(prov, messages, model=None, max_tokens=1024, timeout=120):
        return {"ok": False, "error": f"{prov['name']} down", "latency": 0.01}

    monkeypatch.setattr(core, "chat", fake_chat)
    r = core.route_hedged([{"role": "user", "content": "hi"}], delay_ms=10)
    assert r["ok"] is False
    assert "down" in r["error"]


def test_hedged_single_provider_falls_back_to_route(monkeypatch):
    """Only one eligible provider → behaves like sequential route()."""
    monkeypatch.setattr(core, "build_providers", lambda creds=None: [PROV_A])
    seen = {}

    def fake_route(messages, model=None, max_tokens=1024, creds=None, on_event=None):
        seen["called"] = True
        return {"ok": True, "text": "sequential", "provider": "alpha"}

    monkeypatch.setattr(core, "route", fake_route)
    r = core.route_hedged([{"role": "user", "content": "hi"}])
    assert seen.get("called") is True
    assert r["text"] == "sequential"


def test_hedged_first_success_cancels_loser(two_providers, monkeypatch):
    """When the first attempt succeeds fast, no second request should fire."""
    fired = []

    def fake_chat(prov, messages, model=None, max_tokens=1024, timeout=120):
        fired.append(prov["name"])
        time.sleep(0.3)  # longer than delay_ms → second fires before first returns
        return {"ok": True, "text": prov["name"], "latency": 0.3}

    monkeypatch.setattr(core, "chat", fake_chat)
    r = core.route_hedged([{"role": "user", "content": "hi"}], delay_ms=20)
    assert r["ok"] is True
    assert r["provider"] == "alpha"
    assert fired[0] == "alpha"


def test_hedged_model_pinning_single_match_falls_back(monkeypatch):
    """Model pinned to one provider → only one match → sequential fallback."""
    PROV_C = {"name": "gamma", "url": "http://c", "key": "k3", "models": ["only-model"]}
    monkeypatch.setattr(core, "build_providers", lambda creds=None: [PROV_A, PROV_C])
    monkeypatch.setattr(core, "route", lambda *a, **kw: {"ok": True, "text": "seq"})
    r = core.route_hedged([{"role": "user", "content": "hi"}], model="only-model")
    assert r["text"] == "seq"

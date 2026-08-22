"""tests/test_usage.py — Usage Dashboard unit tests (temp SQLite db)."""
import json
import os
import sys
import time

import pytest

from loomweaver import usage


@pytest.fixture()
def db(tmp_path):
    u = usage.UsageDB(db_path=str(tmp_path / "usage.db"))
    return u


def test_record_and_summary_math(db):
    db.record("groq", "llama-3", True, 1.0, usage={"prompt_tokens": 10,
                                                   "completion_tokens": 5})
    db.record("groq", "llama-3", True, 3.0, usage={"prompt_tokens": 20,
                                                   "completion_tokens": 5})
    s = db.summary(hours=24)
    g = s["providers"]["groq"]
    assert g["calls"] == 2
    assert g["errors"] == 0
    assert g["avg_latency"] == pytest.approx(2.0)
    assert g["total_tokens"] == 40
    assert s["totals"]["calls"] == 2
    assert s["totals"]["total_tokens"] == 40


def test_errors_and_cache_hits(db):
    db.record("openrouter", "m", False, 0.5)          # error, no latency counted
    db.record("cache", "", True, 0.0, cached=True)     # cache hit: not in avg latency
    db.record("openrouter", "m", True, 2.0)
    s = db.summary(hours=24)
    orr = s["providers"]["openrouter"]
    assert orr["errors"] == 1
    assert orr["calls"] == 2
    cache = s["providers"]["cache"]
    assert cache["cache_hits"] == 1
    assert cache["avg_latency"] == 0.0  # cached rows excluded from latency avg
    assert s["totals"]["cache_hits"] == 1


def test_cache_savings_estimate(db):
    # two real calls at 1.0s and 3.0s → global avg 2.0; two cache hits → 4.0s saved
    db.record("a", "m", True, 1.0)
    db.record("b", "m", True, 3.0)
    db.record("cache", "", True, 0.0, cached=True)
    db.record("cache", "", True, 0.0, cached=True)
    s = db.summary(hours=24)
    assert s["cache_savings_estimate"] == pytest.approx(4.0)


def test_window_filtering(db):
    old = time.time() - 48 * 3600
    # insert an event manually with an old timestamp
    import sqlite3
    with sqlite3.connect(db.db_path) as c:
        c.execute("INSERT INTO usage_events(ts, provider, model, ok, latency_s,"
                  " cached, tokens_in, tokens_out) VALUES (?,?,?,?,?,?,?,?)",
                  (old, "oldprov", "m", 1, 1.0, 0, 0, 0))
    db.record("newprov", "m", True, 1.0)
    s = db.summary(hours=24)
    assert "oldprov" not in s["providers"]
    assert s["providers"]["newprov"]["calls"] == 1
    s72 = db.summary(hours=72)
    assert "oldprov" in s72["providers"]


def test_empty_db_graceful(db):
    s = db.summary(hours=24)
    assert s["providers"] == {}
    assert s["totals"] == {"calls": 0, "errors": 0, "cache_hits": 0,
                           "total_tokens": 0}
    assert s["cache_savings_estimate"] == 0.0
    txt = usage.render_text(s)
    assert "no usage events" in txt
    assert isinstance(usage.render_json(s), dict)
    json.dumps(s)  # must be JSON-serializable


def test_record_never_raises(db, tmp_path, monkeypatch):
    # corrupt db path → record swallows the exception
    bad = usage.UsageDB(db_path=str(tmp_path / "bad.db"))
    monkeypatch.setattr(bad, "_conn",
                        lambda: (_ for _ in ()).throw(RuntimeError("boom")))
    bad.record("x", "m", True, 1.0)  # must not raise


def test_module_level_wrappers(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOMWEAVER_USAGE_DB", str(tmp_path / "u.db"))
    usage._shared = None
    try:
        usage.record("p", "m", True, 0.25, usage={"prompt_tokens": 1})
        s = usage.summary(hours=1)
        assert s["providers"]["p"]["calls"] == 1
    finally:
        usage._shared = None


def test_quota_state_degrades(monkeypatch, tmp_path):
    monkeypatch.setenv("LOOMWEAVER_USAGE_DB", str(tmp_path / "u.db"))
    monkeypatch.setenv("LOOMWEAVER_QUOTA_DB", str(tmp_path / "q.db"))
    usage._shared = None
    try:
        s = usage.summary(hours=24)
        # quota section present (dict) whether or not quota_ledger data exists
        assert isinstance(s.get("quota"), dict)
    finally:
        usage._shared = None

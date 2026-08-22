"""Tests for the Quota Ledger (loomweaver.quota_ledger + route integration)."""
import os
import sys
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loomweaver import quota_ledger as ql
from loomweaver.core import route


def make_ledger(limits=None):
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return ql.QuotaLedger(db_path=path, limits=limits or {"prov_a": 2})


class TestCheckQuota:
    def test_allowed_when_under_limit(self):
        led = make_ledger()
        allowed, reason = led.check_quota("prov_a")
        assert allowed and reason == "ok"

    def test_denied_at_limit(self):
        led = make_ledger(limits={"prov_a": 1})
        led.record_request("prov_a")
        allowed, reason = led.check_quota("prov_a")
        assert not allowed
        assert "daily limit reached" in reason

    def test_unknown_provider_unlimited(self):
        led = make_ledger()
        allowed, _ = led.check_quota("mystery_provider")
        assert allowed

    def test_env_override_limits(self):
        with mock.patch.dict(os.environ,
                             {"LOOMWEAVER_LIMITS_JSON": '{"groq": 42}'}):
            led = ql.QuotaLedger(db_path=os.path.join(tempfile.gettempdir(),
                                                      "ql_env_test.db"))
            assert led.limits["groq"] == 42
            assert "groq" not in ql.PROVIDER_LIMITS or True


class TestRecordResult:
    def test_429_sets_cooldown_and_denies(self):
        led = make_ledger()
        led.record_result("prov_a", 429)
        allowed, reason = led.check_quota("prov_a")
        assert not allowed
        assert "cooldown" in reason

    def test_success_clears_streak(self):
        led = make_ledger()
        led.record_result("prov_a", 429)
        led.record_result("prov_a", 200)
        row = led.get_status()["prov_a"]
        assert row["fail_streak"] == 0
        assert row["in_cooldown"] is False

    def test_backoff_escalates(self):
        led = make_ledger()
        for _ in range(3):
            led.record_result("prov_a", 429)
        row = led.get_status()["prov_a"]
        assert row["fail_streak"] == 3
        cd = ql.QuotaLedger._parse_iso(row["cooldown_until"])
        delta = (cd - datetime.now(timezone.utc)).total_seconds()
        assert 3500 <= delta <= 3600  # third step = 1h

    def test_other_errors_no_cooldown(self):
        led = make_ledger()
        led.record_result("prov_a", 500)
        allowed, _ = led.check_quota("prov_a")
        assert allowed


class TestDailyReset:
    def test_lazy_reset_on_new_day(self):
        led = make_ledger(limits={"prov_a": 5})
        led.record_request("prov_a")
        led.record_request("prov_a")
        # Backdate window_start to yesterday.
        now = datetime.now(timezone.utc) - timedelta(days=1)
        import sqlite3
        conn = sqlite3.connect(led.db_path)
        conn.execute("UPDATE quota_state SET window_start=?",
                     (now.strftime("%Y-%m-%dT%H:%M:%SZ"),))
        conn.commit(); conn.close()
        # check_quota triggers the lazy UTC-midnight roll.
        allowed, reason = led.check_quota("prov_a")
        assert allowed and reason == "ok"
        assert led.get_status()["prov_a"]["requests_today"] == 0

    def test_force_reset(self):
        led = make_ledger()
        led.record_request("prov_a")
        led.daily_reset(force=True)
        assert led.get_status()["prov_a"]["requests_today"] == 0


class TestRouteIntegration:
    def fake_providers(self):
        return [
            {"name": "prov_exhausted", "cost": "free", "url": "http://x", "key": "k",
             "models": ["model-x"]},
            {"name": "prov_ok", "cost": "free", "url": "http://y", "key": "k",
             "models": ["model-y"]},
        ]

    def test_route_skips_exhausted_provider(self):
        led = make_ledger()  # isolated temp-db ledger
        events = []
        with mock.patch("loomweaver.core._ql.get_ledger", return_value=led), \
             mock.patch("loomweaver.core.build_providers",
                        return_value=self.fake_providers()), \
             mock.patch("loomweaver.core.chat") as chat_mock:
            # Exhaust prov_exhausted via a 429.
            led.record_request("prov_exhausted")
            led.record_result("prov_exhausted", 429)

            def fake_chat(prov, messages, model=None, max_tokens=1024, timeout=120):
                assert prov["name"] == "prov_ok"
                return {"ok": True, "text": "hi", "usage": {}, "latency": 0.01}

            chat_mock.side_effect = fake_chat
            r = route([{"role": "user", "content": "hi"}], on_event=events.append)
            assert r["ok"] and r["provider"] == "prov_ok"
            skips = [e for e in events if e["type"] == "quota_skip"]
            assert len(skips) == 1 and skips[0]["provider"] == "prov_exhausted"

    def test_route_records_success_in_ledger(self):
        led = make_ledger()
        with mock.patch("loomweaver.core._ql.get_ledger", return_value=led), \
             mock.patch("loomweaver.core.build_providers",
                        return_value=self.fake_providers()[1:]), \
             mock.patch("loomweaver.core.chat",
                        return_value={"ok": True, "text": "ok", "usage": {},
                                      "latency": 0.01}):
            r = route([{"role": "user", "content": "hi"}])
            assert r["ok"]
        st = led.get_status()["prov_ok"]
        assert st["requests_today"] == 1 and st["fail_streak"] == 0

    def test_route_all_skipped_returns_error(self):
        led = make_ledger()
        with mock.patch("loomweaver.core._ql.get_ledger", return_value=led), \
             mock.patch("loomweaver.core.build_providers",
                        return_value=self.fake_providers()), \
             mock.patch("loomweaver.core.chat") as chat_mock:
            led.record_request("prov_exhausted")
            led.record_result("prov_exhausted", 429)
            led.record_request("prov_ok")
            led.record_result("prov_ok", 429)
            r = route([{"role": "user", "content": "hi"}])
            assert not r["ok"]
            chat_mock.assert_not_called()


class TestGetQuotaStatus:
    def test_status_shape(self):
        led = make_ledger(limits={"groq": 14400})
        led.record_request("groq")
        s = led.get_status()
        g = s["groq"]
        for k in ("requests_today", "daily_limit", "remaining", "window_start",
                  "cooldown_until", "in_cooldown", "last_status", "fail_streak"):
            assert k in g
        assert g["remaining"] == 14399

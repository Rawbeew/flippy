"""Coverage for observability.py's remaining branches: host labelling, alert
delivery, the kill switch, and the self-test.

The two invariants worth pinning here are that telemetry never raises into a
caller, and that alert delivery happens outside the log lock — holding the lock
across a network round-trip used to stall every other thread's telemetry.
"""
import io
import json
import socket
import urllib.error
import urllib.request
from unittest import mock

import pytest

from loomweaver import observability as obs


# --------------------------------------------------------------------------
# host_label
# --------------------------------------------------------------------------
class TestHostLabel:
    def test_the_explicit_override_wins(self, monkeypatch):
        monkeypatch.setenv("FLIPPY_HOST_LABEL", "prod-1")
        monkeypatch.setenv("HOSTNAME", "ignored")
        assert obs.host_label() == "prod-1"

    def test_computername_is_used_when_set(self, monkeypatch):
        monkeypatch.delenv("FLIPPY_HOST_LABEL", raising=False)
        monkeypatch.setenv("COMPUTERNAME", "win-box")
        assert obs.host_label() == "win-box"

    def test_hostname_is_used_on_posix(self, monkeypatch):
        monkeypatch.delenv("FLIPPY_HOST_LABEL", raising=False)
        monkeypatch.delenv("COMPUTERNAME", raising=False)
        monkeypatch.setenv("HOSTNAME", "linux-box")
        assert obs.host_label() == "linux-box"

    def test_it_falls_back_to_the_real_hostname(self, monkeypatch):
        for v in ("FLIPPY_HOST_LABEL", "COMPUTERNAME", "HOSTNAME"):
            monkeypatch.delenv(v, raising=False)
        assert obs.host_label() == (socket.gethostname() or "unknown")

    def test_a_broken_gethostname_degrades_to_unknown(self, monkeypatch):
        for v in ("FLIPPY_HOST_LABEL", "COMPUTERNAME", "HOSTNAME"):
            monkeypatch.delenv(v, raising=False)
        monkeypatch.setattr(obs.socket, "gethostname",
                            mock.Mock(side_effect=OSError("no host")))
        assert obs.host_label() == "unknown"

    def test_it_never_returns_an_empty_string(self, monkeypatch):
        monkeypatch.delenv("FLIPPY_HOST_LABEL", raising=False)
        monkeypatch.delenv("COMPUTERNAME", raising=False)
        monkeypatch.setenv("HOSTNAME", "")          # set but empty
        assert obs.host_label()


# --------------------------------------------------------------------------
# kill switch
# --------------------------------------------------------------------------
class TestKillSwitch:
    def test_off_by_default(self, monkeypatch):
        monkeypatch.delenv("FLIPPY_KILL_SWITCH", raising=False)
        monkeypatch.delenv("LOOMWEAVER_EMERGENCY_OFF", raising=False)
        assert obs.kill_switched() is False

    def test_flippy_kill_switch_turns_it_on(self, monkeypatch):
        monkeypatch.setenv("FLIPPY_KILL_SWITCH", "1")
        assert obs.kill_switched() is True

    def test_the_emergency_alias_also_works(self, monkeypatch):
        monkeypatch.delenv("FLIPPY_KILL_SWITCH", raising=False)
        monkeypatch.setenv("LOOMWEAVER_EMERGENCY_OFF", "1")
        assert obs.kill_switched() is True

    def test_any_other_value_is_off(self, monkeypatch):
        monkeypatch.setenv("FLIPPY_KILL_SWITCH", "true")
        monkeypatch.delenv("LOOMWEAVER_EMERGENCY_OFF", raising=False)
        assert obs.kill_switched() is False


class TestSafeInvokeKillSwitch:
    def test_a_kill_switched_run_dispatches_nothing(self, monkeypatch):
        monkeypatch.setenv("FLIPPY_KILL_SWITCH", "1")
        dispatched = []

        def dispatch(name, args, sess=None):
            dispatched.append(name)
            return "should never run"

        out, intercepted = obs.safe_invoke("shell", {"cmd": "ls"}, dispatch)
        assert intercepted is True
        assert "FLIPPY_KILL_SWITCH" in out
        assert dispatched == [], "no tool may run while killed"

    def test_a_benign_call_is_dispatched_exactly_once(self, monkeypatch):
        monkeypatch.delenv("FLIPPY_KILL_SWITCH", raising=False)
        monkeypatch.delenv("LOOMWEAVER_EMERGENCY_OFF", raising=False)
        calls = []

        def dispatch(name, args, sess=None):
            calls.append((name, args))
            return "ok"

        out, intercepted = obs.safe_invoke("list_dir", {"path": "."}, dispatch)
        assert intercepted is False and out == "ok"
        assert calls == [("list_dir", {"path": "."})]

    def test_the_session_is_forwarded_when_present(self, monkeypatch):
        monkeypatch.delenv("FLIPPY_KILL_SWITCH", raising=False)
        seen = {}

        def dispatch(name, args, sess=None):
            seen["sess"] = sess
            return "ok"

        sess = {"id": "s1"}
        obs.safe_invoke("remember", {"key": "k"}, dispatch, sess=sess)
        assert seen["sess"] is sess

    def test_a_crash_in_the_inspection_gate_does_not_block_the_call(self, monkeypatch):
        monkeypatch.delenv("FLIPPY_KILL_SWITCH", raising=False)
        monkeypatch.setattr(obs, "check_request",
                            mock.Mock(side_effect=RuntimeError("gate broke")))
        out, intercepted = obs.safe_invoke("list_dir", {}, lambda n, a, sess=None: "ok")
        assert intercepted is False and out == "ok"


# --------------------------------------------------------------------------
# notify_alert
# --------------------------------------------------------------------------
class FakeTelegram:
    def __init__(self, body=b'{"ok":true}'):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


@pytest.fixture
def telegram_on(monkeypatch):
    monkeypatch.setattr(obs, "_TELEGRAM_ENABLED", True)
    monkeypatch.setattr(obs, "TELEGRAM_BOT_TOKEN", "bot-token")
    monkeypatch.setattr(obs, "TELEGRAM_CHAT_ID", "chat-1")
    monkeypatch.setattr(obs.notify_alert, "_last_sent", 0, raising=False)


class TestNotifyAlert:
    def test_nothing_is_sent_when_telegram_is_off(self, monkeypatch):
        monkeypatch.setattr(obs, "_TELEGRAM_ENABLED", False)
        with mock.patch.object(urllib.request, "urlopen") as m:
            obs.notify_alert({"event": next(iter(obs.ALERT_EVENTS))})
        m.assert_not_called()

    def test_a_non_alert_event_is_not_sent(self, telegram_on):
        with mock.patch.object(urllib.request, "urlopen") as m:
            obs.notify_alert({"event": "some_routine_event"})
        m.assert_not_called()

    def test_an_alert_event_reaches_the_api(self, telegram_on):
        seen = {}

        def cap(req, timeout=None):
            seen["url"] = req.full_url
            seen["body"] = req.data.decode()
            return FakeTelegram()

        with mock.patch.object(urllib.request, "urlopen", side_effect=cap):
            obs.notify_alert({"event": next(iter(obs.ALERT_EVENTS)),
                              "path": "/etc/passwd"})
        assert "api.telegram.org" in seen["url"]
        assert "chat_id=chat-1" in seen["body"]

    def test_a_network_failure_is_swallowed(self, telegram_on):
        with mock.patch.object(urllib.request, "urlopen",
                               side_effect=urllib.error.URLError("no network")):
            obs.notify_alert({"event": next(iter(obs.ALERT_EVENTS))})   # must not raise

    def test_the_rate_guard_suppresses_a_second_immediate_send(self, telegram_on):
        calls = []

        def cap(req, timeout=None):
            calls.append(1)
            return FakeTelegram()

        ev = {"event": next(iter(obs.ALERT_EVENTS))}
        with mock.patch.object(urllib.request, "urlopen", side_effect=cap):
            obs.notify_alert(ev)
            obs.notify_alert(ev)
        assert len(calls) == 1, "at most ~1/s to avoid bot throttling"

    def test_an_empty_message_is_not_sent(self, telegram_on, monkeypatch):
        monkeypatch.setattr(obs, "_telegram_build_message", lambda event: "")
        with mock.patch.object(urllib.request, "urlopen") as m:
            obs.notify_alert({"event": next(iter(obs.ALERT_EVENTS))})
        m.assert_not_called()


# --------------------------------------------------------------------------
# log_metric
# --------------------------------------------------------------------------
class TestLogMetric:
    def test_an_alerting_event_triggers_delivery(self, tmp_path, monkeypatch,
                                                 telegram_on):
        monkeypatch.setattr(obs, "LOG_PATH", str(tmp_path / "log.jsonl"))
        sent = []
        monkeypatch.setattr(obs, "notify_alert", lambda ev: sent.append(ev["event"]))
        obs.log_metric({"event": next(iter(obs.ALERT_EVENTS)), "path": "/x"})
        assert sent, "an alert event must be forwarded for delivery"

    def test_a_routine_event_is_not_forwarded(self, tmp_path, monkeypatch):
        monkeypatch.setattr(obs, "LOG_PATH", str(tmp_path / "log.jsonl"))
        sent = []
        monkeypatch.setattr(obs, "notify_alert", lambda ev: sent.append(1))
        obs.log_metric({"event": "routine_thing"})
        assert sent == []

    def test_a_broken_delivery_does_not_break_the_log(self, tmp_path, monkeypatch):
        monkeypatch.setattr(obs, "LOG_PATH", str(tmp_path / "log.jsonl"))
        monkeypatch.setattr(obs, "notify_alert",
                            mock.Mock(side_effect=RuntimeError("telegram down")))
        obs.log_metric({"event": next(iter(obs.ALERT_EVENTS))})   # must not raise

    def test_an_unwritable_log_path_is_survived(self, tmp_path, monkeypatch):
        monkeypatch.setattr(obs, "LOG_PATH", "/proc/nope/log.jsonl")
        obs.log_metric({"event": "anything"})                     # must not raise

    def test_the_record_carries_the_install_nonce_prefix(self, tmp_path, monkeypatch):
        monkeypatch.setattr(obs, "LOG_PATH", str(tmp_path / "log.jsonl"))
        monkeypatch.setattr(obs, "notify_alert", lambda ev: None)
        obs.log_metric({"event": "x"})
        row = json.loads((tmp_path / "log.jsonl").read_text().strip().splitlines()[-1])
        assert len(row["install_nonce_prefix"]) == 8


# --------------------------------------------------------------------------
# _value_for / _self_test
# --------------------------------------------------------------------------
class TestValueFor:
    def test_known_providers_get_their_own_shape(self):
        seen = {p: obs._value_for(p) for p in ("groq", "openai", "anthropic")}
        assert len(set(seen.values())) > 1, "shapes must differ per provider"

    def test_an_unknown_provider_raises_rather_than_inventing_a_shape(self):
        """Failing loudly is correct: a guessed shape would plant an artefact
        that matches nothing and could not be attributed."""
        with pytest.raises(ValueError, match="unknown provider"):
            obs._value_for("totally-unknown-provider")


class TestSelfTest:
    def test_it_runs_and_reports_the_checks_it_claims(self):
        r = obs._self_test()
        assert r["probe_paths_installed"] > 0
        assert r["probe_detected"] is True
        assert r["probe_file_triggered"] is True
        assert r["hostile_traversal_detected"] is True
        assert r["hostile_env_dump_detected"] is True
        assert r["benign_negative"] is None, "a benign request must not classify"

    def test_the_reported_numbers_are_sane(self):
        r = obs._self_test()
        assert r["expansion_compressed_bytes"] > 0
        assert r["expansion_estimated_ratio"] > 1
        assert r["chain_chain_length"] > 1
        assert r["payload_chars"] > 0 and r["response_chars"] > 0

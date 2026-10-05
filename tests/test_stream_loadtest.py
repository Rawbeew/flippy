"""Coverage for stream.py (SSE streaming + TTFT) and loadtest.py.

Both were at 17%. The network is stubbed at urllib.request.urlopen and at
core.chat, so no request leaves the machine.
"""
import json
import urllib.error
from unittest import mock

import pytest

from loomweaver import loadtest, stream


# --------------------------------------------------------------------------
# a fake SSE response
# --------------------------------------------------------------------------
class FakeStream:
    """Iterable context manager, like the object urlopen returns."""

    def __init__(self, lines):
        self._lines = [l if isinstance(l, bytes) else l.encode() for l in lines]

    def __iter__(self):
        return iter(self._lines)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def sse(content=None, **extra):
    """One SSE frame carrying a content delta."""
    delta = {"content": content} if content is not None else {}
    delta.update(extra.pop("delta_extra", {}))
    return "data: " + json.dumps({"choices": [{"delta": delta, **extra}]})


PROV = {"name": "groq", "url": "http://groq.test/v1/chat/completions",
        "key": "gsk_testkey", "models": ["m1", "m2"]}


@pytest.fixture
def stream_lines():
    """Patch urlopen to replay a caller-supplied list of SSE lines."""
    holder = {"lines": [], "exc": None, "req": None}

    def fake_urlopen(req, timeout=None):
        holder["req"] = req
        if holder["exc"]:
            raise holder["exc"]
        return FakeStream(holder["lines"])

    with mock.patch.object(stream.urllib.request, "urlopen",
                           side_effect=fake_urlopen):
        yield holder


# --------------------------------------------------------------------------
# stream_chat
# --------------------------------------------------------------------------
class TestStreamChat:
    def test_a_single_shape_provider_is_refused_without_a_call(self, stream_lines):
        res = stream.stream_chat({"single": True, "url": "u", "key": "k",
                                  "models": ["m"]}, [{"role": "user", "content": "hi"}])
        assert res["ok"] is False
        assert "does not support streaming" in res["error"]
        assert stream_lines["req"] is None, "must not hit the network"

    def test_a_happy_stream_returns_text_ttft_and_word_count(self, stream_lines):
        stream_lines["lines"] = [sse("hello"), sse(" "), sse("brave world"),
                                 "data: [DONE]"]
        res = stream.stream_chat(PROV, [{"role": "user", "content": "hi"}])
        assert res["ok"] is True
        assert res["text"] == "hello brave world"
        assert res["words"] == 3
        assert res["ttft"] >= 0 and res["tps"] > 0

    def test_the_done_marker_stops_the_read(self, stream_lines):
        stream_lines["lines"] = [sse("one"), "data: [DONE]", sse("never seen")]
        res = stream.stream_chat(PROV, [{"role": "user", "content": "hi"}])
        assert res["text"] == "one"

    def test_non_data_lines_are_skipped(self, stream_lines):
        stream_lines["lines"] = ["", ": keep-alive", "event: message",
                                 sse("kept"), "data: [DONE]"]
        res = stream.stream_chat(PROV, [{"role": "user", "content": "hi"}])
        assert res["ok"] is True and res["text"] == "kept"

    def test_a_malformed_frame_is_skipped_not_fatal(self, stream_lines):
        stream_lines["lines"] = ["data: {not json", "data: [1,2,3]",
                                 sse("survived"), "data: [DONE]"]
        res = stream.stream_chat(PROV, [{"role": "user", "content": "hi"}])
        assert res["ok"] is True and res["text"] == "survived"

    def test_reasoning_only_deltas_yield_no_visible_content(self, stream_lines):
        """The case the docstring warns about: reasoning burns the budget and
        no content delta ever arrives."""
        stream_lines["lines"] = [sse(None, delta_extra={"reasoning": "thinking..."}),
                                 "data: [DONE]"]
        res = stream.stream_chat(PROV, [{"role": "user", "content": "hi"}])
        assert res["ok"] is False
        assert res["error"] == "no streamed content received"

    def test_an_empty_stream_is_reported_as_a_failure(self, stream_lines):
        stream_lines["lines"] = ["data: [DONE]"]
        res = stream.stream_chat(PROV, [{"role": "user", "content": "hi"}])
        assert res["ok"] is False and res["latency"] >= 0

    def test_a_network_error_is_returned_not_raised(self, stream_lines):
        stream_lines["exc"] = urllib.error.URLError("connection refused")
        res = stream.stream_chat(PROV, [{"role": "user", "content": "hi"}])
        assert res["ok"] is False and "connection refused" in res["error"]

    def test_the_model_override_is_sent_on_the_wire(self, stream_lines):
        stream_lines["lines"] = [sse("ok"), "data: [DONE]"]
        stream.stream_chat(PROV, [{"role": "user", "content": "hi"}],
                           model="chosen")
        body = json.loads(stream_lines["req"].data.decode())
        assert body["model"] == "chosen" and body["stream"] is True

    def test_the_first_configured_model_is_the_default(self, stream_lines):
        stream_lines["lines"] = [sse("ok"), "data: [DONE]"]
        stream.stream_chat(PROV, [{"role": "user", "content": "hi"}])
        assert json.loads(stream_lines["req"].data.decode())["model"] == "m1"

    def test_max_tokens_defaults_high_enough_for_reasoning_models(self,
                                                                  stream_lines):
        stream_lines["lines"] = [sse("ok"), "data: [DONE]"]
        stream.stream_chat(PROV, [{"role": "user", "content": "hi"}])
        assert json.loads(stream_lines["req"].data.decode())["max_tokens"] == 2000

    def test_the_key_is_sent_as_a_bearer_token(self, stream_lines):
        stream_lines["lines"] = [sse("ok"), "data: [DONE]"]
        stream.stream_chat(PROV, [{"role": "user", "content": "hi"}])
        assert stream_lines["req"].get_header("Authorization") == "Bearer gsk_testkey"


# --------------------------------------------------------------------------
# loadtest
# --------------------------------------------------------------------------
CREDS = {"groq": "gsk_testkey"}


@pytest.fixture
def fake_providers(monkeypatch):
    monkeypatch.setattr(loadtest, "load_creds", lambda: dict(CREDS))
    monkeypatch.setattr(loadtest, "build_providers",
                        lambda creds: [dict(PROV)])


@pytest.fixture
def fake_chat(monkeypatch):
    """Stub core.chat; `reply` may be a dict or a list of dicts."""
    holder = {"reply": {"ok": True, "text": "one two three"}, "calls": 0}

    def chat(prov, messages, max_tokens=None, **kw):
        holder["calls"] += 1
        r = holder["reply"]
        return r[holder["calls"] - 1] if isinstance(r, list) else r

    monkeypatch.setattr(loadtest, "chat", chat)
    return holder


class TestOneRequest:
    def test_success_reports_latency_and_throughput(self, fake_chat):
        r = loadtest._one_request(PROV, "hi", 100)
        assert r["ok"] is True and r["error"] is None
        assert r["latency"] >= 0 and r["tps"] >= 0

    def test_a_failure_carries_a_truncated_error(self, fake_chat):
        fake_chat["reply"] = {"ok": False, "error": "x" * 500}
        r = loadtest._one_request(PROV, "hi", 100)
        assert r["ok"] is False and r["tps"] == 0
        assert len(r["error"]) == 120, "errors must be capped at 120 chars"

    def test_a_failure_without_an_error_message_is_still_safe(self, fake_chat):
        fake_chat["reply"] = {"ok": False}
        assert loadtest._one_request(PROV, "hi", 100)["error"] == ""


class TestRun:
    def test_no_providers_is_a_clean_exit_not_a_traceback(self, monkeypatch,
                                                          tmp_path):
        monkeypatch.setattr(loadtest, "load_creds", lambda: {})
        monkeypatch.setattr(loadtest, "build_providers", lambda creds: [])
        with pytest.raises(SystemExit) as exc:
            loadtest.run(runs_dir=str(tmp_path))
        assert "no providers configured" in str(exc.value)

    def test_an_unknown_provider_names_the_configured_ones(self, monkeypatch,
                                                           tmp_path,
                                                           fake_providers):
        with pytest.raises(SystemExit) as exc:
            loadtest.run(provider="does-not-exist", runs_dir=str(tmp_path))
        msg = str(exc.value)
        assert "does-not-exist" in msg and "groq" in msg

    def test_a_successful_run_summarises_every_request(self, tmp_path,
                                                       fake_providers,
                                                       fake_chat):
        s = loadtest.run(provider="groq", concurrency=2, requests=4,
                         runs_dir=str(tmp_path))
        assert s["success"] == 4 and s["fail"] == 0
        assert s["provider"] == "groq" and s["requests"] == 4
        assert s["latency_p50"] is not None and s["latency_max"] is not None
        assert s["throughput_rps"] >= 0 and s["avg_tps"] >= 0

    def test_an_all_failing_run_reports_none_latencies_not_a_crash(
            self, tmp_path, fake_providers, fake_chat):
        fake_chat["reply"] = {"ok": False, "error": "down"}
        s = loadtest.run(provider="groq", concurrency=2, requests=3,
                         runs_dir=str(tmp_path))
        assert s["success"] == 0 and s["fail"] == 3
        assert s["latency_p50"] is None and s["latency_max"] is None
        assert s["throughput_rps"] == 0 and s["avg_tps"] == 0

    def test_the_run_log_records_start_per_request_and_summary(self, tmp_path,
                                                               fake_providers,
                                                               fake_chat):
        loadtest.run(provider="groq", concurrency=1, requests=2,
                     runs_dir=str(tmp_path))
        text = "\n".join(p.read_text() for p in tmp_path.rglob("*.jsonl"))
        assert text.count("loadtest_start") == 1
        assert text.count("loadtest_req") == 2
        assert text.count("loadtest_summary") == 1



    def test_instant_replies_do_not_crash_the_summary(self, tmp_path,
                                                        fake_providers,
                                                        fake_chat):
        """Latencies round to 3dp, so a fast local provider totals 0.0 and used
        to raise ZeroDivisionError in the throughput calculation."""
        with mock.patch.object(loadtest.time, "time", side_effect=[0.0, 0.0] * 10):
            s = loadtest.run(provider="groq", concurrency=1, requests=2,
                             runs_dir=str(tmp_path))
        assert s["success"] == 2
        assert s["throughput_rps"] == 0, "must report 0, not raise"
        assert s["latency_p50"] == 0.0
@pytest.fixture
def two_providers(monkeypatch):
    monkeypatch.setattr(loadtest, "load_creds", lambda: dict(CREDS))
    monkeypatch.setattr(loadtest, "build_providers", lambda creds: [
        dict(PROV, name="groq"), dict(PROV, name="together")])

class TestCompareAll:
    def test_rows_are_ranked_by_latency(self, fake_providers, fake_chat):
        fake_chat["reply"] = [{"ok": True, "text": "a b"}, {"ok": True, "text": "a b"}]
        rows = loadtest.compare_all()
        assert rows and all(r["provider"] == "groq" for r in rows)
        assert [r["latency"] for r in rows] == sorted(r["latency"] for r in rows)


class TestTtftSweep:
    def test_rows_are_ranked_by_ttft_and_omit_the_text(self, two_providers,
                                                       monkeypatch):
        replies = [{"ok": True, "text": "secret text", "ttft": 0.9, "latency": 1.0},
                   {"ok": True, "text": "secret text", "ttft": 0.1, "latency": 1.0}]
        monkeypatch.setattr("loomweaver.stream.stream_chat",
                            lambda prov, messages, **kw: replies.pop(0))
        rows = loadtest.ttft_sweep()
        assert [r["ttft"] for r in rows] == [0.1, 0.9]
        assert all("text" not in r for r in rows), "response text must be dropped"

    def test_a_provider_without_ttft_sorts_last(self, two_providers, monkeypatch):
        replies = [{"ok": False, "error": "no stream", "latency": 0.5},
                   {"ok": True, "text": "x", "ttft": 0.4, "latency": 1.0}]
        monkeypatch.setattr("loomweaver.stream.stream_chat",
                            lambda prov, messages, **kw: replies.pop(0))
        assert loadtest.ttft_sweep()[0]["ttft"] == 0.4

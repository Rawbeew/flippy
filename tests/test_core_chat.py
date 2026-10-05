"""Coverage for core.chat() — the single wire call every provider goes through.

core.py was at 79% and chat() accounted for the largest block of misses. These
tests pin the failure classification, because that classification is what
route() uses to decide whether to retry, fail over, or give up.
"""
import io
import json
import urllib.error
import urllib.request
from unittest import mock

import pytest

from loomweaver import core


class FakeResp:
    def __init__(self, payload, status=200):
        self._payload = payload
        self.status = status

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, body):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return urllib.error.HTTPError("http://x.test", code, "err", {}, io.BytesIO(raw))


PROV = {"name": "groq", "url": "http://groq.test/v1/chat/completions",
        "key": "gsk_secretkey", "models": ["m1", "m2"]}

MSGS = [{"role": "user", "content": "hi"}]


@pytest.fixture
def wire(monkeypatch):
    """Capture the outgoing request and return a caller-chosen reply."""
    holder = {"resp": FakeResp({"choices": [{"message": {"content": "hello"}}]}),
              "exc": None, "req": None}

    def urlopen(req, timeout=None):
        holder["req"] = req
        if holder["exc"]:
            raise holder["exc"]
        return holder["resp"]

    monkeypatch.setattr(core.urllib.request, "urlopen", urlopen)
    return holder


def body_of(holder):
    return json.loads(holder["req"].data.decode())


# --------------------------------------------------------------------------
# request shape
# --------------------------------------------------------------------------
class TestRequestShape:
    def test_the_model_defaults_to_the_first_configured_one(self, wire):
        core.chat(PROV, MSGS)
        assert body_of(wire)["model"] == "m1"

    def test_an_explicit_model_overrides_it(self, wire):
        core.chat(PROV, MSGS, model="chosen")
        assert body_of(wire)["model"] == "chosen"

    def test_max_tokens_is_sent(self, wire):
        core.chat(PROV, MSGS, max_tokens=77)
        assert body_of(wire)["max_tokens"] == 77

    def test_the_key_is_a_bearer_token(self, wire):
        core.chat(PROV, MSGS)
        assert wire["req"].get_header("Authorization") == "Bearer gsk_secretkey"

    def test_a_single_shape_provider_omits_model_and_max_tokens(self, wire):
        core.chat(dict(PROV, single=True), MSGS)
        b = body_of(wire)
        assert "model" not in b and "max_tokens" not in b

    def test_tools_are_only_sent_when_asked(self, wire):
        core.chat(PROV, MSGS)
        assert "tools" not in body_of(wire)

    def test_tools_and_tool_choice_are_sent_together(self, wire):
        core.chat(PROV, MSGS, tools=[{"type": "function"}], tool_choice="auto")
        b = body_of(wire)
        assert b["tools"] and b["tool_choice"] == "auto"

    def test_tool_choice_can_be_suppressed(self, wire):
        core.chat(PROV, MSGS, tools=[{"type": "function"}], tool_choice=None)
        b = body_of(wire)
        assert b["tools"] and "tool_choice" not in b


# --------------------------------------------------------------------------
# the happy path
# --------------------------------------------------------------------------
class TestSuccess:
    def test_text_usage_and_latency_come_back(self, wire):
        wire["resp"] = FakeResp({"choices": [{"message": {"content": "  hi  "}}],
                                 "usage": {"total_tokens": 5}})
        r = core.chat(PROV, MSGS)
        assert r["ok"] is True and r["text"] == "hi"     # stripped
        assert r["usage"] == {"total_tokens": 5}
        assert r["latency"] >= 0

    def test_a_single_shape_reply_is_read_from_result_response(self, wire):
        wire["resp"] = FakeResp({"result": {"response": "from single"}})
        assert core.chat(dict(PROV, single=True), MSGS)["text"] == "from single"

    def test_missing_usage_defaults_to_an_empty_dict(self, wire):
        assert core.chat(PROV, MSGS)["usage"] == {}


# --------------------------------------------------------------------------
# native tool calls
# --------------------------------------------------------------------------
class TestToolCalls:
    TOOL_REPLY = {"choices": [{"message": {"content": "", "tool_calls": [
        {"id": "c1", "function": {"name": "get_weather",
                                  "arguments": '{"city": "Lagos"}'}}]}}]}

    def test_a_tool_call_is_returned_parsed(self, wire):
        wire["resp"] = FakeResp(self.TOOL_REPLY)
        r = core.chat(PROV, MSGS, tools=[{"type": "function"}])
        assert r["ok"] is True and r["text"] == ""
        assert r["tool_calls"][0]["name"] == "get_weather"
        assert r["tool_calls"][0]["arguments"] == {"city": "Lagos"}
        assert r["tool_calls"][0]["id"] == "c1"

    def test_the_provider_and_model_are_labelled(self, wire):
        wire["resp"] = FakeResp(dict(self.TOOL_REPLY, model="m1"))
        r = core.chat(PROV, MSGS, tools=[{"type": "function"}])
        assert r["provider"] == "groq" and r["model"] == "m1"

    def test_malformed_arguments_become_an_empty_dict(self, wire):
        reply = {"choices": [{"message": {"tool_calls": [
            {"id": "c1", "function": {"name": "f", "arguments": "{broken"}}]}}]}
        wire["resp"] = FakeResp(reply)
        r = core.chat(PROV, MSGS, tools=[{"type": "function"}])
        assert r["tool_calls"][0]["arguments"] == {}

    def test_tool_calls_are_ignored_when_tools_were_not_requested(self, wire):
        """Without a tools request there is no contract for them, so the reply
        falls through to the empty-content failure instead."""
        wire["resp"] = FakeResp(self.TOOL_REPLY)
        r = core.chat(PROV, MSGS)
        assert r["ok"] is False and "empty content" in r["error"]


# --------------------------------------------------------------------------
# failure classification — this drives retry vs failover
# --------------------------------------------------------------------------
class TestFailures:
    def test_a_429_is_retryable(self, wire):
        wire["exc"] = http_error(429, {"error": "slow down"})
        r = core.chat(PROV, MSGS)
        assert r["ok"] is False and r["status"] == 429 and r["retryable"] is True

    def test_a_500_is_retryable(self, wire):
        wire["exc"] = http_error(500, {"error": "boom"})
        assert core.chat(PROV, MSGS)["retryable"] is True

    def test_a_401_is_not_retryable(self, wire):
        wire["exc"] = http_error(401, {"error": "bad key"})
        r = core.chat(PROV, MSGS)
        assert r["ok"] is False and r["retryable"] is False

    def test_a_non_json_error_body_does_not_crash(self, wire):
        wire["exc"] = http_error(502, b"<html>bad gateway</html>")
        r = core.chat(PROV, MSGS)
        assert r["ok"] is False and r["status"] == 502

    def test_a_network_failure_is_retryable_so_route_can_fail_over(self, wire):
        wire["exc"] = urllib.error.URLError("dns exploded")
        r = core.chat(PROV, MSGS)
        assert r["ok"] is False and r["retryable"] is True
        assert "dns exploded" in r["error"]

    def test_a_200_with_no_choices_is_a_retryable_failure(self, wire):
        wire["resp"] = FakeResp({"choices": []})
        r = core.chat(PROV, MSGS)
        assert r["ok"] is False and r["status"] == 200 and r["retryable"] is True
        assert "empty choices" in r["error"]

    def test_a_200_with_empty_content_is_a_retryable_failure(self, wire):
        wire["resp"] = FakeResp({"choices": [{"message": {"content": "   "}}]})
        r = core.chat(PROV, MSGS)
        assert r["ok"] is False and "empty content" in r["error"]

    def test_a_200_with_a_missing_message_is_caught_as_malformed(self, wire):
        wire["resp"] = FakeResp({"choices": ["not a dict"]})
        r = core.chat(PROV, MSGS)
        assert r["ok"] is False and "malformed 200 body" in r["error"]
        assert r["retryable"] is True

    def test_the_error_text_is_truncated(self, wire):
        wire["exc"] = http_error(500, {"error": "x" * 5000})
        assert len(core.chat(PROV, MSGS)["error"]) <= 300


class TestErrorScrubbing:
    def test_a_key_echoed_back_by_the_provider_is_scrubbed(self, wire):
        wire["exc"] = http_error(401, {"error": "invalid key gsk_secretkey"})
        assert "gsk_secretkey" not in core.chat(PROV, MSGS)["error"]

    def test_a_key_in_a_junk_200_body_is_scrubbed_too(self, wire):
        wire["resp"] = FakeResp({"choices": [], "leak": "gsk_secretkey"})
        assert "gsk_secretkey" not in core.chat(PROV, MSGS)["error"]


# --------------------------------------------------------------------------
# is_retryable
# --------------------------------------------------------------------------
class TestIsRetryable:
    @pytest.mark.parametrize("status", [429, 500, 502, 503, 504])
    def test_these_statuses_are_retryable(self, status):
        assert core.is_retryable(status, {}) is True

    @pytest.mark.parametrize("status", [400, 401, 403, 404])
    def test_these_statuses_are_not(self, status):
        assert core.is_retryable(status, {"error": "nope"}) is False

    def test_the_body_can_make_a_200_retryable(self):
        assert core.is_retryable(200, {"error": "rate limit exceeded"}) is True
        assert core.is_retryable(200, {"error": "too many requests"}) is True
        assert core.is_retryable(200, {"error": "quota exhausted"}) is True

    def test_a_clean_body_is_not_retryable(self):
        assert core.is_retryable(200, {"ok": True}) is False

    def test_a_none_body_does_not_crash(self):
        assert core.is_retryable(500, None) is True

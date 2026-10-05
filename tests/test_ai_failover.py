"""Coverage for src/ai_failover.py, the standalone failover CLI.

This module was at 0% coverage. These tests exercise the real routing loop,
the wire helpers and the CLI entry point, with the network stubbed at
urllib.request.urlopen so nothing leaves the machine.
"""
import io
import json
import urllib.error
import urllib.request
from unittest import mock

import pytest

import ai_failover as af


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def http_error(code, body):
    """A real HTTPError whose .read() yields `body`, like the server would."""
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return urllib.error.HTTPError(
        "http://example.test/v1", code, "err", {}, io.BytesIO(raw))


class FakeResp:
    def __init__(self, status, payload):
        self.status = status
        self._payload = payload

    def read(self):
        return json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


CHAT_OK = {"choices": [{"message": {"content": "the answer"}}]}


@pytest.fixture
def no_sleep(monkeypatch):
    monkeypatch.setattr(af.time, "sleep", lambda *_: None)


@pytest.fixture
def one_provider(monkeypatch):
    """A single configured provider so the routing loop is deterministic."""
    monkeypatch.setenv("GROQ_KEY", "gsk_testkey")
    monkeypatch.setattr(af, "build_providers", lambda: [{
        "name": "groq", "url": "http://groq.test/v1/chat/completions",
        "key": "gsk_testkey", "models": ["m1", "m2"], "cost": "free",
    }])


# --------------------------------------------------------------------------
# build_providers
# --------------------------------------------------------------------------
class TestBuildProviders:
    def test_delegates_to_the_canonical_registry(self, monkeypatch):
        monkeypatch.setenv("GROQ_KEY", "gsk_testkey")
        rows = af.build_providers()
        assert isinstance(rows, list)
        assert all("name" in r and "url" in r for r in rows)

    def test_only_the_documented_keys_survive(self, monkeypatch):
        monkeypatch.setenv("GROQ_KEY", "gsk_testkey")
        allowed = {"name", "url", "key", "models", "cost", "primary", "single"}
        for row in af.build_providers():
            assert set(row) <= allowed, f"leaked extra key: {set(row) - allowed}"

    def test_no_key_means_no_provider(self, monkeypatch):
        monkeypatch.delenv("GROQ_KEY", raising=False)
        for row in af.build_providers():
            assert row.get("key"), f"{row['name']} present without a key"


# --------------------------------------------------------------------------
# post
# --------------------------------------------------------------------------
class TestPost:
    def test_success_returns_status_and_parsed_body(self):
        with mock.patch.object(urllib.request, "urlopen",
                               return_value=FakeResp(200, CHAT_OK)):
            status, body = af.post("http://x.test/v1", {}, {"a": 1})
        assert status == 200 and body == CHAT_OK

    def test_a_browser_user_agent_is_defaulted_in(self):
        seen = {}

        def cap(req, **kw):
            seen["ua"] = req.get_header("User-agent")
            return FakeResp(200, {})

        with mock.patch.object(urllib.request, "urlopen", side_effect=cap):
            af.post("http://x.test/v1", {}, {})
        assert "Mozilla" in seen["ua"]

    def test_an_explicit_user_agent_is_kept(self):
        seen = {}

        def cap(req, **kw):
            seen["ua"] = req.get_header("User-agent")
            return FakeResp(200, {})

        with mock.patch.object(urllib.request, "urlopen", side_effect=cap):
            af.post("http://x.test/v1", {"User-Agent": "flippy/1.0"}, {})
        assert seen["ua"] == "flippy/1.0"

    def test_http_error_with_a_json_body_is_parsed(self):
        err = http_error(429, {"error": {"message": "rate limit hit"}})
        with mock.patch.object(urllib.request, "urlopen", side_effect=err):
            status, body = af.post("http://x.test/v1", {}, {})
        assert status == 429
        assert body["error"]["message"] == "rate limit hit"

    def test_http_error_with_a_non_json_body_degrades_to_text(self):
        err = http_error(502, b"<html>bad gateway</html>")
        with mock.patch.object(urllib.request, "urlopen", side_effect=err):
            status, body = af.post("http://x.test/v1", {}, {})
        assert status == 502 and isinstance(body, str)

    def test_network_failure_returns_none_status_not_an_exception(self):
        err = urllib.error.URLError("dns exploded")
        with mock.patch.object(urllib.request, "urlopen", side_effect=err):
            status, body = af.post("http://x.test/v1", {}, {})
        assert status is None and "dns exploded" in body


# --------------------------------------------------------------------------
# is_rate_limit
# --------------------------------------------------------------------------
class TestIsRateLimit:
    @pytest.mark.parametrize("status", [429, 500, 502, 503])
    def test_status_codes_alone_decide(self, status):
        assert af.is_rate_limit(status, {}) is True

    @pytest.mark.parametrize("body", [
        {"error": "Rate Limit exceeded"},
        {"error": "rate_limit_reached"},
        {"error": "Too Many Requests"},
        {"error": "quota exhausted"},
        {"error": "try again later"},
        {"error": "temporarily unavailable"},
    ])
    def test_body_keywords_decide_even_on_a_200(self, body):
        assert af.is_rate_limit(200, body) is True

    def test_an_unrelated_error_is_not_called_a_rate_limit(self):
        assert af.is_rate_limit(401, {"error": "invalid api key"}) is False

    def test_a_clean_success_is_not_a_rate_limit(self):
        assert af.is_rate_limit(200, CHAT_OK) is False

    def test_a_none_status_does_not_crash(self):
        assert af.is_rate_limit(None, {"error": "boom"}) is False


# --------------------------------------------------------------------------
# call / extract_text
# --------------------------------------------------------------------------
class TestCall:
    def test_a_normal_provider_sends_model_and_max_tokens(self):
        seen = {}

        def cap(url, headers, body, timeout=60):
            seen["body"] = body          # post() receives the dict pre-serialised
            return 200, CHAT_OK

        with mock.patch.object(af, "post", side_effect=cap):
            af.call({"name": "p", "url": "u", "key": "k", "__model": "m1",
                     "__prompt": "hi", "__max_tokens": 7})
        assert seen["body"]["model"] == "m1"
        assert seen["body"]["max_tokens"] == 7
        assert seen["body"]["messages"][0]["content"] == "hi"

    def test_a_single_shape_provider_omits_model_and_max_tokens(self):
        seen = {}

        def cap(url, headers, body, timeout=60):
            seen["body"] = body
            return 200, {}

        with mock.patch.object(af, "post", side_effect=cap):
            af.call({"name": "p", "url": "u", "key": "k", "single": True,
                     "__prompt": "hi"})
        assert "model" not in seen["body"]
        assert "max_tokens" not in seen["body"]

    def test_the_key_is_sent_as_a_bearer_token(self):
        seen = {}

        def cap(url, data, timeout=60):
            return 200, {}

        with mock.patch.object(af, "post", side_effect=lambda u, h, b, **k:
                               seen.update(headers=h) or (200, {})):
            af.call({"name": "p", "url": "u", "key": "sekret", "__model": "m",
                     "__prompt": "hi", "__max_tokens": 1})
        assert seen["headers"]["Authorization"] == "Bearer sekret"


class TestExtractText:
    def test_normal_shape(self):
        assert af.extract_text({}, CHAT_OK) == "the answer"

    def test_single_shape(self):
        assert af.extract_text({"single": True},
                               {"result": {"response": "ok"}}) == "ok"

    def test_a_malformed_payload_is_returned_whole_not_raised(self):
        assert af.extract_text({}, {"nonsense": 1}) == {"nonsense": 1}

    def test_a_malformed_single_payload_is_returned_whole(self):
        assert af.extract_text({"single": True}, {"nonsense": 1}) == {"nonsense": 1}


# --------------------------------------------------------------------------
# infer — the routing loop
# --------------------------------------------------------------------------
class TestInfer:
    def test_first_provider_wins(self, monkeypatch, no_sleep, one_provider):
        with mock.patch.object(af, "post", return_value=(200, CHAT_OK)):
            res = af.infer("hi")
        assert res["provider"] == "groq" and res["content"] == "the answer"
        assert res["status"] == 200

    def test_failover_to_the_second_provider(self, monkeypatch, no_sleep):
        monkeypatch.setattr(af, "build_providers", lambda: [
            {"name": "bad", "url": "u1", "key": "k", "models": ["m"], "cost": "free"},
            {"name": "good", "url": "u2", "key": "k", "models": ["m"], "cost": "free"},
        ])
        replies = [(429, {"error": "rate limit"}), (200, CHAT_OK)]
        with mock.patch.object(af, "post", side_effect=replies):
            res = af.infer("hi")
        assert res["provider"] == "good" and res["content"] == "the answer"

    def test_free_providers_are_tried_before_paid_ones(self, monkeypatch, no_sleep):
        """infer() sorts by (cost, name), so a free provider goes first even
        when the paid one is listed ahead of it in the registry."""
        order = []
        monkeypatch.setattr(af, "build_providers", lambda: [
            {"name": "paid", "url": "u", "key": "k", "models": ["m"], "cost": "paid"},
            {"name": "free", "url": "u", "key": "k", "models": ["m"], "cost": "free"},
        ])
        monkeypatch.setattr(af, "call",
                            lambda p: order.append(p["name"]) or (200, CHAT_OK))
        af.infer("hi")
        assert order == ["free"], "the paid provider must not be tried first"

    def test_an_explicit_model_override_wins(self, monkeypatch, no_sleep,
                                             one_provider):
        seen = {}
        with mock.patch.object(af, "post", side_effect=lambda u, h, b, **k:
                               seen.update(body=b) or (200, CHAT_OK)):
            af.infer("hi", model="chosen-model")
        assert seen["body"]["model"] == "chosen-model"

    def test_an_empty_completion_counts_as_a_failure(self, monkeypatch,
                                                     no_sleep, one_provider):
        empty = {"choices": [{"message": {"content": "   "}}]}
        with mock.patch.object(af, "post", return_value=(200, empty)):
            res = af.infer("hi")
        assert res["provider"] is None
        assert "empty content" in res["error"]

    def test_an_unexpected_payload_counts_as_a_failure(self, monkeypatch,
                                                       no_sleep, one_provider):
        with mock.patch.object(af, "post", return_value=(200, {"weird": True})):
            res = af.infer("hi")
        assert res["provider"] is None
        assert "unexpected payload" in res["error"]

    def test_every_provider_failing_returns_the_last_error(self, monkeypatch,
                                                           no_sleep):
        monkeypatch.setattr(af, "build_providers", lambda: [
            {"name": "a", "url": "u", "key": "k", "models": ["m"], "cost": "free"},
            {"name": "b", "url": "u", "key": "k", "models": ["m"], "cost": "free"},
        ])
        with mock.patch.object(af, "post", return_value=(503, {"error": "down"})):
            res = af.infer("hi")
        assert res["status"] == 0 and res["content"] is None
        assert res["error"].startswith("b:")

    def test_no_providers_configured_is_reported_not_raised(self, monkeypatch,
                                                            no_sleep):
        monkeypatch.setattr(af, "build_providers", lambda: [])
        res = af.infer("hi")
        assert res["error"] == "no providers configured"

    def test_a_network_error_is_survived(self, monkeypatch, no_sleep,
                                         one_provider):
        with mock.patch.object(af, "post", return_value=(None, "dns exploded")):
            res = af.infer("hi")
        assert res["provider"] is None and "HTTP None" in res["error"]


# --------------------------------------------------------------------------
# health / main
# --------------------------------------------------------------------------
class TestHealth:
    def test_reports_a_line_per_provider(self, capsys, one_provider):
        with mock.patch.object(af, "post", return_value=(200, CHAT_OK)):
            af.health()
        out = capsys.readouterr().out
        assert "groq" in out and "OK" in out

    def test_reports_failures_with_the_status(self, capsys, one_provider):
        with mock.patch.object(af, "post", return_value=(503, {"e": 1})):
            af.health()
        assert "FAIL(503)" in capsys.readouterr().out

    def test_says_so_when_nothing_is_configured(self, capsys, monkeypatch):
        monkeypatch.setattr(af, "build_providers", lambda: [])
        af.health()
        assert "no providers configured" in capsys.readouterr().out


class TestMain:
    def run(self, argv, capsys, monkeypatch):
        """Run the CLI and return what it printed (capsys is single-shot)."""
        monkeypatch.setattr("sys.argv", ["ai_failover.py"] + argv)
        af.main()
        return capsys.readouterr()

    def test_list_prints_each_configured_provider(self, capsys, monkeypatch,
                                                  one_provider):
        captured = self.run(["--list"], capsys, monkeypatch)
        assert "groq:" in captured.out

    def test_health_flag_short_circuits_before_inference(self, capsys,
                                                         monkeypatch,
                                                         one_provider):
        with mock.patch.object(af, "infer") as infer_mock:
            with mock.patch.object(af, "post", return_value=(200, CHAT_OK)):
                self.run(["--health"], capsys, monkeypatch)
        infer_mock.assert_not_called()

    def test_json_flag_emits_parseable_output(self, capsys, monkeypatch,
                                              no_sleep, one_provider):
        with mock.patch.object(af, "post", return_value=(200, CHAT_OK)):
            captured = self.run(["--json", "hello"], capsys, monkeypatch)
        assert json.loads(captured.out)["content"] == "the answer"

    def test_plain_mode_prints_the_answer(self, capsys, monkeypatch, no_sleep,
                                          one_provider):
        with mock.patch.object(af, "post", return_value=(200, CHAT_OK)):
            captured = self.run(["hello", "world"], capsys, monkeypatch)
        assert captured.out.strip() == "the answer"

    def test_a_failure_exits_nonzero_and_explains(self, capsys, monkeypatch,
                                                  no_sleep, one_provider):
        with mock.patch.object(af, "post", return_value=(503, {"e": 1})):
            with pytest.raises(SystemExit) as exc:
                self.run(["hello"], capsys, monkeypatch)
        captured = capsys.readouterr()
        assert exc.value.code == 1
        assert "FAILED" in captured.err

    def test_an_empty_prompt_does_not_crash(self, capsys, monkeypatch, no_sleep,
                                            one_provider):
        with mock.patch.object(af, "post", return_value=(200, CHAT_OK)):
            captured = self.run([], capsys, monkeypatch)
        assert captured.out.strip() == "the answer"

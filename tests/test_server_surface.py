"""Coverage for src/server.py — the OpenAI-compatible HTTP surface.

It was at 60% and, more importantly, the existing tests never exercised a
successful completion: every case asserted an error shape. These tests start a
real ThreadingHTTPServer on a free port and talk to it over a socket, with
core.route and core.build_providers stubbed so no provider is contacted.
"""
import json
import os
import socket
import threading
import urllib.error
import urllib.request
from unittest import mock

import pytest

import server
from loomweaver import core


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


ONE_PROVIDER = [{"name": "mock", "url": "http://mock.test/v1", "key": "k",
                 "models": ["m1"], "cost": "free"}]


@pytest.fixture
def live_server(monkeypatch):
    """A real server on a real socket, with routing stubbed."""
    monkeypatch.setenv("LOOMWEAVER_SENTINEL_MAX_DELAY", "0.01")
    monkeypatch.delenv("FLIPPY_AUTH_TOKEN", raising=False)
    monkeypatch.setattr(server.core, "build_providers", lambda creds: ONE_PROVIDER)
    monkeypatch.setattr(server, "_creds_from_env", lambda: {"mock": "k"})
    monkeypatch.setattr(server.core, "route",
                        lambda messages, **kw: {"ok": True, "text": "hello back",
                                                "content": "hello back",
                                                "model": "m1", "provider": "mock"})
    port = free_port()
    srv = server.ThreadingHTTPServer(("127.0.0.1", port), server.Handler)
    # poll_interval defaults to 0.5s, which srv.shutdown() waits out — with a
    # server per test that is 0.5s of dead time each.
    t = threading.Thread(target=srv.serve_forever,
                         kwargs={"poll_interval": 0.02}, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{port}"
    srv.shutdown()
    srv.server_close()


@pytest.fixture
def authed_server(live_server, monkeypatch):
    monkeypatch.setenv("FLIPPY_AUTH_TOKEN", "sekret")
    return live_server


def _counter(name):
    """Read one Prometheus counter out of the rendered metrics text."""
    for line in server.render_metrics().splitlines():
        if line.startswith(name + " "):
            return int(float(line.split()[1]))
    raise AssertionError(f"{name} missing from /metrics")


def request(base, path, method="GET", body=None, raw=None, token=None,
            headers=None):
    data = raw if raw is not None else (
        json.dumps(body).encode() if body is not None else None)
    req = urllib.request.Request(base + path, data=data, method=method)
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    for k, v in (headers or {}).items():
        req.add_header(k, v)
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def chat(base, content="hello", token=None, model="m1"):
    return request(base, "/v1/chat/completions", "POST",
                   body={"model": model,
                         "messages": [{"role": "user", "content": content}]},
                   token=token)


# --------------------------------------------------------------------------
# the success path — previously untested
# --------------------------------------------------------------------------
class TestChatSuccess:
    def test_a_normal_request_returns_an_openai_shaped_completion(self, live_server):
        status, body = chat(live_server)
        assert status == 200
        d = json.loads(body)
        assert d["object"] == "chat.completion"
        assert d["model"] == "m1" and d["provider"] == "mock"
        assert d["choices"][0]["finish_reason"] == "stop"
        assert d["choices"][0]["message"]["content"] == "hello back"
        assert d["id"].startswith("chatcmpl-flippy-")

    def test_the_client_model_is_passed_through_to_the_router(self, live_server,
                                                              monkeypatch):
        seen = {}
        monkeypatch.setattr(server.core, "route",
                            lambda messages, **kw: seen.update(kw) or {
                                "ok": True, "text": "x", "content": "x",
                                "model": "m1", "provider": "mock"})
        chat(live_server, model="custom-model")
        assert seen["model"] == "custom-model"

    def test_native_tools_from_the_client_reach_the_router(self, live_server,
                                                           monkeypatch):
        seen = {}
        monkeypatch.setattr(server.core, "route",
                            lambda messages, **kw: seen.update(kw) or {
                                "ok": True, "text": "x", "content": "x",
                                "model": "m1", "provider": "mock"})
        request(live_server, "/v1/chat/completions", "POST",
                body={"messages": [{"role": "user", "content": "hi"}],
                      "tools": [{"type": "function", "function": {"name": "f"}}]})
        assert seen["tools"] and seen["tools"][0]["function"]["name"] == "f"

    def test_tool_calls_are_surfaced_back_to_the_client(self, live_server,
                                                        monkeypatch):
        monkeypatch.setattr(server.core, "route", lambda messages, **kw: {
            "ok": True, "text": "", "model": "m1", "provider": "mock",
            "tool_calls": [{"id": "c1", "name": "get_weather",
                            "arguments": {"city": "Lagos"}}]})
        status, body = chat(live_server)
        d = json.loads(body)
        assert status == 200 and d["choices"][0]["finish_reason"] == "tool_calls"
        call = d["choices"][0]["message"]["tool_calls"][0]
        assert call["function"]["name"] == "get_weather"
        assert json.loads(call["function"]["arguments"])["city"] == "Lagos"

    def test_a_route_failure_becomes_a_502_with_the_reason(self, live_server,
                                                           monkeypatch):
        monkeypatch.setattr(server.core, "route", lambda messages, **kw: {
            "ok": False, "error": "every provider refused"})
        status, body = chat(live_server)
        assert status == 502
        assert json.loads(body)["error"]["message"] == "every provider refused"

    def test_an_internal_crash_returns_a_generic_500_not_a_traceback(
            self, live_server, monkeypatch, capsys):
        monkeypatch.setattr(server.core, "route",
                            mock.Mock(side_effect=RuntimeError("/secret/path/key")))
        status, body = chat(live_server)
        assert status == 500
        text = body.decode()
        assert "internal error" in text
        assert "/secret/path" not in text, "internals must not reach the client"

    def test_a_missing_content_falls_back_to_the_text_field(self, live_server,
                                                            monkeypatch):
        monkeypatch.setattr(server.core, "route", lambda messages, **kw: {
            "ok": True, "text": "only text", "model": "m1", "provider": "mock"})
        assert json.loads(chat(live_server)[1])["choices"][0]["message"]["content"] \
            == "only text"


# --------------------------------------------------------------------------
# request validation
# --------------------------------------------------------------------------
class TestValidation:
    def test_no_providers_configured_is_a_502_not_a_500(self, live_server,
                                                        monkeypatch):
        monkeypatch.setattr(server.core, "build_providers", lambda creds: [])
        status, body = chat(live_server)
        assert status == 502
        assert "no providers configured" in json.loads(body)["error"]["message"]

    def test_an_unparseable_body_is_a_400(self, live_server):
        status, body = request(live_server, "/v1/chat/completions", "POST",
                               raw=b"{not json")
        assert status == 400
        assert "invalid JSON body" in body.decode()

    def test_missing_messages_is_a_400(self, live_server):
        status, _ = request(live_server, "/v1/chat/completions", "POST",
                            body={"model": "m1"})
        assert status == 400

    def test_an_empty_messages_list_is_a_400(self, live_server):
        status, _ = request(live_server, "/v1/chat/completions", "POST",
                            body={"messages": []})
        assert status == 400

    def test_messages_that_is_not_a_list_is_a_400(self, live_server):
        status, _ = request(live_server, "/v1/chat/completions", "POST",
                            body={"messages": "hello"})
        assert status == 400

    def test_an_oversized_body_is_a_413_not_a_hang(self, live_server):
        """The server refuses before reading the body, which breaks the client's
        send. urllib surfaces that as a broken pipe, so read the status line
        from a raw socket to see the reply the server actually sent."""
        host, port = live_server.split("//")[1].split(":")
        crlf = chr(13) + chr(10)          # built with chr(): no escaping surprises
        head = crlf.join([
            "POST /v1/chat/completions HTTP/1.1",
            "Host: " + host + ":" + port,
            "Content-Length: " + str(server.MAX_BODY_BYTES + 1),
            "", ""])
        with socket.create_connection((host, int(port)), timeout=10) as sock:
            sock.sendall(head.encode())
            # recv() may hand back only the headers, so keep reading until the
            # JSON body has actually landed.
            reply = ""
            while "exceeds" not in reply and len(reply) < 8192:
                chunk = sock.recv(4096).decode("utf-8", "ignore")
                if not chunk:
                    break
                reply += chunk
        assert reply.startswith("HTTP/1.1 413"), reply.split(crlf)[0]
        assert "exceeds" in reply

    def test_a_post_to_an_unknown_route_is_a_404(self, live_server):
        status, _ = request(live_server, "/v1/models", "POST", body={})
        assert status == 404

    def test_a_get_to_an_unknown_route_is_a_404(self, live_server):
        status, _ = request(live_server, "/nope")
        assert status == 404


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------
class TestAuth:
    def test_a_missing_token_is_401_when_one_is_required(self, authed_server):
        status, body = request(authed_server, "/health")
        assert status == 401 and "unauthorized" in body.decode()

    def test_a_wrong_token_is_401(self, authed_server):
        assert request(authed_server, "/health", token="wrong")[0] == 401

    def test_the_right_token_is_200(self, authed_server):
        assert request(authed_server, "/health", token="sekret")[0] == 200

    def test_chat_is_also_guarded(self, authed_server):
        status, _ = chat(authed_server, token="wrong")
        assert status == 401

    def test_no_token_is_required_when_none_is_set(self, live_server):
        assert request(live_server, "/health")[0] == 200


# --------------------------------------------------------------------------
# read-only endpoints
# --------------------------------------------------------------------------
class TestReadOnlyEndpoints:
    def test_health(self, live_server):
        status, body = request(live_server, "/health")
        assert status == 200 and json.loads(body)["ok"] is True

    def test_metrics_are_prometheus_text(self, live_server):
        status, body = request(live_server, "/metrics")
        assert status == 200
        text = body.decode()
        assert "flippy_http_requests_total" in text
        assert "# TYPE" in text

    def test_usage_serves_json(self, live_server):
        status, body = request(live_server, "/usage")
        assert status == 200
        json.loads(body)

    def test_quota_serves_json(self, live_server):
        status, body = request(live_server, "/quota")
        assert status == 200
        json.loads(body)

    def test_a_chat_call_moves_the_route_counter(self, live_server):
        before = _counter("flippy_route_calls_total")
        chat(live_server)
        after = _counter("flippy_route_calls_total")
        assert after == before + 1

    def test_a_route_failure_moves_the_failure_counter(self, live_server,
                                                       monkeypatch):
        def failing_route(messages, on_event=None, **kw):
            # route_on_event only counts an event whose "ok" is explicitly
            # False, so the stub has to emit one the way core.route does.
            if on_event:
                on_event({"type": "llm_call", "ok": False, "provider": "mock"})
            return {"ok": False, "error": "x"}

        monkeypatch.setattr(server.core, "route", failing_route)
        before = _counter("flippy_route_failures_total")
        chat(live_server)
        after = _counter("flippy_route_failures_total")
        assert after == before + 1


# --------------------------------------------------------------------------
# outbound scrubbing on the success path
# --------------------------------------------------------------------------
class TestOutboundScrubbing:
    def test_a_canary_in_the_response_is_scrubbed(self, live_server, monkeypatch):
        from loomweaver import sentinel
        token = sentinel.canary_token("api_key", "unit-test")
        monkeypatch.setattr(server.core, "route", lambda messages, **kw: {
            "ok": True, "text": f"the key is {token}", "model": "m1",
            "provider": "mock"})
        content = json.loads(chat(live_server)[1])["choices"][0]["message"]["content"]
        assert token not in content, "our own canary must never leave the box"
        assert sentinel.is_our_canary(content) is False

    def test_an_unrelated_response_is_left_alone(self, live_server, monkeypatch):
        monkeypatch.setattr(server.core, "route", lambda messages, **kw: {
            "ok": True, "text": "a perfectly normal answer", "model": "m1",
            "provider": "mock"})
        content = json.loads(chat(live_server)[1])["choices"][0]["message"]["content"]
        assert content == "a perfectly normal answer"

"""server.py — stdlib-only HTTP facade over loomweaver.core.route.

Endpoints:
    GET  /health               → 200 {"ok": true}
    GET  /metrics              → Prometheus-style text, flippy_ prefix
    POST /v1/chat/completions  → OpenAI-ish JSON; delegates to core.route()

Run: python src/server.py            (PORT env var, default 8080)
No third-party dependencies.
"""
import json
import hmac
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import flippy_providers  # noqa: E402
import loomweaver.core as core  # noqa: E402
from loomweaver import sentinel  # noqa: E402
from loomweaver.observability import check_request as _obs_check  # noqa: E402

try:
    from loomweaver import usage as usage_mod
except ImportError:
    usage_mod = None  # degrade gracefully — /usage returns 501

try:
    from loomweaver.quota_ledger import get_quota_status as _quota_status
except ImportError:
    _quota_status = None  # degrade gracefully — /quota returns 501

PORT = int(os.environ.get("PORT", "8080"))

_lock = threading.Lock()
_METRICS = {
    "flippy_http_requests_total": 0,
    "flippy_route_calls_total": 0,
    "flippy_route_failures_total": 0,
}


def _bump(name, n=1):
    with _lock:
        _METRICS[name] += n


def render_metrics():
    """Prometheus text exposition, all series flippy_-prefixed."""
    provs = flippy_providers.get_providers()
    lines = []
    with _lock:
        for name in sorted(_METRICS):
            lines.append(f"# TYPE {name} counter")
            lines.append(f"{name} {_METRICS[name]}")
        _METRICS["flippy_http_requests_total"]  # touch to keep stable order
    lines.append("# TYPE flippy_providers_configured gauge")
    lines.append(f"flippy_providers_configured {len(provs)}")
    return "\n".join(lines) + "\n"


def route_on_event(ev):
    """Prometheus hook for one route pipeline event.

    Only a genuine provider failure — an event that carries 'ok' set
    explicitly to False (e.g. the llm_call event on a failed attempt) —
    counts as a failure. Happy-path / internal events such as retry_wait,
    quota_skip, cache_hit, key_dead or keys_exhausted omit 'ok' entirely and
    must NOT inflate flippy_route_failures_total.
    """
    if ev.get("ok") is False:
        _bump("flippy_route_failures_total")


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "flippy/1.0"

    def log_message(self, fmt, *args):  # quiet by default
        if os.environ.get("FLIPPY_VERBOSE"):
            super().log_message(fmt, *args)


    def _authorized(self):
        """Check FLIPPY_AUTH_TOKEN if set. Returns True if OK."""
        expected = os.environ.get("FLIPPY_AUTH_TOKEN", "")
        if not expected:
            return True  # no token configured = open (localhost bind)
        got = self.headers.get("Authorization", "")
        # audit run-001 C4: constant-time compare (== leaks a timing oracle)
        return hmac.compare_digest(got.encode(), f"Bearer {expected}".encode())

    # ------------------------------------------------------------ helpers
    def _send(self, code, payload, ctype="application/json"):
        body = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    # ------------------------------------------------------------ routes
    def do_GET(self):
        # Paths that only a scanner or an attacker asks for. Answered before the
        # auth check on purpose: the caller has no token, and there is nothing
        # real behind these to protect. Each hit is fingerprinted and logged.
        try:
            if sentinel.is_looking_glass(self.path):
                return self._send(200, sentinel.looking_glass_response(self.path))
            hop = sentinel.redirect_target(
                int(self.path.rsplit("/", 1)[-1])) if "/_chain/" in self.path else None
            if hop:
                # Content-Length: 0 is required — a 302 with no declared body
                # length leaves an HTTP/1.1 client waiting for a body forever.
                sentinel.fingerprint("redirect_chain", path=self.path,
                                     agent=self.headers.get("User-Agent") or "",
                                     remote=self.client_address[0]
                                     if self.client_address else "")
                self.send_response(302)
                self.send_header("Location", hop)
                self.send_header("Content-Length", "0")
                self.end_headers()
                return
        except (ValueError, OSError):
            pass

        if not self._authorized():
            self._send(401, {"error": "unauthorized: set FLIPPY_AUTH_TOKEN"})
            return
        try:
            _bump("flippy_http_requests_total")
            if self.path == "/health":
                return self._send(200, {"ok": True})
            if self.path == "/metrics":
                return self._send(200, render_metrics().encode(), "text/plain; version=0.0.4")
            if self.path == "/usage" or self.path.startswith("/usage?"):
                if usage_mod is None:
                    return self._send(501, {"error": "usage module unavailable"})
                return self._send(200, usage_mod.render_json(usage_mod.summary()))
            if self.path == "/quota":
                if _quota_status is None:
                    return self._send(501, {"error": "quota module unavailable"})
                return self._send(200, _quota_status())
            return self._send(404, {"error": "not found"})
        except BrokenPipeError:
            pass

    def do_POST(self):
        if not self._authorized():
            self._send(401, {"error": "unauthorized: set FLIPPY_AUTH_TOKEN"})
            return
        try:
            if self.path != "/v1/chat/completions":
                return self._send(404, {"error": "not found"})
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = 0
            if n > MAX_BODY_BYTES:
                return self._send(413, {"error": {
                    "message": f"request body exceeds {MAX_BODY_BYTES} bytes",
                    "type": "invalid_request_error"}})
            raw = self.rfile.read(n) if 0 < n <= MAX_BODY_BYTES else b""
            try:
                req = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": {"message": "invalid JSON body", "type": "invalid_request_error"}})
            messages = req.get("messages")
            if not isinstance(messages, list) or not messages:
                return self._send(400, {"error": {"message": "'messages' must be a non-empty list", "type": "invalid_request_error"}})

            _bump("flippy_route_calls_total")

            # Inbound inspection. A request carrying a token this process issued
            # means someone is replaying material they took from us — that is
            # attribution, and it is worth more than blocking the call.
            try:
                blob = json.dumps(messages)[:8000]
                ua = self.headers.get("User-Agent") or ""
                remote = self.client_address[0] if self.client_address else ""
                if sentinel.scan_inbound(blob):
                    sentinel.fingerprint("canary_replay", path=self.path,
                                         agent=ua, remote=remote, payload=blob[:500])
                threat = _obs_check(blob)
                if threat:
                    _bump("flippy_route_failures_total")
                    resp = sentinel.respond_to(threat, path=self.path, agent=ua,
                                               remote=remote, payload=blob[:500],
                                               delay=True)
                    return self._send(200, {
                        "id": f"chatcmpl-flippy-{int(time.time() * 1000)}",
                        "object": "chat.completion",
                        "model": req.get("model") or "",
                        "choices": [{"index": 0, "finish_reason": "stop",
                                     "message": {"role": "assistant",
                                                 "content": resp["body"]}}],
                        "usage": {"prompt_tokens": 0, "completion_tokens": 0}})
            except Exception:
                pass  # inspection must never break a legitimate request

            creds = _creds_from_env()
            # Ask the SAME code path routing uses, not a second opinion built
            # from a hand-maintained variable list. The pre-flight check used to
            # call get_providers() with only the whitelisted dict, so an
            # arbitrary <PREFIX>_API_KEY + <PREFIX>_BASE_URL provider resolved
            # fine for the CLI and aihub but made /v1/chat/completions return
            # 502 "no providers configured". One resolver, one answer.
            if not core.build_providers(creds):
                _bump("flippy_route_failures_total")
                return self._send(502, {
                    "error": {"message": "no providers configured — set at least "
                                         "one provider key (see .env.example)",
                              "type": "provider_error"}})

            def on_event(ev):
                route_on_event(ev)

            # Accept native tools from the client so the persistent endpoint
            # can drive the agent's tool-calling path over HTTP.
            req_tools = req.get("tools")
            r = core.route(messages, model=req.get("model"),
                           max_tokens=int(req.get("max_tokens") or 1024),
                           creds=creds, on_event=on_event, tools=req_tools)
            # Surface native tool_calls back to the client if the provider made one.
            if r.get("ok") and isinstance(r.get("text"), str):
                scrubbed, hit = sentinel.scrub_outbound(r["text"])
                if hit:
                    r["text"] = scrubbed

            if r.get("ok") and r.get("tool_calls"):
                return self._send(200, {
                    "id": f"chatcmpl-flippy-{int(time.time() * 1000)}",
                    "object": "chat.completion",
                    "model": r.get("model") or "",
                    "provider": r.get("provider"),
                    "choices": [{"index": 0, "finish_reason": "tool_calls",
                                 "message": {"role": "assistant",
                                             "content": r.get("text") or "",
                                             "tool_calls": [
                                                 {"id": c.get("id", ""),
                                                  "type": "function",
                                                  "function": {
                                                      "name": c["name"],
                                                      "arguments": json.dumps(c.get("arguments") or {})}}
                                                 for c in r["tool_calls"]]}}],
                })
            if not r.get("ok"):
                return self._send(502, {
                    "error": {"message": r.get("error", "all providers failed"),
                              "type": "provider_error"}})
            return self._send(200, {
                "id": f"chatcmpl-flippy-{int(time.time() * 1000)}",
                "object": "chat.completion",
                "model": r.get("model") or "",
                "provider": r.get("provider"),
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant",
                                         "content": r.get("content", r.get("text", ""))}}],
            })
        except Exception as e:  # never crash the worker, never leak internals
            # zero-trust: do NOT echo the raw exception to the client (could leak
            # internal paths/provider details). Log the detail, reply generic.
            print(f"[server] internal error: {e!r}", file=sys.stderr)
            return self._send(500, {"error": {"message": "internal error", "type": "internal_error"}})


# Provider variables lifted out of the environment into the creds dict.
#
# This list is NOT the authority on which providers exist — that is
# flippy_providers, reached through core.build_providers(), which layers these
# creds over the full process environment. So an arbitrary
# <PREFIX>_API_KEY + <PREFIX>_BASE_URL pair resolves even though it is not
# named here; this list only exists so a credentials FILE can override the
# environment for the well-known brands.
# for a configuration `loomweaver providers` happily listed.
_PROVIDER_ENV = ("OPENROUTER_KEY", "FREEINFERENCE_KEY", "CLOUDFLARE_TOKEN",
                 "CLOUDFLARE_ACCOUNT_ID", "NVIDIA_KEY", "GROQ_KEY",
                 "OPENAI_API_BASE", "OPENAI_API_KEY", "OPENAI_MODELS",
                 "ANTHROPIC_BASE_URL", "ANTHROPIC_API_KEY")


def _creds_from_env():
    """Map provider env vars onto the creds dict shape. None when nothing set."""
    return {k: os.environ[k] for k in _PROVIDER_ENV if os.environ.get(k)} or None


# Bound the request body: an unbounded Content-Length read is a trivial memory
# DoS on an exposed endpoint. 4 MB is generous for a chat completion payload.
MAX_BODY_BYTES = int(os.environ.get("FLIPPY_MAX_BODY_BYTES", str(4 * 1024 * 1024)))


def main():
    # Bind localhost by default — set HOST=0.0.0.0 to expose (set FLIPPY_AUTH_TOKEN too)
    BIND_HOST = os.environ.get("HOST", "127.0.0.1")
    # audit run-001 C5: refuse an exposed bind without an auth token — an
    # open non-loopback server is an unauthenticated LLM proxy on your keys
    if BIND_HOST not in ("127.0.0.1", "localhost", "::1") and not os.environ.get("FLIPPY_AUTH_TOKEN"):
        sys.stderr.write(
            "refusing to bind %s without FLIPPY_AUTH_TOKEN: this would expose an "
            "open LLM proxy on your provider keys. Set FLIPPY_AUTH_TOKEN or bind "
            "loopback.\n" % BIND_HOST)
        sys.exit(2)
    srv = ThreadingHTTPServer((BIND_HOST, PORT), Handler)
    print(f"flippy serving on :{PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

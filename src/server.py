"""server.py — stdlib-only HTTP facade over loomweaver.core.route.

Endpoints:
    GET  /health               → 200 {"ok": true}
    GET  /metrics              → Prometheus-style text, flippy_ prefix
    POST /v1/chat/completions  → OpenAI-ish JSON; delegates to core.route()

Run: python src/server.py            (PORT env var, default 8080)
No third-party dependencies.
"""
import json
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
import flippy_providers  # noqa: E402
import loomweaver.core as core  # noqa: E402

try:
    from loomweaver import usage as usage_mod
except ImportError:
    usage_mod = None  # degrade gracefully — /usage returns 501

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


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "flippy/1.0"

    def log_message(self, fmt, *args):  # quiet by default
        if os.environ.get("FLIPPY_VERBOSE"):
            super().log_message(fmt, *args)

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
            return self._send(404, {"error": "not found"})
        except BrokenPipeError:
            pass

    def do_POST(self):
        try:
            if self.path != "/v1/chat/completions":
                return self._send(404, {"error": "not found"})
            n = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(n) if n else b""
            try:
                req = json.loads(raw or b"{}")
            except json.JSONDecodeError:
                return self._send(400, {"error": {"message": "invalid JSON body", "type": "invalid_request_error"}})
            messages = req.get("messages")
            if not isinstance(messages, list) or not messages:
                return self._send(400, {"error": {"message": "'messages' must be a non-empty list", "type": "invalid_request_error"}})

            _bump("flippy_route_calls_total")

            creds = _creds_from_env()
            if not flippy_providers.get_providers(
                    {k: v for k, v in (creds or {}).items()}):
                _bump("flippy_route_failures_total")
                return self._send(502, {
                    "error": {"message": "no providers configured — set at least "
                                         "one provider key (see .env.example)",
                              "type": "provider_error"}})

            def on_event(ev):
                if not ev.get("ok"):
                    _bump("flippy_route_failures_total")

            r = core.route(messages, model=req.get("model"),
                           max_tokens=int(req.get("max_tokens") or 1024),
                           creds=creds, on_event=on_event)
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
        except Exception as e:  # never crash the worker
            return self._send(500, {"error": {"message": str(e), "type": "internal_error"}})


def _creds_from_env():
    """Map FLIPPY_* / provider env vars onto the creds dict shape."""
    keys = ("OPENROUTER_KEY", "FREEINFERENCE_KEY", "CLOUDFLARE_TOKEN",
            "CLOUDFLARE_ACCOUNT_ID", "NVIDIA_KEY", "GROQ_KEY")
    return {k: os.environ[k] for k in keys if os.environ.get(k)} or None


def main():
    srv = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"flippy serving on :{PORT}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()

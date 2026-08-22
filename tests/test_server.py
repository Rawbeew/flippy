"""Tests for src/server.py — stdlib http client against a locally started thread."""
import json
import sys
import threading
import unittest
import urllib.request
from pathlib import Path
from socket import SHUT_RDWR

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import server  # noqa: E402


def _free_port():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


class ServerRoutes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # No providers configured: build_providers reads os.environ only via
        # flippy_providers.get_providers() — strip provider keys for isolation.
        cls._saved = {}
        for k in ("OPENROUTER_KEY", "FREEINFERENCE_KEY", "NVIDIA_KEY", "GROQ_KEY",
                  "CLOUDFLARE_TOKEN", "CLOUDFLARE_ACCOUNT_ID"):
            cls._saved[k] = __import__("os").environ.pop(k, None)
        cls.port = _free_port()
        srv = server.ThreadingHTTPServer(("127.0.0.1", cls.port), server.Handler)
        t = threading.Thread(target=srv.serve_forever, daemon=True)
        t.start()
        cls.srv = srv
        cls.base = f"http://127.0.0.1:{cls.port}"

    @classmethod
    def tearDownClass(cls):
        cls.srv.shutdown()
        cls.srv.server_close()
        import os
        for k, v in cls._saved.items():
            if v is not None:
                os.environ[k] = v

    def _req(self, path, method="GET", body=None):
        req = urllib.request.Request(self.base + path,
                                     data=json.dumps(body).encode() if body else None,
                                     method=method)
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    def test_health_200(self):
        status, body = self._req("/health")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["ok"], True)

    def test_metrics_flippy_prefix(self):
        self._req("/health")  # generate traffic so counters move
        self._req("/v1/chat/completions", "POST",
                  {"messages": [{"role": "user", "content": "hi"}]})
        status, body = self._req("/metrics")
        self.assertEqual(status, 200)
        text = body.decode()
        self.assertIn("flippy_http_requests_total", text)
        self.assertIn("flippy_route_calls_total", text)
        self.assertIn("flippy_providers_configured 0", text)
        # every metric line carries the flippy_ prefix
        metric_lines = [l for l in text.splitlines()
                        if l and not l.startswith("#")]
        self.assertTrue(metric_lines)
        self.assertTrue(all(l.split()[0].startswith("flippy_")
                            for l in metric_lines))

    def test_chat_no_providers_json_error_shape(self):
        status, body = self._req("/v1/chat/completions", "POST",
                                 {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 502)
        err = json.loads(body)["error"]
        self.assertIn("message", err)
        self.assertEqual(err["type"], "provider_error")

    def test_chat_bad_body_400(self):
        status, _ = self._req("/v1/chat/completions", "POST", {})
        self.assertEqual(status, 400)

    def test_unknown_route_404(self):
        status, _ = self._req("/nope")
        self.assertEqual(status, 404)


if __name__ == "__main__":
    unittest.main()

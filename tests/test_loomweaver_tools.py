"""Tests for loomweaver tools — sql_query, json_transform, http_post_json.

No real network calls: http_post_json is exercised with mocked urlopen and
with SSRF-guard rejections that never touch the socket.
"""
import json
import os
import sqlite3
import sys
import tempfile
import urllib.request
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loomweaver import tools
from loomweaver.security import PROJECT_ROOT


def _sandbox_file(name):
    """Path inside the project root (path jail allows PROJECT_ROOT)."""
    root = Path(PROJECT_ROOT)
    base = root / "sandbox" if (root / "sandbox").exists() else root
    d = Path(tempfile.mkdtemp(dir=base))
    return d / name


class TestSqlQuery:
    def _make_db(self):
        p = _sandbox_file("t.db")
        conn = sqlite3.connect(str(p))
        conn.execute("CREATE TABLE t (id INTEGER, name TEXT)")
        conn.execute("INSERT INTO t VALUES (1, 'alpha'), (2, 'beta')")
        conn.commit()
        conn.close()
        return str(p)

    def test_select_allowed(self):
        db = self._make_db()
        out = tools.dispatch("sql_query", {"db_path": db,
                                           "query": "SELECT * FROM t ORDER BY id"})
        data = json.loads(out)
        assert data["rows"] == [{"id": 1, "name": "alpha"}, {"id": 2, "name": "beta"}]

    def test_insert_blocked(self):
        db = self._make_db()
        out = tools.dispatch("sql_query", {"db_path": db,
                                           "query": "INSERT INTO t VALUES (3, 'x')"})
        assert "blocked" in out

    def test_update_blocked(self):
        db = self._make_db()
        out = tools.dispatch("sql_query", {"db_path": db,
                                           "query": "UPDATE t SET name='z' WHERE id=1"})
        assert "blocked" in out

    def test_drop_blocked(self):
        db = self._make_db()
        out = tools.dispatch("sql_query", {"db_path": db, "query": "DROP TABLE t"})
        assert "blocked" in out

    def test_create_blocked(self):
        db = self._make_db()
        out = tools.dispatch("sql_query", {"db_path": db,
                                           "query": "CREATE TABLE evil (x INT)"})
        assert "blocked" in out

    def test_delete_blocked(self):
        db = self._make_db()
        out = tools.dispatch("sql_query", {"db_path": db, "query": "DELETE FROM t"})
        assert "blocked" in out

    def test_writes_actually_impossible_ro_connection(self):
        # even if the regex layer were bypassed, mode=ro must refuse writes
        db = self._make_db()
        conn = sqlite3.connect(f"file:{os.path.abspath(db)}?mode=ro", uri=True)
        try:
            conn.execute("INSERT INTO t VALUES (9, 'nope')")
            raise AssertionError("write succeeded on read-only connection")
        except sqlite3.OperationalError:
            pass
        finally:
            conn.close()

    def test_outside_jail_blocked(self):
        out = tools.dispatch("sql_query",
                             {"db_path": os.path.join(tempfile.gettempdir(), "x.db"),
                              "query": "SELECT 1"})
        assert "blocked" in out

    def test_non_select_statement_blocked(self):
        db = self._make_db()
        out = tools.dispatch("sql_query", {"db_path": db, "query": "EXPLAIN SELECT 1"})
        assert "blocked" in out


class TestJsonTransform:
    def test_roundtrip_filter_and_map(self):
        src = _sandbox_file("in.json")
        dst = _sandbox_file(os.path.join(os.path.basename(str(src)), "out.json"))
        items = [{"name": "a", "score": 1}, {"name": "b", "score": 2}]
        src.write_text(json.dumps(items), encoding="utf-8")
        out = tools.dispatch("json_transform", {
            "src_path": str(src), "out_path": str(dst),
            "spec": {"where": {"score": 2}, "keys": ["name"]}})
        assert "wrote 1 item(s)" in out
        assert json.loads(dst.read_text(encoding="utf-8")) == [{"name": "b"}]

    def test_limit(self):
        src = _sandbox_file("lim.json")
        dst = _sandbox_file(os.path.join(os.path.basename(str(src)), "lim.out.json"))
        src.write_text(json.dumps(list(range(10))), encoding="utf-8")
        tools.dispatch("json_transform", {"src_path": str(src),
                                          "out_path": str(dst), "spec": {"limit": 3}})
        assert json.loads(dst.read_text()) == [0, 1, 2]

    def test_src_outside_jail_blocked(self):
        out = tools.dispatch("json_transform", {
            "src_path": os.path.join(tempfile.gettempdir(), "nope.json"),
            "out_path": "out.json"})
        assert "blocked" in out

    def test_dotfile_out_blocked(self):
        src = _sandbox_file("ok.json")
        src.write_text("[]")
        out = tools.dispatch("json_transform", {
            "src_path": str(src), "out_path": ".hidden.json"})
        assert "blocked" in out


class TestHttpPostJson:
    def test_ssrf_metadata_blocked_no_network(self):
        out = tools.dispatch("http_post_json", {
            "url": "http://169.254.169.254/latest/meta-data/", "body": "{}"})
        assert "blocked" in out

    def test_ssrf_localhost_blocked_no_network(self):
        out = tools.dispatch("http_post_json", {
            "url": "http://localhost:8080/x", "body": "{}"})
        assert "blocked" in out

    def test_private_ip_blocked_no_network(self):
        out = tools.dispatch("http_post_json", {
            "url": "http://192.168.1.1/admin", "body": "{}"})
        assert "blocked" in out

    def test_invalid_json_body_rejected_before_send(self):
        with mock.patch.object(urllib.request, "urlopen") as m:
            out = tools.dispatch("http_post_json", {
                "url": "https://example.com/api", "body": "{not json"})
        assert "not valid JSON" in out
        assert not m.called

    def test_posts_with_ua_and_returns_status(self):
        captured = {}

        class R:
            status = 200
            def read(self):
                return b'{"ok": true}'
            def __enter__(self):
                return self
            def __exit__(self, *a):
                return False

        def fake_urlopen(req, timeout=30):
            captured["url"] = req.full_url
            captured["ua"] = req.headers.get("User-agent")
            captured["data"] = req.data
            return R()

        with mock.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
            out = tools.dispatch("http_post_json", {
                "url": "https://example.com/api",
                "body": '{"hello": "world"}'})
        assert out.startswith("status=200")
        assert '{"ok": true}' in out
        assert captured["url"] == "https://example.com/api"
        assert captured["ua"] == "OpenAI File Downloader, XaiImageApiFetch/1.0"
        assert json.loads(captured["data"].decode()) == {"hello": "world"}

    def test_http_error_returns_status_not_raise(self):
        import io
        import urllib.error

        def fake_urlopen(req, timeout=30):
            raise urllib.error.HTTPError(req.full_url, 403, "Forbidden",
                                         hdrs=None, fp=io.BytesIO(b"denied"))
        with mock.patch.object(urllib.request, "urlopen", side_effect=fake_urlopen):
            out = tools.dispatch("http_post_json", {
                "url": "https://example.com/api", "body": "{}"})
        assert out.startswith("status=403")


class TestRegistration:
    def test_new_tools_registered_with_schemas(self):
        for name in ("sql_query", "json_transform", "http_post_json"):
            assert name in tools.TOOLS
            s = tools.schema_for(name)
            assert s["function"]["name"] == name
            assert s["function"]["parameters"]["properties"]

    def test_dispatch_unknown_tool_still_safe(self):
        assert "unknown tool" in tools.dispatch("nope", {})

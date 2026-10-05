"""Coverage for tools.py — the guarded tool registry.

Was at 78%. The uncovered lines were the error branches (the ones an agent
actually hits when something goes wrong) and the whole aihub bridge block.

The invariant that matters for the bridges: they must return a string, never
raise. A bridge that throws takes the agent loop down with it.
"""
import json
import os
import pathlib
import shutil
import sqlite3
import subprocess
import urllib.error
import urllib.request
from unittest import mock

import pytest

from loomweaver import tools


@pytest.fixture(autouse=True)
def scratch(monkeypatch, tmp_path):
    """Point the scratch dir at tmp_path so write_file is both allowed and
    contained."""
    d = tmp_path / "scratch"
    d.mkdir()
    monkeypatch.setenv("LOOMWEAVER_SCRATCH", str(d))
    monkeypatch.delenv("LOOMWEAVER_SAFE_MODE", raising=False)
    return d


@pytest.fixture
def writable(tmp_path):
    """A path inside the project's agent-writable sandbox root. The write jail
    is relative to PROJECT_ROOT with an allowlist (sandbox/, data/, extracted/),
    so a tmp_path target is refused by design. sandbox/ is gitignored."""
    d = pathlib.Path(tools.security.PROJECT_ROOT) / "sandbox" / "tool_tests"
    d.mkdir(parents=True, exist_ok=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


class FakeResp:
    def __init__(self, body, status=200):
        self._body = body
        self.status = status

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


# --------------------------------------------------------------------------
# http_get
# --------------------------------------------------------------------------
class TestHttpGet:
    def test_a_private_address_is_blocked(self):
        out = tools.http_get("http://169.254.169.254/latest/meta-data/")
        assert out.startswith("blocked:")

    def test_a_public_url_is_fetched_and_returned(self, monkeypatch):
        with mock.patch.object(tools.security, "guarded_urlopen",
                               return_value=FakeResp(b"hello world")):
            assert tools.http_get("https://example.com") == "hello world"

    def test_output_is_capped_at_max_chars(self, monkeypatch):
        with mock.patch.object(tools.security, "guarded_urlopen",
                               return_value=FakeResp(b"x" * 5000)):
            assert len(tools.http_get("https://example.com", max_chars=100)) == 100

    def test_a_secret_in_the_response_is_redacted(self, monkeypatch):
        body = b"key is gsk_AbCdEfGhIjKlMnOpQrStUvWx here"
        with mock.patch.object(tools.security, "guarded_urlopen",
                               return_value=FakeResp(body)):
            out = tools.http_get("https://example.com")
        assert "gsk_AbCdEfGhIjKlMnOpQrStUvWx" not in out


# --------------------------------------------------------------------------
# read_file / write_file / list_dir
# --------------------------------------------------------------------------
class TestFileTools:
    def test_reading_a_dotfile_is_blocked(self):
        assert tools.read_file(".env").startswith("blocked:")

    def test_reading_a_credential_named_file_is_blocked(self, scratch):
        p = scratch / "id_rsa"
        p.write_text("secret")
        assert tools.read_file(str(p)).startswith("blocked:")

    def test_writing_inside_a_writable_root_succeeds(self, writable):
        out = tools.write_file(str(writable / "note.txt"), "hello")
        assert out.startswith("wrote 5 chars"), out
        assert (writable / "note.txt").read_text() == "hello"

    def test_writing_creates_missing_parent_directories(self, writable):
        out = tools.write_file(str(writable / "a" / "b" / "c.txt"), "deep")
        assert out.startswith("wrote"), out
        assert (writable / "a" / "b" / "c.txt").read_text() == "deep"

    def test_writing_outside_the_writable_roots_is_denied(self, scratch):
        """LOOMWEAVER_SCRATCH widens what may be read, not what may be written."""
        out = tools.write_file(str(scratch / "note.txt"), "hello")
        assert out.startswith("blocked:")
        assert "agent-writable roots" in out

    def test_writing_outside_the_jail_is_blocked(self):
        assert tools.write_file("/etc/passwd", "x").startswith("blocked:")

    def test_listing_a_blocked_path_is_refused(self):
        assert tools.list_dir("/root/.ssh").startswith("blocked:")

    def test_listing_scratch_works(self, scratch):
        (scratch / "a.txt").write_text("x")
        assert "a.txt" in tools.list_dir(str(scratch))


# --------------------------------------------------------------------------
# shell
# --------------------------------------------------------------------------
class TestShell:
    def test_a_hanging_command_times_out_and_reports_it(self, monkeypatch):
        def raise_timeout(*a, **k):
            raise subprocess.TimeoutExpired(cmd="sleep", timeout=1)
        monkeypatch.setattr(tools.subprocess, "run", raise_timeout)
        assert tools.shell("sleep 999", timeout=1) == "timeout after 1s"

    def test_safe_mode_disables_the_shell(self, monkeypatch):
        monkeypatch.setattr(tools, "SAFE_MODE", True)
        assert "disabled" in tools.shell("echo hi").lower() or \
               tools.shell("echo hi").startswith("blocked")


# --------------------------------------------------------------------------
# remember
# --------------------------------------------------------------------------
class TestRemember:
    def test_no_session_bound_is_reported_not_raised(self):
        assert tools.remember("k", "v", sess=None) == "no session bound"

    def test_a_bound_session_stores_the_fact(self):
        sess = {"id": "s1", "messages": [], "facts": {}}
        tools.remember("project", "flippy", sess=sess)
        assert sess["facts"]["project"] == "flippy"


# --------------------------------------------------------------------------
# sql_query
# --------------------------------------------------------------------------
class TestSqlQuery:
    def test_a_missing_database_is_reported(self, scratch):
        out = tools.sql_query(str(scratch / "nope.db"), "SELECT 1")
        assert "no such database" in out

    def test_a_valid_select_returns_rows(self, scratch):
        db = scratch / "t.db"
        conn = sqlite3.connect(str(db))
        conn.execute("CREATE TABLE t (a TEXT)")
        conn.execute("INSERT INTO t VALUES ('x')")
        conn.commit()
        conn.close()
        assert "x" in tools.sql_query(str(db), "SELECT a FROM t")

    def test_a_non_select_is_stopped_by_the_guard_first(self, scratch):
        db = scratch / "t.db"
        sqlite3.connect(str(db)).execute("CREATE TABLE t (a TEXT)")
        out = tools.sql_query(str(db), "SELEKT nonsense FROM")
        assert out.startswith("blocked:") and "SELECT or WITH" in out

    def test_a_sql_error_from_the_engine_is_caught_not_raised(self, scratch):
        db = scratch / "t.db"
        sqlite3.connect(str(db)).execute("CREATE TABLE t (a TEXT)")
        out = tools.sql_query(str(db), "SELECT * FROM no_such_table")
        assert out.startswith("sql error:")

    def test_a_write_is_refused(self, scratch):
        db = scratch / "t.db"
        sqlite3.connect(str(db)).execute("CREATE TABLE t (a TEXT)")
        out = tools.sql_query(str(db), "DROP TABLE t")
        assert "blocked" in out.lower() or "error" in out.lower()


# --------------------------------------------------------------------------
# json_transform
# --------------------------------------------------------------------------
class TestJsonTransform:
    def test_a_missing_source_is_reported(self, scratch, writable):
        out = tools.json_transform(str(scratch / "nope.json"),
                                   str(writable / "o.json"))
        assert "error reading" in out

    def test_an_unparseable_source_is_reported(self, scratch, writable):
        src = scratch / "bad.json"
        src.write_text("{not json at all")
        assert "error reading" in tools.json_transform(str(src),
                                                       str(writable / "o.json"))

    def test_a_write_failure_is_reported_not_raised(self, scratch, writable):
        """Not a mocked exception: a target beneath a path that is a file, so
        os.makedirs really does raise NotADirectoryError (an OSError)."""
        src = scratch / "in.json"
        src.write_text(json.dumps([{"a": 1}]))
        blocker = writable / "blocker"
        blocker.write_text("i am a file, not a directory")
        out = tools.json_transform(str(src), str(blocker / "sub" / "o.json"))
        assert out.startswith("error writing"), out

    def test_a_valid_transform_writes_the_output(self, scratch, writable):
        src = scratch / "in.json"
        src.write_text(json.dumps([{"a": 1}, {"a": 2}]))
        out_path = writable / "out.json"
        out = tools.json_transform(str(src), str(out_path))
        assert out_path.exists(), out


# --------------------------------------------------------------------------
# the aihub bridges — they must return a string, never raise
# --------------------------------------------------------------------------
@pytest.mark.skipif(not tools._AIHUB_AVAILABLE, reason="aihub not importable")
class TestAihubBridges:
    def test_tts_returns_the_output_path(self, monkeypatch):
        monkeypatch.setattr(tools._aihub, "tts", lambda text, outpath=None: "/tmp/x.mp3")
        assert tools.tts_bridge("hello") == "/tmp/x.mp3"

    def test_tts_passes_no_outpath_when_blank(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(tools._aihub, "tts",
                            lambda text, outpath=None: seen.update(op=outpath) or "/tmp/x.mp3")
        tools.tts_bridge("hello", outpath="")
        assert seen["op"] is None

    def test_rag_query_formats_hits(self, monkeypatch):
        monkeypatch.setattr(tools._aihub, "rag_query",
                            lambda q, top_k=3: [{"score": 0.87, "text": "a match"}])
        out = tools.rag_query_bridge("q")
        assert "0.870" in out and "a match" in out

    def test_rag_query_with_no_hits_returns_empty_not_an_error(self, monkeypatch):
        monkeypatch.setattr(tools._aihub, "rag_query", lambda q, top_k=3: [])
        assert tools.rag_query_bridge("q") == ""

    def test_rag_query_swallows_a_backend_failure(self, monkeypatch):
        monkeypatch.setattr(tools._aihub, "rag_query",
                            mock.Mock(side_effect=RuntimeError("no vectors")))
        assert "rag query unavailable" in tools.rag_query_bridge("q")

    def test_rag_add_reports_the_id(self, monkeypatch):
        monkeypatch.setattr(tools._aihub, "rag_add", lambda text, meta=None: "abc123")
        assert tools.rag_add_bridge("some text") == "added to RAG store: abc123"

    def test_rag_add_parses_metadata(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(tools._aihub, "rag_add",
                            lambda text, meta=None: seen.update(m=meta) or "id1")
        tools.rag_add_bridge("t", meta='{"source": "unit"}')
        assert seen["m"] == {"source": "unit"}

    def test_rag_add_with_bad_metadata_falls_back_to_empty(self, monkeypatch):
        seen = {}
        monkeypatch.setattr(tools._aihub, "rag_add",
                            lambda text, meta=None: seen.update(m=meta) or "id1")
        tools.rag_add_bridge("t", meta="{not json")
        assert seen["m"] == {}

    def test_rag_add_swallows_a_backend_failure(self, monkeypatch):
        monkeypatch.setattr(tools._aihub, "rag_add",
                            mock.Mock(side_effect=RuntimeError("no key")))
        assert "rag add unavailable" in tools.rag_add_bridge("t")

    def test_summarize_returns_the_summary(self, monkeypatch):
        monkeypatch.setattr(tools._aihub, "summarize", lambda text, **kw: "SHORT")
        assert tools.summarize_bridge("a long text") == "SHORT"

    def test_summarize_swallows_a_backend_failure(self, monkeypatch):
        monkeypatch.setattr(tools._aihub, "summarize",
                            mock.Mock(side_effect=RuntimeError("no providers")))
        assert "summarize unavailable" in tools.summarize_bridge("t")

    def test_embed_reports_the_vector_dimension(self, monkeypatch):
        monkeypatch.setattr(tools._aihub, "embed", lambda texts: [[0.1] * 1024])
        assert "1024" in tools.embed_bridge("hello")

    def test_embed_swallows_a_backend_failure(self, monkeypatch):
        monkeypatch.setattr(tools._aihub, "embed",
                            mock.Mock(side_effect=RuntimeError("FREEINFERENCE_KEY required")))
        assert "embed unavailable" in tools.embed_bridge("hello")


# --------------------------------------------------------------------------
# registry shape
# --------------------------------------------------------------------------
class TestRegistry:
    def test_every_tool_declares_a_description_and_params(self):
        for name, spec in tools.TOOLS.items():
            assert spec["desc"], f"{name} has no description"
            assert isinstance(spec["params"], dict)
            assert callable(spec["fn"])

    def test_dispatch_reaches_the_registered_function(self, monkeypatch):
        monkeypatch.setattr(tools._aihub, "summarize", lambda text, **kw: "OK")
        assert tools.dispatch("summarize", {"text": "x"}) == "OK"

    def test_an_unknown_tool_is_reported_not_raised(self):
        out = tools.dispatch("not_a_real_tool", {})
        assert "not_a_real_tool" in str(out)

    def test_a_schema_is_available_for_every_tool(self):
        for name in tools.TOOLS:
            s = tools.schema_for(name)
            assert s["function"]["name"] == name

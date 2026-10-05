"""Coverage for src/aihub.py — the single entry point.

It was at 34% despite being the one command the project tells you to run.
Every network call is stubbed at urllib.request.urlopen; the vector store is
redirected into tmp_path; HOME is redirected so the TTS guard cannot write to
the real home directory.
"""
import base64
import io
import json
import os
import sys
import types
import urllib.error
import urllib.request
from unittest import mock

import pytest

import aihub


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
class FakeResp:
    def __init__(self, payload, raw=None):
        self._payload = payload
        self._raw = raw

    def read(self):
        return self._raw if self._raw is not None else json.dumps(self._payload).encode()

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False


def http_error(code, body):
    raw = body if isinstance(body, bytes) else json.dumps(body).encode()
    return urllib.error.HTTPError("http://x.test", code, "err", {}, io.BytesIO(raw))


@pytest.fixture(autouse=True)
def isolated_aihub(monkeypatch, tmp_path):
    """Vector store and HOME both point into tmp_path."""
    monkeypatch.setenv("AIHUB_VECTOR_STORE", str(tmp_path / "vectors.json"))
    monkeypatch.setenv("HOME", str(tmp_path))
    yield


@pytest.fixture
def embeds(monkeypatch):
    """Stub embed() with a deterministic vector derived from the text."""
    def fake_embed(texts):
        if isinstance(texts, str):
            texts = [texts]
        return [[float(len(t)), 1.0, 0.0] for t in texts]
    monkeypatch.setattr(aihub, "embed", fake_embed)


# --------------------------------------------------------------------------
# embed
# --------------------------------------------------------------------------
class TestEmbed:
    def test_a_missing_key_is_a_clear_error_not_a_401(self, monkeypatch):
        monkeypatch.delenv("FREEINFERENCE_KEY", raising=False)
        with pytest.raises(RuntimeError, match="FREEINFERENCE_KEY required"):
            aihub.embed("hello")

    def test_a_bare_string_is_wrapped_into_a_list(self, monkeypatch):
        monkeypatch.setenv("FREEINFERENCE_KEY", "fi_test")
        seen = {}

        def cap(req, timeout=None):
            seen["body"] = json.loads(req.data.decode())
            return FakeResp({"data": [{"embedding": [1.0, 2.0]}]})

        with mock.patch.object(urllib.request, "urlopen", side_effect=cap):
            out = aihub.embed("one string")
        assert seen["body"]["input"] == ["one string"]
        assert out == [[1.0, 2.0]]

    def test_an_http_error_names_the_status(self, monkeypatch):
        monkeypatch.setenv("FREEINFERENCE_KEY", "fi_test")
        with mock.patch.object(urllib.request, "urlopen",
                               side_effect=http_error(401, {"error": "bad key"})):
            with pytest.raises(RuntimeError, match=r"embed HTTP 401"):
                aihub.embed("hello")


# --------------------------------------------------------------------------
# vector store + rag
# --------------------------------------------------------------------------
class TestVectorStore:
    def test_the_env_var_overrides_the_default_path(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AIHUB_VECTOR_STORE", str(tmp_path / "custom.json"))
        assert aihub._store_path().endswith("custom.json")

    def test_a_missing_store_reads_as_empty_not_an_error(self, tmp_path):
        assert aihub._load_store() == []

    def test_save_then_load_round_trips(self, tmp_path):
        aihub._save_store([{"id": "a", "text": "t", "vector": [1.0], "meta": {}}])
        assert aihub._load_store()[0]["id"] == "a"


class TestRag:
    def test_add_then_query_finds_the_document(self, embeds):
        rid = aihub.rag_add("flippy routes LLM calls")
        assert len(rid) == 12
        hits = aihub.rag_query("flippy routes LLM calls")
        assert hits and hits[0]["text"] == "flippy routes LLM calls"
        assert hits[0]["score"] == pytest.approx(1.0, abs=0.01)

    def test_readding_the_same_text_replaces_it_rather_than_duplicating(self, embeds):
        aihub.rag_add("same text")
        aihub.rag_add("same text")
        assert len(aihub._load_store()) == 1

    def test_add_accepts_metadata(self, embeds):
        aihub.rag_add("doc", meta={"source": "unit-test"})
        assert aihub._load_store()[0]["meta"]["source"] == "unit-test"

    def test_query_on_an_empty_store_returns_empty(self, embeds):
        assert aihub.rag_query("anything") == []

    def test_top_k_limits_the_result_count(self, embeds):
        for i in range(5):
            aihub.rag_add(f"document number {i}")
        assert len(aihub.rag_query("document", top_k=2)) == 2

    def test_results_are_ranked_best_match_first(self, embeds):
        aihub.rag_add("a")        # vector [1,1,0] — closest to a 1-char query
        aihub.rag_add("a much longer piece of text")
        assert aihub.rag_query("a")[0]["text"] == "a"


# --------------------------------------------------------------------------
# tts — the write guard is the security-relevant part
# --------------------------------------------------------------------------
class TestTtsOutpathGuard:
    @pytest.fixture
    def tts_dir(self, tmp_path):
        return tmp_path / "aihub_tts"

    def test_the_default_target_is_inside_the_tts_dir(self, tts_dir):
        with pytest.raises(RuntimeError):   # no edge-tts, no Groq key
            aihub.tts("hello")
        assert tts_dir.is_dir(), "the TTS dir should have been created"

    def test_an_arbitrary_absolute_path_is_refused(self, tmp_path, tts_dir):
        with pytest.raises(ValueError, match="outside ~/aihub_tts"):
            aihub.tts("hello", outpath=str(tmp_path / "escaped.mp3"))

    def test_a_traversal_target_is_refused(self, tts_dir):
        with pytest.raises(ValueError, match="outside ~/aihub_tts"):
            aihub.tts("hello", outpath="../../root.mp3")

    def test_a_full_path_inside_the_tts_dir_is_allowed(self, tts_dir):
        tts_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(RuntimeError):   # guard passed; synthesis then failed
            aihub.tts("hello", outpath=str(tts_dir / "voice.mp3"))

    def test_a_bare_filename_is_refused_because_abspath_uses_the_cwd(self, tts_dir):
        """Stricter than the docstring suggests, and deliberately so: abspath()
        resolves a bare name against the cwd, which is not the TTS dir."""
        tts_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(ValueError, match="outside ~/aihub_tts"):
            aihub.tts("hello", outpath="voice.mp3")

    def test_the_opt_in_env_var_allows_other_paths(self, monkeypatch, tmp_path):
        monkeypatch.setenv("AIHUB_ALLOW_ARBITRARY_OUTPATH", "1")
        target = tmp_path / "anywhere.mp3"
        fake = types.ModuleType("edge_tts")

        class C:
            def __init__(self, text, voice):
                pass

            async def save(self, path):
                open(path, "wb").write(b"audio")
        fake.Communicate = C
        monkeypatch.setitem(sys.modules, "edge_tts", fake)
        assert aihub.tts("hello", outpath=str(target)) == str(target)
        assert target.read_bytes() == b"audio"


class TestTtsSynthesis:
    def test_edge_tts_is_used_when_present(self, monkeypatch, tmp_path):
        fake = types.ModuleType("edge_tts")
        calls = {}

        class C:
            def __init__(self, text, voice):
                calls["text"], calls["voice"] = text, voice

            async def save(self, path):
                calls["path"] = path
                open(path, "wb").write(b"mp3")
        fake.Communicate = C
        monkeypatch.setitem(sys.modules, "edge_tts", fake)
        out = aihub.tts("say this")
        assert calls["text"] == "say this" and os.path.exists(out)

    def test_groq_orpheus_is_the_fallback(self, monkeypatch, tmp_path):
        monkeypatch.delitem(sys.modules, "edge_tts", raising=False)
        monkeypatch.setenv("GROQ_KEY", "gsk_test")
        with mock.patch.object(urllib.request, "urlopen",
                               return_value=FakeResp({}, raw=b"groq-audio")):
            out = aihub.tts("hello")
        assert open(out, "rb").read() == b"groq-audio"

    def test_with_no_backend_the_error_says_why(self, monkeypatch):
        monkeypatch.delitem(sys.modules, "edge_tts", raising=False)
        monkeypatch.delenv("GROQ_KEY", raising=False)
        with pytest.raises(RuntimeError, match="TTS failed"):
            aihub.tts("hello")


# --------------------------------------------------------------------------
# stt / vision
# --------------------------------------------------------------------------
class TestStt:
    def test_missing_credentials_are_reported_before_any_call(self, monkeypatch):
        monkeypatch.delenv("CLOUDFLARE_TOKEN", raising=False)
        with pytest.raises(RuntimeError, match="CLOUDFLARE_TOKEN"):
            aihub.stt("/nonexistent.wav")

    def test_the_audio_is_base64_encoded_into_the_body(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLOUDFLARE_TOKEN", "cf_tok")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc123")
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"RIFFdata")
        seen = {}

        def cap(req, timeout=None):
            seen["body"] = json.loads(req.data.decode())
            seen["url"] = req.full_url
            return FakeResp({"result": {"text": "transcribed"}})

        with mock.patch.object(urllib.request, "urlopen", side_effect=cap):
            assert aihub.stt(str(wav)) == "transcribed"
        assert base64.b64decode(seen["body"]["audio"]) == b"RIFFdata"
        assert "acc123" in seen["url"]

    def test_a_fallback_top_level_text_key_is_accepted(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLOUDFLARE_TOKEN", "cf_tok")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc123")
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"x")
        with mock.patch.object(urllib.request, "urlopen",
                               return_value=FakeResp({"text": "flat"})):
            assert aihub.stt(str(wav)) == "flat"

    def test_an_http_error_names_the_status(self, monkeypatch, tmp_path):
        monkeypatch.setenv("CLOUDFLARE_TOKEN", "cf_tok")
        monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "acc123")
        wav = tmp_path / "a.wav"
        wav.write_bytes(b"x")
        with mock.patch.object(urllib.request, "urlopen",
                               side_effect=http_error(403, {"e": "nope"})):
            with pytest.raises(RuntimeError, match=r"stt HTTP 403"):
                aihub.stt(str(wav))


class TestVision:
    def test_a_missing_key_is_reported_before_reading_the_file(self, monkeypatch):
        monkeypatch.delenv("FREEINFERENCE_KEY", raising=False)
        with pytest.raises(RuntimeError, match="FREEINFERENCE_KEY required"):
            aihub.vision("/nonexistent.jpg", "what is this")

    def test_the_image_is_sent_as_a_data_url(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FREEINFERENCE_KEY", "fi_test")
        img = tmp_path / "p.jpg"
        img.write_bytes(b"JPEG")
        seen = {}

        def cap(req, timeout=None):
            seen["body"] = json.loads(req.data.decode())
            return FakeResp({"choices": [{"message": {"content": "a cat"}}]})

        with mock.patch.object(urllib.request, "urlopen", side_effect=cap):
            assert aihub.vision(str(img), "what is this") == "a cat"
        parts = seen["body"]["messages"][0]["content"]
        assert parts[0]["text"] == "what is this"
        assert parts[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")

    def test_an_http_error_names_the_status(self, monkeypatch, tmp_path):
        monkeypatch.setenv("FREEINFERENCE_KEY", "fi_test")
        img = tmp_path / "p.jpg"
        img.write_bytes(b"JPEG")
        with mock.patch.object(urllib.request, "urlopen",
                               side_effect=http_error(500, {"e": 1})):
            with pytest.raises(RuntimeError, match=r"vision HTTP 500"):
                aihub.vision(str(img), "q")


# --------------------------------------------------------------------------
# tooltest
# --------------------------------------------------------------------------
class TestTooltest:
    @pytest.fixture
    def fake_router(self, monkeypatch):
        holder = {"resp": {"model": "minimax-m3", "usage": {},
                           "choices": [{"message": {"role": "assistant",
                                                    "content": "hi"}}]},
                  "exc": None}
        router = mock.Mock()

        def completion(**kw):
            if holder["exc"]:
                raise holder["exc"]
            return holder["resp"]
        router.completion = completion
        monkeypatch.setattr(aihub, "build_router", lambda: (router, None))
        return holder

    def test_a_plain_completion_is_returned(self, fake_router):
        msg = aihub.tooltest()
        assert msg["content"] == "hi"

    def test_a_requested_tool_is_dispatched_through_the_guards(self, fake_router,
                                                               monkeypatch):
        fake_router["resp"]["choices"][0]["message"]["tool_calls"] = [{
            "function": {"name": "list_dir", "arguments": '{"path": "."}'}}]
        monkeypatch.setattr(aihub.hub, "record_outcome", lambda *a, **k: None)
        msg = aihub.tooltest()
        assert "list_dir" in msg.get("tool_results", {})

    def test_a_tool_that_is_not_registered_is_ignored(self, fake_router, monkeypatch):
        fake_router["resp"]["choices"][0]["message"]["tool_calls"] = [{
            "function": {"name": "rm_everything", "arguments": "{}"}}]
        monkeypatch.setattr(aihub.hub, "record_outcome", lambda *a, **k: None)
        msg = aihub.tooltest()
        assert "rm_everything" not in msg.get("tool_results", {})

    def test_malformed_tool_arguments_do_not_crash_the_call(self, fake_router,
                                                            monkeypatch):
        fake_router["resp"]["choices"][0]["message"]["tool_calls"] = [{
            "function": {"name": "list_dir", "arguments": "{not json"}}]
        monkeypatch.setattr(aihub.hub, "record_outcome", lambda *a, **k: None)
        msg = aihub.tooltest()
        assert "list_dir" in msg.get("tool_results", {})

    def test_a_router_failure_becomes_a_runtime_error(self, fake_router, monkeypatch):
        fake_router["exc"] = ValueError("upstream 500")
        monkeypatch.setattr(aihub.hub, "record_outcome", lambda *a, **k: None)
        with pytest.raises(RuntimeError, match="tool call failed"):
            aihub.tooltest()


# --------------------------------------------------------------------------
# run_engine / print_profile / health
# --------------------------------------------------------------------------
class TestRunEngine:
    def test_the_exit_code_is_passed_through(self, monkeypatch):
        monkeypatch.setattr("loomweaver.cli.main", lambda argv: 3)
        assert aihub.run_engine(["usage"]) == 3

    def test_a_none_return_counts_as_success(self, monkeypatch):
        monkeypatch.setattr("loomweaver.cli.main", lambda argv: None)
        assert aihub.run_engine(["usage"]) == 0

    def test_an_argparse_exit_is_not_treated_as_a_failure(self, monkeypatch):
        def boom(argv):
            raise SystemExit(0)
        monkeypatch.setattr("loomweaver.cli.main", boom)
        assert aihub.run_engine(["usage", "--help"]) == 0

    def test_a_crash_is_reported_as_1_not_a_traceback(self, monkeypatch, capsys):
        def boom(argv):
            raise RuntimeError("engine exploded")
        monkeypatch.setattr("loomweaver.cli.main", boom)
        assert aihub.run_engine(["usage"]) == 1
        assert "engine command failed" in capsys.readouterr().out


class TestProfileAndHealth:
    def test_print_profile_survives_a_missing_learning_layer(self, monkeypatch,
                                                              capsys):
        def boom():
            raise ImportError("no learning module")
        monkeypatch.setattr("loomweaver.learning.get_store", boom)
        aihub.print_profile()
        assert "profile unavailable" in capsys.readouterr().out

    def test_health_lists_providers_keys_and_models(self, monkeypatch, capsys):
        monkeypatch.setattr(aihub, "build_deployments", lambda: [
            {"model_name": "m1", "metadata": {"provider": "groq", "key_index": 0}},
            {"model_name": "m1", "metadata": {"provider": "groq", "key_index": 1}},
            {"model_name": "m2", "metadata": {"provider": "together", "key_index": 0}},
        ])
        monkeypatch.setattr(aihub.hub, "services", lambda: {"cache": True})
        aihub.health()
        out = capsys.readouterr().out
        assert "providers (2)" in out and "groq" in out and "together" in out
        assert "live keys: 3" in out and "models: 2" in out
        assert "[on ] cache" in out

    def test_health_says_so_when_nothing_is_configured(self, monkeypatch, capsys):
        monkeypatch.setattr(aihub, "build_deployments", lambda: [])
        monkeypatch.setattr(aihub.hub, "services", lambda: {})
        aihub.health()
        assert "set at least one provider env var" in capsys.readouterr().out


# --------------------------------------------------------------------------
# main — dispatch only, every handler stubbed
# --------------------------------------------------------------------------
class TestMainDispatch:
    def run(self, argv, monkeypatch, capsys):
        monkeypatch.setattr("sys.argv", ["aihub.py"] + argv)
        aihub.main()
        return capsys.readouterr()

    def test_health_flag(self, monkeypatch, capsys):
        monkeypatch.setattr(aihub, "health", lambda: print("HEALTH"))
        assert "HEALTH" in self.run(["--health"], monkeypatch, capsys).out

    def test_providers_flag_reuses_health(self, monkeypatch, capsys):
        monkeypatch.setattr(aihub, "health", lambda: print("HEALTH"))
        assert "HEALTH" in self.run(["--providers"], monkeypatch, capsys).out

    def test_all_providers_prints_the_catalog(self, monkeypatch, capsys):
        out = self.run(["--all-providers"], monkeypatch, capsys).out
        assert "Every provider flippy speaks" in out and "activate with" in out

    def test_profile_flag(self, monkeypatch, capsys):
        monkeypatch.setattr(aihub, "print_profile", lambda: print("PROFILE"))
        assert "PROFILE" in self.run(["--profile"], monkeypatch, capsys).out

    def test_chat_flag_passes_simple_through(self, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr(aihub, "smart_chat",
                            lambda msgs, **kw: seen.update(kw) or {"content": "hi"})
        self.run(["--chat", "hello", "--simple"], monkeypatch, capsys)
        assert seen["simple"] is True

    def test_engine_flag_forwards_trailing_positional_args(self, monkeypatch,
                                                            capsys):
        seen = {}
        monkeypatch.setattr(aihub, "run_engine",
                            lambda argv: seen.update(argv=argv) or 0)
        with pytest.raises(SystemExit) as exc:
            self.run(["--engine", "usage", "extra"], monkeypatch, capsys)
        assert exc.value.code == 0
        assert seen["argv"] == ["usage", "extra"]

    def test_engine_flag_does_not_accept_unknown_flags(self, monkeypatch, capsys):
        """Known limitation: aihub's own argparse runs first, so flags meant for
        the engine (e.g. --engine loadtest --concurrency 4) are rejected."""
        with pytest.raises(SystemExit) as exc:
            self.run(["--engine", "loadtest", "--concurrency", "4"],
                     monkeypatch, capsys)
        assert exc.value.code == 2, "argparse usage error"

    def test_agent_flag_forwards_the_goal(self, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr(aihub, "run_engine",
                            lambda argv: seen.update(argv=argv) or 0)
        with pytest.raises(SystemExit):
            self.run(["--agent", "do a thing", "--tools", "shell"],
                     monkeypatch, capsys)
        assert seen["argv"][:2] == ["agent", "do a thing"]
        assert "--tools" in seen["argv"]

    def test_summarize_flag(self, monkeypatch, capsys):
        monkeypatch.setattr(aihub, "summarize", lambda t, **kw: "SHORT")
        assert "SHORT" in self.run(["--summarize", "long text"],
                                   monkeypatch, capsys).out

    def test_tts_flag_prints_the_output_path(self, monkeypatch, capsys):
        monkeypatch.setattr(aihub, "tts", lambda t, **kw: "/tmp/x.mp3")
        assert "/tmp/x.mp3" in self.run(["--tts", "hi"], monkeypatch, capsys).out

    def test_stt_flag(self, monkeypatch, capsys):
        monkeypatch.setattr(aihub, "stt", lambda p: "transcribed")
        assert "transcribed" in self.run(["--stt", "a.wav"], monkeypatch, capsys).out

    def test_vision_flag_uses_the_prompt_or_a_default(self, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr(aihub, "vision",
                            lambda p, prompt, **kw: seen.update(prompt=prompt) or "a cat")
        self.run(["--vision", "p.jpg", "what", "is", "this"], monkeypatch, capsys)
        assert seen["prompt"] == "what is this"

    def test_vision_flag_defaults_the_prompt(self, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr(aihub, "vision",
                            lambda p, prompt, **kw: seen.update(prompt=prompt) or "a cat")
        self.run(["--vision", "p.jpg"], monkeypatch, capsys)
        assert seen["prompt"] == "Describe this image."

    def test_rag_add_without_text_prints_usage(self, monkeypatch, capsys):
        assert "usage:" in self.run(["--rag", "add"], monkeypatch, capsys).out

    def test_rag_add_reports_the_id(self, monkeypatch, capsys, embeds):
        out = self.run(["--rag", "add", "some", "text"], monkeypatch, capsys).out
        assert out.startswith("added:")

    def test_rag_query_with_no_matches_says_so(self, monkeypatch, capsys, embeds):
        assert "(no matches)" in self.run(["--rag", "query", "nothing"],
                                          monkeypatch, capsys).out

    def test_no_arguments_prints_help(self, monkeypatch, capsys):
        assert "usage:" in self.run([], monkeypatch, capsys).out

    def test_learn_stores_a_lesson(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setenv("LOOMWEAVER_LEARNING_DB", str(tmp_path / "learn.db"))
        out = self.run(["--learn", "always reply in French"], monkeypatch, capsys).out
        assert "learned (lesson" in out

    def test_forget_clears_lessons(self, monkeypatch, capsys, tmp_path):
        monkeypatch.setenv("LOOMWEAVER_LEARNING_DB", str(tmp_path / "learn.db"))
        self.run(["--learn", "a rule"], monkeypatch, capsys)
        assert "forgot every lesson" in self.run(["--forget"], monkeypatch, capsys).out

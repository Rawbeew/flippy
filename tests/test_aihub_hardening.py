"""T3-1: aihub secret hygiene + clean import under zero third-party deps.

aihub.py is the ONE module allowed third-party imports (litellm / edge-tts /
pillow). All must stay optional and guarded so the module imports cleanly with
NONE installed, and an authentication failure must never leak a full secret
into an exception message or a log line.
"""

import importlib.util
import pytest
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import aihub


SECRET_KEY = "sk-AbC1234567890XyZqRstUvWxYz-9876543210"


class TestModuleImportsCleanWithoutThirdParty:
    """aihub must be importable even though litellm/edge_tts/PIL are absent."""

    def test_import_does_not_require_third_party_packages(self):
        # BLOCK every third-party dependency aihub guards by name; a working
        # import proves the guards are in place, not that the deps are present.
        import builtins

        real_import = builtins.__import__

        def fake_import(name, *a, **k):
            if name.split(".")[0] in ("litellm", "edge_tts", "PIL",
                                      "flippy_providers"):
                raise ImportError(f"blocked third-party import: {name}")
            return real_import(name, *a, **k)

        with mock.patch.object(builtins, "__import__", side_effect=fake_import):
            import importlib
            mod = importlib.reload(aihub)
        # functions that lazily import third-party deps must exist and be callable
        assert callable(getattr(mod, "build_router", None))
        assert callable(getattr(mod, "tts", None))


class TestAuthFailureNeverLeaksSecret:
    """A simulated litellm auth failure must surface a sanitized message."""

    def _fake_router_that_fails(self):
        class FakeRouter:
            def completion(self, **kw):
                # a litellm auth error may embed the full key in its message
                raise RuntimeError(
                    f"AuthenticationError 401: invalid api key "
                    f"'{SECRET_KEY}' for model"
                )
        return mock.patch.object(
            aihub, "build_router",
            return_value=(FakeRouter(), None),
        )

    def test_smart_chat_exception_never_contains_full_secret(self):
        with self._fake_router_that_fails():
            with mock.patch.object(aihub, "rag_query", return_value=[]):
                with mock.patch.object(aihub, "embed",
                                       side_effect=AssertionError("no embed")):
                    try:
                        aihub.smart_chat([{"role": "user", "content": "hi"}])
                        raise AssertionError("expected RuntimeError")
                    except RuntimeError as e:
                        msg = str(e)
        assert SECRET_KEY not in msg, f"full secret leaked into exception: {msg!r}"
        # the provider still learns the call failed, but the credential is scrubbed
        assert "failed" in msg or "completion" in msg

    def test_summarize_exception_never_contains_full_secret(self):
        with self._fake_router_that_fails():
            try:
                aihub.summarize("summarize this text")
                raise AssertionError("expected RuntimeError")
            except RuntimeError as e:
                msg = str(e)
        assert SECRET_KEY not in msg, f"full secret leaked into exception: {msg!r}"

    def test_edge_tts_log_line_never_contains_full_secret(self, capsys):
        # a failed edge-tts call can embed a credential in its exception; the
        # `[aihub] edge-tts failed: ...` stderr log line must be sanitized.
        import builtins
        real_import = builtins.__import__

        class _FakeEdge:
            class Communicate:
                def __init__(self, *a, **k):
                    pass

                async def save(self, path):
                    raise RuntimeError(f"edge-tts error near token {SECRET_KEY}")

        fake_module = _FakeEdge()

        def fake_import(name, *a, **k):
            if name == "edge_tts":
                return fake_module
            return real_import(name, *a, **k)

        with mock.patch.dict("os.environ", {"GROQ_KEY": ""}, clear=False):
            with mock.patch.object(builtins, "__import__", side_effect=fake_import):
                try:
                    aihub.tts("hello world")
                    raise AssertionError("expected RuntimeError")
                except RuntimeError as e:
                    assert SECRET_KEY not in str(e)
        captured = capsys.readouterr()
        log = captured.err
        assert SECRET_KEY not in log, f"full secret leaked into stderr log: {log!r}"
        assert "edge-tts failed" in log


class TestRedactSecretsUnit:
    def test_configured_key_fully_scrubbed(self):
        with mock.patch.dict("os.environ", {"GROQ_KEY": SECRET_KEY}, clear=False):
            out = aihub._redact_secrets(f"error near {SECRET_KEY} in call")
        assert SECRET_KEY not in out
        assert "[REDACTED]" in out

    def test_common_key_shape_scrubbed_even_when_not_configured(self):
        out = aihub._redact_secrets("key was " + SECRET_KEY)
        assert SECRET_KEY not in out
        assert "[REDACTED]" in out

    def test_plain_text_passthrough(self):
        assert aihub._redact_secrets("benign message") == "benign message"

    # ---------- zero-trust: bound tts outpath to home unless operator opts out ----------
    def _call_tts_guard(self, outpath):
        """Call aihub.tts with an outpath and return whatever exception the
        write-guard raised (ValueError) OR None if it passed the guard. Network /
        'unavailable' RuntimeError after a passing guard is not a guard failure."""
        try:
            aihub.tts("hi", outpath=outpath)
            return None
        except ValueError as e:
            return e
        except RuntimeError:
            return None

    def test_tts_outpath_guarded_against_traversal(self):
        import os
        for bad in ["C:/Windows/system32/evil.mp3", "/etc/evil.mp3",
                    os.path.join("..", "..", "root.mp3")]:
            exc = self._call_tts_guard(bad)
            assert isinstance(exc, ValueError), f"unsafe outpath not refused: {bad} ({exc!r})"

    @pytest.mark.skipif(
        importlib.util.find_spec('edge_tts') is None,
        reason="TTS path needs the optional edge-tts extra; without it aihub falls back to a live network call")
    def test_tts_outpath_default_is_safe_home_dir(self):
        exc = self._call_tts_guard(None)
        assert not isinstance(exc, ValueError)

    @pytest.mark.skipif(
        importlib.util.find_spec('edge_tts') is None,
        reason="TTS path needs the optional edge-tts extra; without it aihub falls back to a live network call")
    def test_tts_outpath_opt_in_required_for_arbitrary(self, monkeypatch):
        exc = self._call_tts_guard("C:/Windows/Temp/x.mp3")
        assert isinstance(exc, ValueError), f"expected ValueError without opt-in, got {exc!r}"
        monkeypatch.setenv("AIHUB_ALLOW_ARBITRARY_OUTPATH", "1")
        exc = self._call_tts_guard("C:/Windows/Temp/x.mp3")
        assert not isinstance(exc, ValueError), "opt-in should pass the write guard"


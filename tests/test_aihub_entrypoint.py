"""aihub is the single entry point, and it routes through litellm.

These tests pin the contract that follows from that decision:
  * the universal provider registry reaches the litellm Router, so "use only
    aihub" does not mean "lose bring-your-own-keys";
  * the self-learning layer is wired into smart_chat on BOTH the success and
    the failure path, so "use only aihub" does not mean "lose the memory";
  * an auth failure still never surfaces the configured key.
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

aihub = pytest.importorskip("aihub")


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOMWEAVER_LEARNING_DB", str(tmp_path / "learn.db"))
    from loomweaver import learning
    st = learning.LearningStore(db_path=str(tmp_path / "learn.db"))
    learning.set_store(st)
    yield st
    learning.set_store(None)


class FakeRouter:
    """Stands in for litellm.Router so no network or key is needed."""

    def __init__(self, fail=False):
        self.fail = fail
        self.calls = []

    def completion(self, model=None, messages=None, **kw):
        self.calls.append({"model": model, "messages": messages})
        if self.fail:
            raise RuntimeError("401 invalid api key gsk_SUPERSECRETVALUE")
        return {"choices": [{"message": {"content": "hello"}}], "model": model,
                "usage": {"prompt_tokens": 5, "completion_tokens": 2}}


class TestProviderRegistryReachesLitellm:
    def test_arbitrary_byo_key_provider_is_routed(self, monkeypatch):
        """The headline property: an unknown endpoint works through aihub."""
        monkeypatch.setenv("ACME_API_KEY", "ak_live_1")
        monkeypatch.setenv("ACME_BASE_URL", "https://llm.acme.io/v1")
        monkeypatch.setenv("ACME_MODELS", "acme-large")
        models = aihub.build_router_models()
        names = {friendly for _lit, friendly, _k, _b in models}
        assert "acme-large" in names
        lit, _f, key, base = next(m for m in models if m[1] == "acme-large")
        assert lit == "openai/acme-large"
        assert base == "https://llm.acme.io/v1"
        assert key == "ak_live_1"

    def test_local_keyless_endpoint_is_routed(self, monkeypatch):
        monkeypatch.setenv("OLLAMA_BASE_URL", "http://localhost:11434")
        bases = {b for _l, _f, _k, b in aihub.build_router_models()}
        assert "http://localhost:11434" in bases

    def test_no_providers_gives_an_actionable_error(self, monkeypatch):
        """The message must point at the universal path, not four old brands."""
        for var in list(os.environ):
            if var.endswith(("_KEY", "_TOKEN", "_BASE_URL", "_API_BASE", "_BASE",
                             "_ENDPOINT", "_URL", "_MODELS")):
                monkeypatch.delenv(var, raising=False)
        with pytest.raises(RuntimeError) as exc:
            aihub.build_router()
        msg = str(exc.value)
        assert "<PREFIX>_API_KEY" in msg and "providers --all" in msg

    def test_router_builds_with_a_configured_provider(self, monkeypatch):
        monkeypatch.setenv("GROQ_KEY", "gsk_test_value")
        router, litellm = aihub.build_router()
        assert type(router).__name__ == "Router"
        assert router.model_list


class TestSmartChatLearns:
    def test_success_is_recorded(self, store):
        router = FakeRouter()
        with mock.patch.object(aihub, "build_router", return_value=(router, None)):
            out = aihub.smart_chat([{"role": "user",
                                     "content": "summarise the invoice"}])
        assert out["content"] == "hello"
        assert store.stats()["interactions"] == 1
        prof = store.profile()
        assert prof["success_rate"] == 1.0
        assert "invoice" in prof["vocabulary"]

    def test_failure_is_recorded_and_becomes_a_lesson(self, store):
        with mock.patch.object(aihub, "build_router",
                               return_value=(FakeRouter(fail=True), None)):
            with pytest.raises(RuntimeError):
                aihub.smart_chat([{"role": "user",
                                   "content": "summarise the invoice"}])
        stats = store.stats()
        assert stats["interactions"] == 1
        assert stats["lessons"] == 1, "a real failure must file a lesson"
        assert store.profile()["success_rate"] == 0.0

    def test_failure_message_never_leaks_a_recognisable_key(self, store):
        """A realistically-shaped key is stripped from the raised message."""
        key = "gsk_" + "A1b2C3d4E5f6G7h8I9j0K1l2M3n4"

        class Router:
            def completion(self, **kw):
                raise RuntimeError(f"401 invalid api key {key}")

        with mock.patch.object(aihub, "build_router", return_value=(Router(), None)):
            with pytest.raises(RuntimeError) as exc:
                aihub.smart_chat([{"role": "user", "content": "hi"}])
        assert key not in str(exc.value)


class TestSecretHygieneCoversEveryProvider:
    """The provider surface is open-ended, so redaction cannot be a fixed list.

    A hardcoded six-name list stopped covering the providers added later, which
    meant e.g. a configured MISTRAL_API_KEY could be echoed in an error message.
    """

    def test_universal_provider_keys_are_redacted_by_value(self, monkeypatch):
        monkeypatch.setenv("MISTRAL_API_KEY", "mstk_real_secret_value")
        monkeypatch.setenv("ACME_API_KEY", "ak_live_unrecognised_shape")
        msg = aihub._safe_message(
            Exception("401 mstk_real_secret_value and ak_live_unrecognised_shape"))
        assert "mstk_real_secret_value" not in msg
        assert "ak_live_unrecognised_shape" not in msg

    def test_credential_env_vars_are_derived_not_hardcoded(self, monkeypatch):
        monkeypatch.setenv("TOGETHER_API_KEY", "t")
        monkeypatch.setenv("SOME_VENDOR_TOKEN", "t")
        covered = set(aihub._secret_env_vars())
        assert {"TOGETHER_API_KEY", "SOME_VENDOR_TOKEN"} <= covered
        # the legacy brands stay covered even with no registry available
        assert "GROQ_KEY" in covered

    def test_non_secret_text_is_preserved(self):
        msg = aihub._safe_message(Exception("model minimax-m3 not found"))
        assert "minimax-m3" in msg, "redaction must not eat ordinary error text"

    def test_learning_is_optional_not_fatal(self, monkeypatch):
        """aihub must still work if loomweaver cannot be imported."""
        import builtins
        real_import = builtins.__import__

        def guarded(name, *a, **k):
            if name.startswith("loomweaver"):
                raise ImportError("loomweaver unavailable")
            return real_import(name, *a, **k)

        monkeypatch.setattr(builtins, "__import__", guarded)
        with mock.patch.object(aihub, "build_router",
                               return_value=(FakeRouter(), None)):
            out = aihub.smart_chat([{"role": "user", "content": "hi"}])
        assert out["content"] == "hello"


class TestEntryPointSurface:
    """The commands that make aihub sufficient on its own."""

    def test_print_profile_does_not_raise_on_a_cold_start(self, store, capsys):
        aihub.print_profile()
        assert "No interactions recorded" in capsys.readouterr().out

    def test_print_profile_shows_learned_state(self, store, capsys):
        store.record_route("deploy to staging", "groq", "m", True, 0.5, 1)
        aihub.print_profile()
        assert "preferred provider" in capsys.readouterr().out

    def test_describe_catalog_is_importable_from_here(self):
        from flippy_providers import describe_catalog
        assert len(describe_catalog()) >= 20

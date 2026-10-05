"""The bridge between the litellm front door and the loomweaver engine.

aihub routes through litellm; the things that make flippy production-shaped —
multi-key rotation, the semantic cache, the quota ledger, usage analytics, the
learning memory — live in loomweaver. These tests pin the join, because each of
these was silently absent from the litellm path before it existed:

  * a comma-separated key list actually rotates (and dead keys drop out);
  * a near-duplicate prompt does not re-bill a provider;
  * one call records to usage, quota and learning together;
  * the goal text survives the trip, or the memory is just a counter.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from loomweaver import hub


@pytest.fixture(autouse=True)
def isolated_dbs(tmp_path, monkeypatch):
    """Every engine store gets its own file, so tests cannot cross-contaminate."""
    for var, name in (("LOOMWEAVER_KEYROTATION_DB", "rot"),
                      ("LOOMWEAVER_QUOTA_DB", "quota"),
                      ("LOOMWEAVER_USAGE_DB", "usage"),
                      ("LOOMWEAVER_CACHE_DB", "cache"),
                      ("LOOMWEAVER_LEARNING_DB", "learn")):
        monkeypatch.setenv(var, str(tmp_path / f"{name}.db"))
    from loomweaver import key_rotation, learning, quota_ledger, semantic_cache, usage
    key_rotation.set_state(key_rotation.RotationState(
        db_path=str(tmp_path / "rot.db")))
    learning.set_store(learning.LearningStore(db_path=str(tmp_path / "learn.db")))
    yield
    learning.set_store(None)
    key_rotation.set_state(None)


KEYS = ["k1", "k2", "k3"]


class TestKeyRotation:
    def test_all_keys_live_initially(self):
        assert hub.live_keys("groq", KEYS) == [(0, "k1"), (1, "k2"), (2, "k3")]

    def test_dead_key_is_dropped(self):
        from loomweaver import key_rotation
        key_rotation.get_state().mark_dead("groq", 0, "revoked")
        assert hub.live_keys("groq", KEYS) == [(1, "k2"), (2, "k3")]

    def test_exhausted_key_is_dropped_until_cooldown_ends(self):
        from loomweaver import key_rotation
        key_rotation.get_state().mark_exhausted("groq", 1, retry_after=300)
        assert hub.live_keys("groq", KEYS) == [(0, "k1"), (2, "k3")]

    def test_empty_key_list_is_safe(self):
        assert hub.live_keys("groq", []) == []

    def test_unpacking_is_key_then_index(self):
        """Regression: live_pairs yields (key, index); reading it as (index, key)
        raised a TypeError that a blanket except turned into 'every key is
        live', silently disabling rotation."""
        from loomweaver import key_rotation
        key_rotation.get_state().mark_dead("groq", 0, "revoked")
        got = hub.live_keys("groq", KEYS)
        assert [i for i, _k in got] == [1, 2], f"indices wrong: {got}"
        assert [k for _i, k in got] == ["k2", "k3"], f"keys wrong: {got}"


class TestDeployments:
    def _prov(self, keys=KEYS, models=("m-a", "m-b")):
        return {"name": "groq", "keys": list(keys), "key": keys[0],
                "models": list(models), "litellm_base": None}

    def test_one_deployment_per_live_key_per_model(self):
        deps = hub.key_deployments(self._prov(), lambda m: f"groq/{m}")
        assert len(deps) == 6, "3 keys x 2 models"
        assert {d["litellm_params"]["api_key"] for d in deps} == set(KEYS)

    def test_dead_keys_produce_no_deployments(self):
        """The point of the join: a retired key must not be handed to litellm."""
        from loomweaver import key_rotation
        key_rotation.get_state().mark_dead("groq", 0, "revoked")
        key_rotation.get_state().mark_exhausted("groq", 1, retry_after=300)
        deps = hub.key_deployments(self._prov(), lambda m: f"groq/{m}")
        assert len(deps) == 2
        assert all(d["litellm_params"]["api_key"] == "k3" for d in deps)

    def test_deployments_carry_attribution_metadata(self):
        deps = hub.key_deployments(self._prov(), lambda m: f"groq/{m}")
        assert all(d["metadata"]["provider"] == "groq" for d in deps)
        assert {d["metadata"]["key_index"] for d in deps} == {0, 1, 2}

    def test_model_name_fn_is_applied(self):
        deps = hub.key_deployments(self._prov(models=("m-a",)), lambda m: f"x/{m}")
        assert deps[0]["litellm_params"]["model"] == "x/m-a"

    def test_api_base_only_set_when_present(self):
        p = self._prov()
        p["litellm_base"] = "https://gw/v1"
        assert hub.key_deployments(p, lambda m: m)[0]["litellm_params"]["api_base"] == "https://gw/v1"
        p["litellm_base"] = None
        assert "api_base" not in hub.key_deployments(p, lambda m: m)[0]["litellm_params"]


class TestKeyFailureFeedback:
    def test_401_retires_the_key_permanently(self):
        hub.note_key_failure("groq", 1, status=401)
        assert [k for _i, k in hub.live_keys("groq", KEYS)] == ["k1", "k3"]

    def test_429_cools_the_key_down(self):
        hub.note_key_failure("groq", 2, status=429)
        assert [k for _i, k in hub.live_keys("groq", KEYS)] == ["k1", "k2"]

    def test_500_does_not_retire_a_key(self):
        hub.note_key_failure("groq", 0, status=500)
        assert len(hub.live_keys("groq", KEYS)) == 3

    def test_record_outcome_routes_the_failure_to_rotation(self):
        """Callers record one outcome; the rotation state must hear about it."""
        hub.record_outcome("groq", "m-a", False, 0.1, status=403, key_index=1)
        assert [k for _i, k in hub.live_keys("groq", KEYS)] == ["k1", "k3"]


class TestSemanticCache:
    MSGS = [{"role": "user", "content": "what is the capital of france"}]

    def test_miss_then_hit(self, monkeypatch):
        monkeypatch.setenv("LOOMWEAVER_CACHE_ENABLED", "1")
        from loomweaver import semantic_cache
        semantic_cache._default_cache = None
        assert hub.cache_get(self.MSGS) is None
        hub.cache_put(self.MSGS, "Paris", model_tag="m-a")
        hit = hub.cache_get(self.MSGS)
        assert hit and hit["text"] == "Paris" and hit["cached"] is True

    def test_stateful_conversations_skip_the_cache(self, monkeypatch):
        monkeypatch.setenv("LOOMWEAVER_CACHE_ENABLED", "1")
        msgs = [{"role": "user", "content": "call the tool"},
                {"role": "tool", "tool_call_id": "1", "content": "result"}]
        assert hub.cache_enabled(msgs) is False
        hub.cache_put(msgs, "should not be stored")
        assert hub.cache_get(msgs) is None

    def test_cache_never_raises_when_disabled(self, monkeypatch):
        monkeypatch.setenv("LOOMWEAVER_CACHE_ENABLED", "0")
        assert hub.cache_get(self.MSGS) is None
        hub.cache_put(self.MSGS, "x")  # must be a silent no-op


class TestQuotaGate:
    def test_available_by_default(self):
        allowed, reason = hub.quota_check("groq")
        assert allowed is True

    def test_a_down_ledger_never_blocks_a_request(self, monkeypatch):
        def boom():
            raise RuntimeError("ledger exploded")
        monkeypatch.setattr(hub._ql, "get_ledger", boom)
        assert hub.quota_check("groq") == (True, "")


class TestOutcomeSink:
    def test_one_call_reaches_usage_and_learning(self):
        from loomweaver import learning, usage
        hub.record_outcome("groq", "m-a", True, 0.42,
                           usage={"prompt_tokens": 10, "completion_tokens": 4},
                           goal="summarise the invoice")
        providers = usage.get_db().summary(hours=1)["providers"]
        assert "groq" in providers and providers["groq"]["calls"] == 1
        assert learning.get_store().stats()["interactions"] == 1

    def test_goal_text_is_not_lost(self):
        """Without the goal the memory degrades to a bare success counter:
        no vocabulary, no similarity retrieval, no self-correction."""
        from loomweaver import learning
        hub.record_outcome("groq", "m-a", True, 0.1,
                           goal="summarise the quarterly invoice")
        vocab = learning.get_store().profile()["vocabulary"]
        assert "invoice" in vocab or "quarterly" in vocab, f"vocab={vocab}"

    def test_goal_is_optional_for_pure_metrics(self):
        """Quota/usage bookkeeping should not require a prompt."""
        from loomweaver import learning
        hub.record_outcome("groq", "m-a", True, 0.1)
        assert learning.get_store().stats()["interactions"] == 0

    def test_a_broken_usage_db_does_not_stop_learning(self, monkeypatch):
        from loomweaver import learning
        monkeypatch.setattr(hub._usage, "record",
                            lambda *a, **k: (_ for _ in ()).throw(RuntimeError("db")))
        hub.record_outcome("groq", "m-a", True, 0.1, goal="remember this")
        assert learning.get_store().stats()["interactions"] == 1


class TestServices:
    def test_reports_every_service(self):
        got = hub.services()
        assert set(got) == {"semantic_cache", "quota_ledger", "key_rotation",
                            "usage", "learning"}
        assert all(isinstance(v, bool) for v in got.values())

    def test_a_broken_probe_reports_false_not_raise(self, monkeypatch):
        monkeypatch.setattr(hub._ql, "get_ledger",
                            lambda: (_ for _ in ()).throw(RuntimeError("x")))
        assert hub.services()["quota_ledger"] is False


class TestServerSharesTheResolver:
    """/v1/chat/completions must resolve providers the same way the CLI does.

    Regression: the HTTP pre-flight called get_providers() with only a
    hand-maintained whitelist of env vars, so an arbitrary
    <PREFIX>_API_KEY + <PREFIX>_BASE_URL provider worked everywhere except over
    HTTP, where it returned 502 "no providers configured".
    """

    def test_arbitrary_provider_resolves_through_the_server_path(self, monkeypatch):
        import server
        monkeypatch.setenv("ACME_API_KEY", "ak_live_1")
        monkeypatch.setenv("ACME_BASE_URL", "https://llm.acme.io/v1")
        creds = server._creds_from_env()          # the whitelist misses ACME_*
        assert "ACME_API_KEY" not in (creds or {})
        # ...but the resolver routing actually uses still finds it
        names = [p["name"] for p in server.core.build_providers(creds)]
        assert "acme" in names

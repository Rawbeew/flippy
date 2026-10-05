"""Universal bring-your-own-keys: nobody is limited to the bundled brands.

flippy speaks the OpenAI chat/completions wire format, so any endpoint that does
too is a provider. These tests pin the four registration paths and the one
behaviour that must never happen: picking up a credential that has no endpoint.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import flippy_providers as fp


def names(env):
    return fp.provider_names(env)


class TestBuiltinCatalog:
    def test_every_builtin_activates_on_its_own_key(self):
        """Each catalog entry must actually appear when its key is set."""
        for spec in fp.BUILTIN_PROVIDERS:
            if spec.get("no_key"):
                continue
            env = {spec["key_env"][0]: "k" * 12}
            assert spec["name"] in names(env), f"{spec['name']} never registered"

    def test_local_providers_opt_in_via_base_url_only(self):
        assert "ollama" not in names({})
        assert "ollama" in names({"OLLAMA_BASE_URL": "http://localhost:11434"})
        p = next(x for x in fp.get_providers({"OLLAMA_BASE_URL": "http://localhost:11434"})
                 if x["name"] == "ollama")
        assert p["url"] == "http://localhost:11434/chat/completions"

    def test_nothing_configured_means_nothing_routed(self):
        assert fp.get_providers({}) == []

    def test_catalog_describes_every_provider_and_its_switch(self):
        rows = fp.describe_catalog()
        assert len(rows) >= 20, "the catalog should cover the market, not five brands"
        for row in rows:
            assert row["activate_with"], f"{row['name']} has no documented switch"
        assert any(r["name"] == "<any>" for r in rows), "must document the generic path"

    def test_models_can_be_overridden_per_provider(self):
        env = {"GROQ_KEY": "gsk_x", "GROQ_MODELS": "my-fine-tune,other"}
        p = next(x for x in fp.get_providers(env) if x["name"] == "groq")
        assert p["models"] == ["my-fine-tune", "other"]


class TestArbitraryProviderDiscovery:
    """The headline capability: an endpoint flippy has never heard of."""

    def test_api_key_plus_base_url_registers_itself(self):
        env = {"ACME_API_KEY": "ak_live_1", "ACME_BASE_URL": "https://llm.acme.io/v1"}
        assert "acme" in names(env)
        p = next(x for x in fp.get_providers(env) if x["name"] == "acme")
        assert p["url"] == "https://llm.acme.io/v1/chat/completions"
        assert p["key"] == "ak_live_1"

    def test_every_credential_suffix_form_is_accepted(self):
        for suffix in ("_API_KEY", "_APIKEY", "_KEY", "_TOKEN"):
            env = {"NORDIC" + suffix: "k", "NORDIC_BASE_URL": "https://n.ai/v1"}
            assert "nordic" in names(env), f"{suffix} form was not recognised"

    def test_every_endpoint_variable_form_is_accepted(self):
        for suffix in ("_BASE_URL", "_API_BASE", "_BASE", "_ENDPOINT", "_URL"):
            env = {"NORDIC_API_KEY": "k", "NORDIC" + suffix: "https://n.ai/v1"}
            assert "nordic" in names(env), f"{suffix} form was not recognised"

    def test_prefix_is_parsed_correctly_not_greedily(self):
        """ACME_API_KEY must yield prefix ACME, not ACME_API."""
        env = {"ACME_API_KEY": "k", "ACME_BASE_URL": "https://x/v1",
               "ACME_MODELS": "acme-large"}
        p = next(x for x in fp.get_providers(env) if x["name"] == "acme")
        assert p["env_key"] == "ACME_API_KEY"
        assert p["models"] == ["acme-large"]

    def test_a_key_with_no_endpoint_is_ignored(self):
        """A Stripe/SendGrid key in the environment must not become an LLM."""
        env = {"STRIPE_API_KEY": "sk_test_secret", "SENDGRID_TOKEN": "SG.x"}
        assert "stripe" not in names(env)
        assert "sendgrid" not in names(env)

    def test_known_brands_are_not_double_registered(self):
        env = {"GROQ_API_KEY": "gsk_x", "GROQ_BASE_URL": "https://api.groq.com/openai/v1"}
        assert names(env).count("groq") == 1

    def test_infrastructure_credentials_are_never_treated_as_providers(self):
        env = {}
        for k in ("GITHUB", "AWS", "SLACK", "NPM", "AZURE", "DOCKER"):
            env[k + "_API_KEY"] = "secret"
            env[k + "_BASE_URL"] = "https://x/v1"
        assert names(env) == []

    def test_multiple_keys_split_into_a_rotation_list(self):
        env = {"ACME_API_KEY": "k1, k2 ,k3", "ACME_BASE_URL": "https://x/v1"}
        p = next(x for x in fp.get_providers(env) if x["name"] == "acme")
        assert p["keys"] == ["k1", "k2", "k3"] and p["key"] == "k1"


class TestDocumentedCustomContract:
    """The pre-existing `custom` name must keep working (tests/test_providers.py)."""

    def test_custom_still_wins_over_the_brand_entry(self):
        env = {"OPENAI_API_BASE": "https://gateway.example.com/v1",
               "OPENAI_API_KEY": "sk-mine"}
        got = names(env)
        assert got == ["custom"]
        assert "openai" not in got

    def test_brand_provider_works_when_no_custom_base_is_set(self):
        assert "openai" in names({"OPENAI_API_KEY": "sk-real"})


class TestCatalogFile:
    def test_json_catalog_registers_providers(self, tmp_path):
        path = tmp_path / "providers.json"
        path.write_text(json.dumps([
            {"name": "corp-llm", "base_url": "https://internal.corp/v1",
             "api_key_env": "CORP_LLM_TOKEN", "models": ["corp-70b"]},
            {"name": "keyless-lab", "base_url": "http://lab.local:8000/v1",
             "no_key": True},
            {"name": "unusable", "base_url": "https://no-cred/v1"},
        ]))
        env = {"FLIPPY_PROVIDERS_JSON": str(path), "CORP_LLM_TOKEN": "t0k"}
        got = names(env)
        assert "corp-llm" in got, "catalog entry with a resolvable key"
        assert "keyless-lab" in got, "explicitly keyless entry"
        assert "unusable" not in got, "no credential and not keyless -> skip"

    def test_a_malformed_catalog_degrades_to_nothing(self, tmp_path):
        path = tmp_path / "bad.json"
        path.write_text("{not json at all")
        assert fp.get_providers({"FLIPPY_PROVIDERS_JSON": str(path)}) == []

    def test_a_missing_catalog_file_is_not_an_error(self):
        assert fp.get_providers({"FLIPPY_PROVIDERS_JSON": "/nope/none.json"}) == []


class TestNumberedProviders:
    def test_numbered_entry_registers_under_its_own_name(self):
        env = {"FLIPPY_PROVIDER_1_NAME": "BankLLM",
               "FLIPPY_PROVIDER_1_BASE_URL": "https://internal.bank/v1",
               "FLIPPY_PROVIDER_1_API_KEY": "bk_1",
               "FLIPPY_PROVIDER_1_MODELS": "bank-7b"}
        assert "bankllm" in names(env)
        assert "flippy_provider_1" not in names(env), "must not double-register"

    def test_numbered_entries_without_a_base_url_are_skipped(self):
        env = {"FLIPPY_PROVIDER_1_NAME": "Nope",
               "FLIPPY_PROVIDER_1_API_KEY": "bk_1"}
        assert "nope" not in names(env)


class TestOrdering:
    def test_primary_then_free_then_paid(self):
        env = {"OPENROUTER_KEY": "sk-or-x", "GROQ_KEY": "gsk_x",
               "XAI_API_KEY": "xai_x"}
        got = fp.get_providers(env)
        assert got[0]["name"] == "openrouter", "primary must route first"
        costs = [p["cost"] for p in got]
        assert costs == sorted(costs, key=lambda c: 0 if c == "free" else 1)

    def test_ordering_is_stable_across_calls(self):
        env = {"GROQ_KEY": "g", "MISTRAL_API_KEY": "m", "XAI_API_KEY": "x"}
        assert names(env) == names(env)

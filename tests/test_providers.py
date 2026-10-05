"""Provider-agnostic surface: no user is forced to use a specific brand.

A user with their own OpenAI-compatible (or Anthropic-via-gateway) endpoint
should be able to run flippy with ONLY that endpoint configured — no Groq,
no OpenRouter, no brand required. flippy speaks the OpenAI chat/completions
wire format, so ANY endpoint that does too works through the generic `custom`
provider (OPENAI_API_BASE + OPENAI_API_KEY, or ANTHROPIC_BASE_URL +
ANTHROPIC_API_KEY).
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import flippy_providers as fp


class TestGenericCustomProvider:
    def test_openai_compatible_env_alone_yields_custom(self):
        env = {"OPENAI_API_BASE": "https://gateway.example.com/v1",
               "OPENAI_API_KEY": "sk-my-custom-key",
               "OPENAI_MODELS": "gpt-4o,claude-sonnet-4"}
        provs = fp.get_providers(env)
        assert [p["name"] for p in provs] == ["custom"]
        c = provs[0]
        assert c["models"] == ["gpt-4o", "claude-sonnet-4"]
        assert c["url"].endswith("/chat/completions")

    def test_anthropic_style_env_alone_yields_custom(self):
        env = {"ANTHROPIC_BASE_URL": "https://api.anthropic.com/v1",
               "ANTHROPIC_API_KEY": "sk-ant-xxx"}
        provs = fp.get_providers(env)
        names = [p["name"] for p in provs]
        assert "custom" in names
        c = next(p for p in provs if p["name"] == "custom")
        assert "api.anthropic.com" in c["url"]

    def test_custom_coexists_with_brand_keys(self):
        env = {"GROQ_KEY": "gsk_x", "OPENAI_API_BASE": "https://x/v1",
               "OPENAI_API_KEY": "sk-y"}
        names = sorted(p["name"] for p in fp.get_providers(env))
        assert "groq" in names and "custom" in names

    def test_default_model_when_models_unset(self):
        env = {"OPENAI_API_BASE": "https://x/v1", "OPENAI_API_KEY": "sk-y"}
        c = next(p for p in fp.get_providers(env) if p["name"] == "custom")
        assert c["models"], "custom provider must always name at least one model"
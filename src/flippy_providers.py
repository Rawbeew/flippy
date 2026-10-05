"""providers.py — THE canonical provider registry for flippy.

One source of truth consumed by ai_failover (CLI router), loomweaver.core
(harness), and aihub (litellm hub). Provider dicts carry everything each
consumer needs; consumers filter/shape rather than re-declare.

Provider dict shape:
    {
        "name":   str,          # canonical id: groq, nvidia, openrouter...
        "url":    str,          # OpenAI-compatible chat/completions endpoint
        "key":    str,          # bearer token
        "models": [str, ...],   # model IDs this provider serves
        "cost":   "free"|"paid",
        "primary": bool,        # tried first
        "single":  bool,        # True = model embedded in URL (Cloudflare)
        "litellm_base": str,    # api base for litellm (aihub)
        "env_key": str,         # env var name the key comes from
    }

Keys come from environment variables only — never hardcode.
"""
import os

UA = "flippy/0.1.0 (github.com/Rawbeew/flippy)"


def split_keys(raw: str) -> list[str]:
    """Parse a comma-separated key list into individual keys.

    Whitespace-tolerant; empty segments dropped. Single value -> [value].
    """
    return [k.strip() for k in raw.split(",") if k.strip()]


def get_providers(env=None):
    """Build the provider list from environment variables.

    Free-first ordering: primary first, then free, then paid.
    Returns [] when no keys are configured.
    """
    e = env if env is not None else os.environ
    P = []

    def base(env_key: str) -> dict | None:
        """Common key handling: comma-separated env values become 'keys'
        (rotation list); 'key' keeps the first one for backwards compat."""
        raw = e.get(env_key)
        if not raw:
            return None
        keys = split_keys(raw)
        return {"keys": keys, "key": keys[0], "env_key": env_key}

    d = base("OPENROUTER_KEY")
    if d:
        P.append({
            "name": "openrouter", "cost": "free", "primary": True,
            "url": "https://openrouter.ai/api/v1/chat/completions",
            "litellm_base": "https://openrouter.ai/api/v1",
            **d,
            "models": ["stealth/ox-alpha"],
        })
    d = base("FREEINFERENCE_KEY")
    if d:
        P.append({
            "name": "freeinference", "cost": "free",
            "url": "https://freeinference.org/v1/chat/completions",
            "litellm_base": "https://freeinference.org/v1",
            **d,
            "models": ["minimax-m3", "qwen3.6-35b", "deepseek-v4-flash", "glm-5.1"],
        })
    d = base("CLOUDFLARE_TOKEN")
    if d and e.get("CLOUDFLARE_ACCOUNT_ID"):
        acc = e["CLOUDFLARE_ACCOUNT_ID"]
        P.append({
            "name": "cloudflare", "cost": "free", "single": True,
            "url": f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
            "litellm_base": None,
            **d,
            "models": ["@cf/meta/llama-3.3-70b-instruct-fp8-fast"],
        })
    d = base("NVIDIA_KEY")
    if d:
        P.append({
            "name": "nvidia", "cost": "free",
            "url": "https://integrate.api.nvidia.com/v1/chat/completions",
            "litellm_base": "https://integrate.api.nvidia.com/v1",
            **d,
            "models": ["nvidia/llama-3.3-nemotron-super-49b-v1"],
        })
    d = base("GROQ_KEY")
    if d:
        P.append({
            "name": "groq", "cost": "free",
            "url": "https://api.groq.com/openai/v1/chat/completions",
            "litellm_base": "https://api.groq.com/openai/v1",
            **d,
            "models": ["openai/gpt-oss-20b", "openai/gpt-oss-120b"],
        })

    # Generic OpenAI-compatible (or Anthropic-via-compatible-gateway) custom
    # endpoint — flippy speaks the OpenAI chat/completions wire format, so any
    # provider that does too can be added WITHOUT a brand being required.
    # Set OPENAI_API_BASE (or ANTHROPIC_BASE_URL) + OPENAI_API_KEY (or
    # ANTHROPIC_API_KEY) + OPENAI_MODELS (comma-separated). No Groq needed.
    custom_base = e.get("OPENAI_API_BASE") or e.get("ANTHROPIC_BASE_URL")
    custom_key = e.get("OPENAI_API_KEY") or e.get("ANTHROPIC_API_KEY")
    if custom_base and custom_key:
        models = [m.strip() for m in (e.get("OPENAI_MODELS") or "").split(",") if m.strip()]
        if not models:
            models = ["gpt-4o-mini"]  # sane default for an OpenAI-compatible endpoint
        custom_url = custom_base.rstrip("/")
        if not custom_url.endswith("/chat/completions"):
            custom_url += "/chat/completions"
        P.append({
            "name": "custom", "cost": "free",
            "url": custom_url,
            "litellm_base": custom_base.rstrip("/"),
            "key": custom_key, "keys": [custom_key],
            "env_key": "OPENAI_API_KEY" if e.get("OPENAI_API_KEY") else "ANTHROPIC_API_KEY",
            "models": models,
        })

    return sorted(P, key=lambda p: (0 if p.get("primary") else 1, p["name"]))


def order_free_first(providers):
    return sorted(providers, key=lambda p: (0 if p.get("primary") else 1, p["name"]))

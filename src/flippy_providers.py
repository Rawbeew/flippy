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

UA = "OpenAI File Downloader, XaiImageApiFetch/1.0"


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

    return sorted(P, key=lambda p: (0 if p.get("primary") else 1, p["name"]))


def order_free_first(providers):
    return sorted(providers, key=lambda p: (0 if p.get("primary") else 1, p["name"]))

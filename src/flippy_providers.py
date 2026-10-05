"""flippy_providers.py — THE canonical provider registry.

One source of truth consumed by ai_failover (CLI router), loomweaver.core
(harness), server.py and aihub. Consumers filter/shape; they never re-declare.

flippy speaks the OpenAI chat/completions wire format, so **any** endpoint that
does too is a provider. You are not limited to the built-ins: there are four
ways to add one, in increasing order of explicitness.

1. A built-in, activated by setting its key:
       OPENROUTER_KEY, FREEINFERENCE_KEY, CLOUDFLARE_TOKEN + CLOUDFLARE_ACCOUNT_ID,
       NVIDIA_KEY, GROQ_KEY, OPENAI_API_KEY, TOGETHER_API_KEY, FIREWORKS_API_KEY,
       DEEPINFRA_API_TOKEN, MISTRAL_API_KEY, CEREBRAS_API_KEY, SAMBANOVA_API_KEY,
       XAI_API_KEY, PERPLEXITY_API_KEY, MOONSHOT_API_KEY, DASHSCOPE_API_KEY,
       SILICONFLOW_API_KEY, HF_TOKEN, ANTHROPIC_API_KEY (+ ANTHROPIC_BASE_URL)

2. A local/self-hosted endpoint, activated by its base URL:
       OLLAMA_BASE_URL=http://localhost:11434   LMSTUDIO_BASE_URL=http://localhost:1234
       VLLM_BASE_URL=http://localhost:8000      OPENAI_API_BASE=https://gateway/v1

3. Any `<PREFIX>_API_KEY` + `<PREFIX>_BASE_URL` pair you invent:
       ACME_API_KEY=...  ACME_BASE_URL=https://llm.acme.io/v1
       ACME_MODELS=acme-large,acme-mini            (optional; defaults below)
   Registers a provider literally named `acme`.

4. A catalog file for anything the env-var conventions cannot express:
       FLIPPY_PROVIDERS_JSON=/etc/flippy/providers.json
   Shape: [{"name": "...", "base_url": "...", "api_key_env": "...",
            "models": ["..."], "cost": "free", "primary": false}, ...]

Every provider dict carries:
    name, url, key, keys[], models[], cost, primary, single, litellm_base, env_key

Keys come from the environment (or the credentials file) only — never hardcoded.
"""
import json
import os
import re

UA = "flippy/0.1.0 (github.com/Rawbeew/flippy)"

# Models to assume when a provider is configured without an explicit list.
DEFAULT_MODELS = ["gpt-4o-mini"]


def split_keys(raw: str) -> list[str]:
    """Parse a comma-separated key list into individual keys.

    Whitespace-tolerant; empty segments dropped. Single value -> [value].
    """
    return [k.strip() for k in raw.split(",") if k.strip()]


def _split_models(raw) -> list[str]:
    if not raw:
        return []
    return [m.strip() for m in str(raw).split(",") if m.strip()]


def _norm_base(base: str) -> str:
    """Normalize an api base into a chat/completions URL."""
    url = base.strip().rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


# ---------------------------------------------------------------------------
# Built-in catalog. `key_env` may list alternatives, tried in order.
# `models_env` names the variable that overrides the model list.
# ---------------------------------------------------------------------------

BUILTIN_PROVIDERS = [
    {"name": "openrouter", "cost": "free", "primary": True,
     "key_env": ("OPENROUTER_KEY",), "models_env": "OPENROUTER_MODELS",
     "base": "https://openrouter.ai/api/v1",
     "models": ["stealth/ox-alpha"]},

    {"name": "freeinference", "cost": "free",
     "key_env": ("FREEINFERENCE_KEY",), "models_env": "FREEINFERENCE_MODELS",
     "base": "https://freeinference.org/v1",
     "models": ["minimax-m3", "qwen3.6-35b", "deepseek-v4-flash", "glm-5.1"]},

    {"name": "groq", "cost": "free",
     "key_env": ("GROQ_KEY", "GROQ_API_KEY"), "models_env": "GROQ_MODELS",
     "base": "https://api.groq.com/openai/v1",
     "models": ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]},

    {"name": "nvidia", "cost": "free",
     "key_env": ("NVIDIA_KEY", "NVIDIA_API_KEY"), "models_env": "NVIDIA_MODELS",
     "base": "https://integrate.api.nvidia.com/v1",
     "models": ["nvidia/llama-3.3-nemotron-super-49b-v1"]},

    {"name": "openai", "cost": "paid",
     "key_env": ("OPENAI_API_KEY",), "models_env": "OPENAI_MODELS",
     "base_env": "OPENAI_API_BASE", "base": "https://api.openai.com/v1",
     "models": ["gpt-4o-mini", "gpt-4o"]},

    {"name": "anthropic", "cost": "paid",
     "key_env": ("ANTHROPIC_API_KEY",), "models_env": "ANTHROPIC_MODELS",
     "base_env": "ANTHROPIC_BASE_URL", "base": "https://api.anthropic.com/v1",
     "models": ["claude-sonnet-4", "claude-3-5-haiku"]},

    {"name": "together", "cost": "paid",
     "key_env": ("TOGETHER_API_KEY",), "models_env": "TOGETHER_MODELS",
     "base": "https://api.together.xyz/v1",
     "models": ["meta-llama/Llama-3.3-70B-Instruct-Turbo"]},

    {"name": "fireworks", "cost": "paid",
     "key_env": ("FIREWORKS_API_KEY",), "models_env": "FIREWORKS_MODELS",
     "base": "https://api.fireworks.ai/inference/v1",
     "models": ["accounts/fireworks/models/llama-v3p1-70b-instruct"]},

    {"name": "deepinfra", "cost": "free",
     "key_env": ("DEEPINFRA_API_TOKEN", "DEEPINFRA_API_KEY"),
     "models_env": "DEEPINFRA_MODELS",
     "base": "https://api.deepinfra.com/v1/openai",
     "models": ["meta-llama/Llama-3.3-70B-Instruct"]},

    {"name": "mistral", "cost": "paid",
     "key_env": ("MISTRAL_API_KEY",), "models_env": "MISTRAL_MODELS",
     "base": "https://api.mistral.ai/v1",
     "models": ["mistral-small-latest", "open-mistral-nemo"]},

    {"name": "cerebras", "cost": "free",
     "key_env": ("CEREBRAS_API_KEY",), "models_env": "CEREBRAS_MODELS",
     "base": "https://api.cerebras.ai/v1",
     "models": ["llama3.1-8b", "llama-3.3-70b"]},

    {"name": "sambanova", "cost": "free",
     "key_env": ("SAMBANOVA_API_KEY",), "models_env": "SAMBANOVA_MODELS",
     "base": "https://api.sambanova.ai/v1",
     "models": ["Meta-Llama-3.1-8B-Instruct"]},

    {"name": "xai", "cost": "paid",
     "key_env": ("XAI_API_KEY",), "models_env": "XAI_MODELS",
     "base": "https://api.x.ai/v1",
     "models": ["grok-2-latest"]},

    {"name": "perplexity", "cost": "paid",
     "key_env": ("PERPLEXITY_API_KEY",), "models_env": "PERPLEXITY_MODELS",
     "base": "https://api.perplexity.ai",
     "models": ["llama-3.1-sonar-small-128k-chat"]},

    {"name": "moonshot", "cost": "paid",
     "key_env": ("MOONSHOT_API_KEY",), "models_env": "MOONSHOT_MODELS",
     "base": "https://api.moonshot.cn/v1",
     "models": ["moonshot-v1-8k"]},

    {"name": "dashscope", "cost": "free",
     "key_env": ("DASHSCOPE_API_KEY",), "models_env": "DASHSCOPE_MODELS",
     "base": "https://dashscope.aliyuncs.com/compatible-mode/v1",
     "models": ["qwen-plus", "qwen-turbo"]},

    {"name": "siliconflow", "cost": "free",
     "key_env": ("SILICONFLOW_API_KEY",), "models_env": "SILICONFLOW_MODELS",
     "base": "https://api.siliconflow.cn/v1",
     "models": ["Qwen/Qwen2.5-7B-Instruct"]},

    {"name": "huggingface", "cost": "free",
     "key_env": ("HF_TOKEN", "HUGGINGFACE_API_KEY"), "models_env": "HF_MODELS",
     "base": "https://router.huggingface.co/v1",
     "models": ["meta-llama/Llama-3.3-70B-Instruct"]},

    # Local / self-hosted — no key required, activated by base URL.
    {"name": "ollama", "cost": "free", "no_key": True,
     "base_env": "OLLAMA_BASE_URL", "models_env": "OLLAMA_MODELS",
     "models": ["llama3.2", "qwen2.5"]},
    {"name": "lmstudio", "cost": "free", "no_key": True,
     "base_env": "LMSTUDIO_BASE_URL", "models_env": "LMSTUDIO_MODELS",
     "models": ["local-model"]},
    {"name": "vllm", "cost": "free", "no_key": True,
     "base_env": "VLLM_BASE_URL", "models_env": "VLLM_MODELS",
     "models": ["served-model"]},
]

# Cloudflare Workers AI embeds the model in the URL, so it is shaped by hand.
def _cloudflare(e):
    key = e.get("CLOUDFLARE_TOKEN") or e.get("CLOUDFLARE_API_TOKEN")
    acc = e.get("CLOUDFLARE_ACCOUNT_ID")
    if not (key and acc):
        return None
    model = (e.get("CLOUDFLARE_MODEL")
             or "@cf/meta/llama-3.3-70b-instruct-fp8-fast")
    return {
        "name": "cloudflare", "cost": "free", "single": True,
        "url": f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/{model}",
        "litellm_base": None,
        "keys": split_keys(key), "key": split_keys(key)[0],
        "env_key": "CLOUDFLARE_TOKEN", "models": [model],
    }


# Env-var prefixes that are NOT user providers, so auto-discovery stays clean.
_DISCOVERY_SKIP = {
    "OPENAI", "ANTHROPIC", "GROQ", "NVIDIA", "OPENROUTER", "FREEINFERENCE",
    "CLOUDFLARE", "TOGETHER", "FIREWORKS", "DEEPINFRA", "MISTRAL", "CEREBRAS",
    "SAMBANOVA", "XAI", "PERPLEXITY", "MOONSHOT", "DASHSCOPE", "SILICONFLOW",
    "HF", "HUGGINGFACE", "OLLAMA", "LMSTUDIO", "VLLM",
    "FLIPPY", "LOOMWEAVER", "AIHUB", "AWS", "AZURE", "GCP", "GOOGLE",
    "GITHUB", "GH", "SLACK", "STRIPE", "TELEGRAM", "PATH", "HOME", "USER",
    "SSH", "GPG", "NPM", "PYPI", "DOCKER", "KUBE",
}

# Credential suffixes, longest first. Deliberately NOT a single regex with
# alternation: a greedy `[A-Z0-9_]+` prefix would parse ACME_API_KEY as
# prefix "ACME_API" + suffix "_KEY", so the ACME_BASE_URL next to it would
# never be found. Matching the longest literal suffix keeps the prefix clean.
_KEY_SUFFIXES = ("_API_KEY", "_APIKEY", "_KEY", "_TOKEN")
_BASE_ENV_SUFFIXES = ("_BASE_URL", "_API_BASE", "_BASE", "_ENDPOINT", "_URL")
_PREFIX_RE = re.compile(r"^[A-Z][A-Z0-9_]{1,40}$")
# FLIPPY_PROVIDER_<n>_* is claimed by _numbered(); generic discovery must not
# also register it as a provider literally named "flippy_provider_1".
_NUMBERED_RE = re.compile(r"^FLIPPY_PROVIDER_\d+$")


def _key_prefix(var):
    """'ACME_API_KEY' -> 'ACME'; returns None for non-credential variables."""
    if not var or var.upper() != var:
        return None
    for suf in _KEY_SUFFIXES:
        if var.endswith(suf):
            prefix = var[: -len(suf)]
            return prefix if _PREFIX_RE.match(prefix) else None
    return None


def _discover_custom(e):
    """Register any `<PREFIX>_API_KEY` + `<PREFIX>_BASE_URL` pair.

    This is the "bring your own keys, any provider" path: no code change and no
    catalog entry is needed to route through a gateway flippy has never heard of.
    """
    found = {}
    for name, value in e.items():
        prefix = _key_prefix(name)
        if not prefix or not value:
            continue
        if prefix in _DISCOVERY_SKIP or _NUMBERED_RE.match(prefix):
            continue
        base = next((e.get(prefix + suf) for suf in _BASE_ENV_SUFFIXES
                     if e.get(prefix + suf)), None)
        if not base:
            continue  # a key with no endpoint tells us nothing routable
        found.setdefault(prefix, {"key": value, "base": base})
    out = []
    for prefix, cfg in found.items():
        models = _split_models(e.get(prefix + "_MODELS")) or list(DEFAULT_MODELS)
        out.append({
            "name": prefix.lower(), "cost": e.get(prefix + "_COST", "free"),
            "url": _norm_base(cfg["base"]),
            "litellm_base": cfg["base"].rstrip("/"),
            "keys": split_keys(cfg["key"]), "key": split_keys(cfg["key"])[0],
            "env_key": next((prefix + suf for suf in _KEY_SUFFIXES
                             if e.get(prefix + suf)), prefix + "_API_KEY"),
            "models": models,
        })
    return out


def _catalog_file(e):
    """Load an operator catalog: [{"name","base_url","api_key_env","models"}, ...]"""
    path = e.get("FLIPPY_PROVIDERS_JSON")
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, encoding="utf-8") as f:
            entries = json.load(f)
    except (OSError, ValueError):
        return []
    out = []
    for entry in entries if isinstance(entries, list) else []:
        if not isinstance(entry, dict):
            continue
        base = entry.get("base_url") or entry.get("base")
        if not base:
            continue
        key = ""
        if entry.get("api_key"):
            key = entry["api_key"]
        elif entry.get("api_key_env"):
            key = e.get(entry["api_key_env"], "")
        if not key and not entry.get("no_key"):
            continue  # no credential and not explicitly keyless -> skip
        keys = split_keys(key) if key else [""]
        out.append({
            "name": str(entry.get("name") or "custom"),
            "cost": entry.get("cost", "free"),
            "primary": bool(entry.get("primary")),
            "single": bool(entry.get("single")),
            "url": base if entry.get("single") else _norm_base(base),
            "litellm_base": base.rstrip("/"),
            "keys": keys, "key": keys[0],
            "env_key": entry.get("api_key_env", ""),
            "models": _split_models(entry.get("models")) or list(DEFAULT_MODELS),
        })
    return out


def _numbered(e):
    """Explicit extra endpoints: FLIPPY_PROVIDER_<n>_{NAME,BASE_URL,API_KEY,MODELS}."""
    out, idx = [], 1
    while True:
        p = f"FLIPPY_PROVIDER_{idx}_"
        if not any(k.startswith(p) for k in e):
            if idx > 32:
                break
            idx += 1
            if idx > 32:
                break
            continue
        base = e.get(p + "BASE_URL") or e.get(p + "BASE")
        key = e.get(p + "API_KEY") or e.get(p + "KEY") or ""
        if base:
            keys = split_keys(key) if key else [""]
            out.append({
                "name": (e.get(p + "NAME") or f"custom{idx}").lower(),
                "cost": e.get(p + "COST", "free"),
                "primary": e.get(p + "PRIMARY") == "1",
                "url": _norm_base(base), "litellm_base": base.rstrip("/"),
                "keys": keys, "key": keys[0], "env_key": p + "API_KEY",
                "models": _split_models(e.get(p + "MODELS")) or list(DEFAULT_MODELS),
            })
        idx += 1
        if idx > 32:
            break
    return out


def get_providers(env=None):
    """Build the provider list from the environment.

    Ordering: primary first, then free, then paid, then alphabetical — so
    cold-start behaviour is stable and predictable. Returns [] when nothing is
    configured.
    """
    e = dict(os.environ if env is None else env)
    P = []

    for spec in BUILTIN_PROVIDERS:
        key = ""
        key_env_used = ""
        for var in spec.get("key_env", ()):
            if e.get(var):
                key, key_env_used = e[var], var
                break
        base = e.get(spec.get("base_env", ""), "") or spec.get("base", "")
        if spec.get("no_key"):
            if not e.get(spec.get("base_env", "")):
                continue  # local providers opt in via their base URL
        elif not key:
            continue
        if not base:
            continue
        keys = split_keys(key) if key else [""]
        P.append({
            "name": spec["name"], "cost": spec.get("cost", "free"),
            "primary": spec.get("primary", False),
            "url": _norm_base(base),
            "litellm_base": base.rstrip("/"),
            "keys": keys, "key": keys[0], "env_key": key_env_used,
            "models": _split_models(e.get(spec.get("models_env", ""), ""))
                      or list(spec["models"]),
        })

    # Generic OpenAI-compatible (or Anthropic-via-compatible-gateway) custom
    # endpoint. flippy speaks the OpenAI chat/completions wire format, so any
    # endpoint that does too works with no brand required: set OPENAI_API_BASE
    # (or ANTHROPIC_BASE_URL) + OPENAI_API_KEY (or ANTHROPIC_API_KEY), and
    # optionally OPENAI_MODELS. This is the documented "bring your own endpoint"
    # path and it owns the name `custom`.
    custom_base = e.get("OPENAI_API_BASE") or e.get("ANTHROPIC_BASE_URL")
    custom_key = e.get("OPENAI_API_KEY") or e.get("ANTHROPIC_API_KEY")
    if custom_base and custom_key:
        P.append({
            "name": "custom", "cost": "free",
            "url": _norm_base(custom_base),
            "litellm_base": custom_base.rstrip("/"),
            "keys": split_keys(custom_key), "key": split_keys(custom_key)[0],
            "env_key": ("OPENAI_API_KEY" if e.get("OPENAI_API_KEY")
                        else "ANTHROPIC_API_KEY"),
            "models": _split_models(e.get("OPENAI_MODELS")) or list(DEFAULT_MODELS),
        })
        # the user's gateway replaces the stock brand endpoint, so the brand
        # entry must not also register against the same credentials.
        taken = "openai" if e.get("OPENAI_API_BASE") else "anthropic"
        P = [p for p in P if p["name"] != taken]

    cf = _cloudflare(e)
    if cf:
        P.append(cf)

    P.extend(_discover_custom(e))
    P.extend(_catalog_file(e))
    P.extend(_numbered(e))

    # de-duplicate by name, first registration wins (built-ins beat discovery)
    seen, out = set(), []
    for p in P:
        if p["name"] in seen:
            continue
        seen.add(p["name"])
        out.append(p)
    return order_free_first(out)


def order_free_first(providers):
    """primary -> free -> paid, alphabetical inside each band."""
    return sorted(providers, key=lambda p: (
        0 if p.get("primary") else 1,
        0 if p.get("cost") == "free" else 1,
        p["name"]))


def provider_names(env=None):
    """Convenience: the configured provider names, in routing order."""
    return [p["name"] for p in get_providers(env)]


def describe_catalog():
    """Every provider flippy knows how to talk to, configured or not.

    Used by `flippy providers --all` and `doctor` to tell an operator exactly
    which variable would switch each one on.
    """
    rows = []
    for spec in BUILTIN_PROVIDERS:
        switch = " + ".join(spec.get("key_env", ()))
        if spec.get("no_key"):
            # keyless local endpoints are switched on by their base URL instead
            switch = spec.get("base_env", "")
        elif spec.get("base_env"):
            # a brand endpoint can also be repointed, but the key is the switch
            switch += f"  (repoint: {spec['base_env']})"
        rows.append({
            "name": spec["name"],
            "cost": spec.get("cost", "free"),
            "activate_with": switch,
            "models_env": spec.get("models_env", ""),
            "default_models": list(spec["models"]),
        })
    rows.append({"name": "cloudflare", "cost": "free",
                 "activate_with": "CLOUDFLARE_TOKEN + CLOUDFLARE_ACCOUNT_ID",
                 "models_env": "CLOUDFLARE_MODEL",
                 "default_models": ["@cf/meta/llama-3.3-70b-instruct-fp8-fast"]})
    rows.append({"name": "<any>", "cost": "free",
                 "activate_with": "<PREFIX>_API_KEY + <PREFIX>_BASE_URL",
                 "models_env": "<PREFIX>_MODELS",
                 "default_models": list(DEFAULT_MODELS)})
    rows.append({"name": "<catalog>", "cost": "free",
                 "activate_with": "FLIPPY_PROVIDERS_JSON=/path/providers.json",
                 "models_env": "", "default_models": list(DEFAULT_MODELS)})
    rows.append({"name": "<extra>", "cost": "free",
                 "activate_with": "FLIPPY_PROVIDER_1_{NAME,BASE_URL,API_KEY,MODELS}",
                 "models_env": "", "default_models": list(DEFAULT_MODELS)})
    return rows

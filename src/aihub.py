#!/usr/bin/env python3
"""
aihub.py — unified AI wiring: router, vision, tools, embeddings/RAG, TTS, STT.

Backbone: litellm Router across FREE OpenAI-compatible providers with automatic
failover (429/5xx, free-first ordering, cooldown after failures). On top of that:
vision, tool/function calling, embeddings + local vector store for RAG, TTS, STT.

Credentials: read from environment variables. At minimum set ONE free provider
key (FREEINFERENCE_KEY, GROQ_KEY, NVIDIA_KEY, or Cloudflare).

Usage:
    python aihub.py --health               # list providers + capability checks
    python aihub.py --chat "your prompt"
    python aihub.py --simple "your prompt" # route to cheapest model
    python aihub.py --tooltest             # exercise function calling
    python aihub.py --vision image.jpg "what's in this?"
    python aihub.py --embed "text"
    python aihub.py --rag add "text"       # add to vector store
    python aihub.py --rag query "q"        # retrieve top-k
    python aihub.py --rag-chat "q"         # retrieve then answer
    python aihub.py --summarize "text"
    python aihub.py --tts "hello world"
    python aihub.py --stt audio.mp3

Requires:
    pip install litellm edge-tts pillow
"""
import os, sys, json, time, argparse, base64, hashlib, math, re as _re


# ---------- Secret hygiene (Tier-3) ----------
# Legacy brand variables, kept as a floor so redaction still works when the
# registry cannot be imported.
_LEGACY_SECRET_ENV_VARS = ("FREEINFERENCE_KEY", "GROQ_KEY", "NVIDIA_KEY",
                           "CLOUDFLARE_TOKEN", "OPENROUTER_KEY",
                           "ANTHROPIC_API_KEY")

# Any variable that looks like a credential. Needed because the provider
# surface is now open-ended: a hardcoded six-name list cannot cover
# MISTRAL_API_KEY, TOGETHER_API_KEY, or an operator's own ACME_API_KEY.
_CRED_VAR_RE = _re.compile(
    r"^[A-Z][A-Z0-9_]*_(API_KEY|APIKEY|KEY|TOKEN|SECRET|PASSWORD)$")


def _secret_env_vars():
    """Every environment variable that holds a credential.

    Derived rather than hardcoded: the registry is open-ended, so a fixed list
    would silently stop covering new providers and leave their keys echoable in
    an error message. Never raises — redaction must not become the failure.
    """
    found = set(_LEGACY_SECRET_ENV_VARS)
    try:
        found |= {v for v in os.environ if _CRED_VAR_RE.match(v or "")}
    except Exception:
        pass
    try:
        from flippy_providers import get_providers
        for p in get_providers():
            if p.get("env_key"):
                found.add(p["env_key"])
    except Exception:
        pass
    return tuple(found)


def _redact_secrets(text):
    """Strip ANY configured key, plus common key shapes, from a message before
    it reaches a log line or exception. The Full-Key rule: never echo a
    complete secret. Defensive — never raises, degrades to the raw input."""
    try:
        if not text:
            return text
        s = str(text)
        for env in _secret_env_vars():
            v = os.environ.get(env)
            if v:
                s = s.replace(v, "[REDACTED]")
        pattern = _re.compile(
            r"(sk-[A-Za-z0-9_-]{10,}|gsk_[A-Za-z0-9]{20,}|nvapi-[A-Za-z0-9_-]{10,}|"
            r"cfut_[A-Za-z0-9_-]{10,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|"
            r"AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{20,}|"
            r"xai-[a-z0-9]{20,}|eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})")
        return pattern.sub("[REDACTED]", s)
    except Exception:
        return str(text)


def _safe_message(exc):
    """Render an exception's message with any embedded secret scrubbed."""
    return _redact_secrets(getattr(exc, "message", None) or str(exc))


# ---------- Provider registry ----------
def build_router_models():
    """Derived from the canonical registry in providers.py."""
    from flippy_providers import get_providers
    out = []
    for p in get_providers():
        if p["name"] == "cloudflare":
            # litellm shape for CF differs; keep legacy tuple
            acc = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
            cf_base = f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai"
            for m in p["models"]:
                out.append((f"openai/{m}", m, p["key"], cf_base))
        elif p["name"] == "groq":
            for m in p["models"]:
                out.append((f"groq/{m}", m, p["key"], None))
        else:
            base = p.get("litellm_base")
            for m in p["models"]:
                out.append((f"openai/{m}", m, p["key"], base))
    return out


def _litellm_name(provider_name, model):
    """The litellm-prefixed model id. The prefix is per-provider knowledge."""
    if provider_name == "groq":
        return f"groq/{model}"
    return f"openai/{model}"


def build_deployments():
    """One litellm deployment per (LIVE key, model), across every provider.

    This is the join to the loomweaver engine. The naive version — one entry
    per provider using `provider["key"]` — pins every request to the first key
    in a comma-separated list and burns its quota while the rest sit idle, and
    it keeps handing out keys the rotation state has already retired.
    """
    from flippy_providers import get_providers
    from loomweaver import hub
    out = []
    for p in get_providers():
        name = p["name"]
        if name == "cloudflare":
            # litellm's Cloudflare shape puts the account in the base URL
            acc = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")
            p = dict(p)
            p["litellm_base"] = (
                f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai")
        out.extend(hub.key_deployments(p, lambda m, _n=name: _litellm_name(_n, m)))
    return out


def build_router():
    """Build a litellm Router with the configured providers. Returns (router, litellm)."""
    import litellm
    model_list = build_deployments()
    if not model_list:
        raise RuntimeError(
            "no providers configured. Any OpenAI-compatible endpoint works: set "
            "its key (e.g. GROQ_KEY, TOGETHER_API_KEY, MISTRAL_API_KEY), or point "
            "flippy at your own with <PREFIX>_API_KEY + <PREFIX>_BASE_URL. "
            "Run `flippy providers --all` for the full list.")
    return litellm.Router(
        model_list=model_list,
        set_verbose=False,
        num_retries=3,
        allowed_fails=2,
        cooldown_time=60,
        enable_pre_call_checks=True,
    ), litellm


# The engine bridge. Imported once at module scope so every call site shares
# the same view; guarded so aihub still imports if loomweaver is missing.
try:
    from loomweaver import hub
except Exception:  # pragma: no cover - aihub must remain importable alone
    hub = None


# ---------- Token-savings + smart routing ----------
def smart_chat(messages, simple=False, use_rag=False, top_k=3, model=None,
               cache=True, max_tokens=1024):
    """Route to the cheapest model that fits, optionally inject RAG context."""
    router, litellm = build_router()
    if use_rag:
        prompt = messages[-1]["content"] if messages else ""
        hits = rag_query(prompt, top_k=top_k)
        if hits:
            ctx = "\n\n".join("- " + h["text"] for h in hits)
            sys_msg = ("Use this retrieved context to answer. If it's irrelevant, "
                       "say so and answer generally.\nCONTEXT:\n" + ctx)
            messages = [{"role": "system", "content": sys_msg}] + messages
    if not model:
        model = "deepseek-v4-flash" if simple else "minimax-m3"

    # ---- the loomweaver engine, in front of and behind the litellm call ----
    # aihub is the front door; the cache, quota ledger, key rotation, usage
    # analytics and learning memory are the engine. They are not optional
    # decorations: without this join a near-duplicate prompt re-bills a
    # provider and a quota-exhausted provider keeps being tried until
    # litellm's own cooldown happens to notice.
    if hub is None:
        raise RuntimeError(
            "the loomweaver engine is required: aihub routes through litellm but "
            "relies on loomweaver for key rotation, the semantic cache, the quota "
            "ledger, usage analytics and the learning memory. Install the package "
            "so `from loomweaver import hub` resolves.")

    if cache:
        hit = hub.cache_get(messages)
        if hit:
            hub.record_outcome("cache", model, True, 0.0, cached=True)
            return {"content": hit["text"], "model": model, "prompt_tokens": 0,
                    "completion_tokens": 0, "cache_read": 0,
                    "cached": True, "similarity": hit["similarity"]}

    allowed, reason = hub.quota_check(model)
    if not allowed:
        raise RuntimeError(f"provider quota exhausted for {model}: {reason}")

    _t0 = hub.timed()
    try:
        resp = router.completion(model=model, messages=messages,
                                 max_tokens=max_tokens, caching=False)
    except Exception as e:
        status = getattr(e, "status_code", None) or getattr(e, "status", None)
        hub.record_outcome(model, model, False, hub.since(_t0), status=status,
                           goal=messages)
        try:
            from loomweaver import learning
            learning.note_failure(messages, _safe_message(e))
        except Exception:
            pass
        # Full-Key rule: an auth failure must never surface the configured key.
        raise RuntimeError(f"completion failed: {_safe_message(e)}") from None

    c = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage") or {}
    cache_read = 0
    try:
        ptd = usage.get("prompt_tokens_details") or {}
        if hasattr(ptd, "get"):
            cache_read = ptd.get("cached_tokens", 0)
    except Exception:
        cache_read = 0
    hub.record_outcome(model, resp.get("model") or model, True, hub.since(_t0),
                       usage=usage, status=200, goal=messages)
    if cache:
        hub.cache_put(messages, c, model_tag=model)
    return {"content": c, "model": resp.get("model"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cache_read": cache_read, "cached": False}


def summarize(text, max_words=80, model="deepseek-v4-flash"):
    """Cheap summarization — useful for compressing long contexts."""
    router, _ = build_router()
    try:
        resp = router.completion(model=model,
            messages=[{"role": "user", "content":
                f"Compress the following into a concise summary of at most {max_words} words. "
                f"Keep key facts, numbers, names:\n\n{text[:8000]}"}])
    except Exception as e:
        raise RuntimeError(f"summary failed: {_safe_message(e)}") from None
    return resp["choices"][0]["message"]["content"].strip()

def embed(texts):
    """bge-m3 from freeinference. Returns list of 1024-dim vectors."""
    import urllib.request, urllib.error
    key = os.environ.get("FREEINFERENCE_KEY")
    if not key:
        raise RuntimeError("FREEINFERENCE_KEY required for embeddings")
    if isinstance(texts, str):
        texts = [texts]
    body = {"model": "bge-m3", "input": texts}
    req = urllib.request.Request("https://freeinference.org/v1/embeddings",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json",
                 "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/126.0"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode())
            return [item["embedding"] for item in d["data"]]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"embed HTTP {e.code}: "
                           f"{_safe_message(e.read().decode()[:200])}")


# ---------- Local vector store (file-backed JSON) ----------
def _store_path():
    path = os.environ.get("AIHUB_VECTOR_STORE",
                          os.path.join(os.path.expanduser("~"), ".aihub_vectors.json"))
    return path


def _load_store():
    p = _store_path()
    if os.path.exists(p):
        return json.load(open(p, encoding="utf-8"))
    return []


def _save_store(store):
    json.dump(store, open(_store_path(), "w", encoding="utf-8"))


def rag_add(text, meta=None):
    vec = embed([text])[0]
    store = _load_store()
    rid = hashlib.md5(text.encode()).hexdigest()[:12]
    store = [s for s in store if s["id"] != rid]
    store.append({"id": rid, "text": text, "vector": vec, "meta": meta or {}})
    _save_store(store)
    return rid


def rag_query(q, top_k=3):
    qv = embed([q])[0]
    store = _load_store()
    if not store:
        return []
    def cos(a, b):
        dot = sum(x*y for x, y in zip(a, b))
        na = math.sqrt(sum(x*x for x in a)) or 1
        nb = math.sqrt(sum(x*x for x in b)) or 1
        return dot/(na*nb)
    scored = sorted(((cos(qv, r["vector"]), r) for r in store), key=lambda t: -t[0])
    return [{"text": s["text"], "score": round(sc, 4), "meta": s["meta"]}
            for sc, s in scored[:top_k]]


# ---------- TTS (edge-tts, no key needed; Groq orpheus if set) ----------
def tts(text, outpath=None):
    # zero-trust write guard: only ever write under the user's home TTS dir,
    # unless the operator explicitly opts into arbitrary paths. Refuse anything
    # that could overwrite a sensitive/absolute/traversing target. (aihub is
    # CLI-only today; this bounds a future agent-tool exposure too.)
    home = os.path.expanduser("~")
    tts_dir = os.path.join(home, "aihub_tts")
    os.makedirs(tts_dir, exist_ok=True)
    if outpath is None:
        outpath = os.path.join(tts_dir, "tts.mp3")
    elif os.environ.get("AIHUB_ALLOW_ARBITRARY_OUTPATH") != "1":
        ap = os.path.abspath(outpath)
        # STRICTLY under the dedicated TTS dir. No loose "under home" clause:
        # abspath("../../root.mp3") resolves inside home and would otherwise be
        # allowed as an arbitrary write. Only ~/aihub_tts (and basename-matches
        # there) are in scope.
        tts_abs = os.path.abspath(tts_dir)
        if not (ap == os.path.join(tts_abs, os.path.basename(outpath))
                or ap.startswith(tts_abs + os.sep)):
            raise ValueError(
                "refusing tts outpath outside ~/aihub_tts: use "
                "AIHUB_ALLOW_ARBITRARY_OUTPATH=1 to allow arbitrary writes "
                "(not recommended)")
    try:
        import edge_tts, asyncio
        async def _run():
            c = edge_tts.Communicate(text, "en-US-JennyNeural")
            await c.save(outpath)
        asyncio.run(_run())
        return outpath
    except Exception as e:
        print(f"[aihub] edge-tts failed: {_safe_message(e)}", file=sys.stderr)
    if os.environ.get("GROQ_KEY"):
        import urllib.request, urllib.error
        body = {"model": "canopylabs/orpheus-v1-english", "input": text, "voice": "tara"}
        req = urllib.request.Request("https://api.groq.com/openai/v1/audio/speech",
            data=json.dumps(body).encode(),
            headers={"Authorization": f"Bearer {os.environ['GROQ_KEY']}",
                     "Content-Type": "application/json",
                     "User-Agent": "Mozilla/5.0 Chrome/126.0"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                with open(outpath, "wb") as f:
                    f.write(r.read())
                return outpath
        except urllib.error.HTTPError as e:
            print(f"[aihub] groq orpheus failed ({e.code})", file=sys.stderr)
    raise RuntimeError("TTS failed (edge-tts unavailable, no Groq orpheus)")


# ---------- STT (Cloudflare whisper-large-v3-turbo) ----------
def stt(audio_path):
    import urllib.request, urllib.error
    acc = os.environ.get("CLOUDFLARE_ACCOUNT_ID")
    cf = os.environ.get("CLOUDFLARE_TOKEN")
    if not (acc and cf):
        raise RuntimeError("STT requires CLOUDFLARE_TOKEN + CLOUDFLARE_ACCOUNT_ID")
    with open(audio_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body = {"audio": b64}
    req = urllib.request.Request(
        f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/"
        "@cf/openai/whisper-large-v3-turbo",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {cf}",
                 "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            d = json.loads(r.read().decode())
            return d.get("result", {}).get("text") or d.get("text") or d
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"stt HTTP {e.code}: "
                           f"{_safe_message(e.read().decode()[:200])}")


# ---------- Vision ----------
def vision(image_path, prompt, model="minimax-m3"):
    """Use a chat completion with a base64-encoded image (vision-capable model)."""
    import urllib.request, urllib.error
    key = os.environ.get("FREEINFERENCE_KEY")
    if not key:
        raise RuntimeError("FREEINFERENCE_KEY required for vision in this build")
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    body = {"model": model,
            "messages": [{"role": "user",
                          "content": [{"type": "text", "text": prompt},
                                      {"type": "image_url",
                                       "image_url": {"url": f"data:image/jpeg;base64,{b64}"}}]}],
            "max_tokens": 1024}
    req = urllib.request.Request("https://freeinference.org/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as r:
            d = json.loads(r.read().decode())
            return d["choices"][0]["message"]["content"]
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"vision HTTP {e.code}: "
                           f"{_safe_message(e.read().decode()[:200])}")


# ---------- Tool / function-calling demo ----------
def tooltest():
    """Exercise function calling through the router and the guarded tool registry.

    Previously this hand-rolled a raw `urlopen` against one hardcoded provider:
    no SSRF guard, no redaction, no failover, and it bypassed the registry
    entirely. It now goes through the same litellm Router as `smart_chat`, and
    any tool the model asks for is dispatched via `loomweaver.tools`, so it is
    subject to the same allowlist, path jail and redaction as the agent.
    """
    router, _litellm = build_router()
    tools = [{"type": "function",
              "function": {"name": "get_weather",
                           "description": "Get current weather for a city",
                           "parameters": {"type": "object",
                                         "properties": {"city": {"type": "string"}},
                                         "required": ["city"]}}}]
    t0 = hub.timed()
    try:
        resp = router.completion(model="minimax-m3",
                                 messages=[{"role": "user",
                                            "content": "What's the weather in Lagos?"}],
                                 tools=tools, max_tokens=256, caching=False)
    except Exception as e:
        hub.record_outcome("minimax-m3", "minimax-m3", False, hub.since(t0))
        raise RuntimeError(f"tool call failed: {_safe_message(e)}") from None
    hub.record_outcome("minimax-m3", resp.get("model") or "minimax-m3", True,
                       hub.since(t0), usage=resp.get("usage") or {}, status=200)
    msg = resp["choices"][0]["message"]
    # If the model asked for a real registered tool, run it through the guards
    # rather than trusting the request.
    for call in (msg.get("tool_calls") or []):
        fn = (call.get("function") or {})
        name = fn.get("name") or call.get("name")
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except (ValueError, TypeError):
            args = {}
        try:
            from loomweaver import tools as _tools, observability as _obs
            if name in _tools.TOOLS:
                obs, intercepted = _obs.safe_invoke(name, args, _tools.dispatch)
                msg.setdefault("tool_results", {})[name] = (
                    "[intercepted]" if intercepted else obs[:500])
        except Exception as e:
            msg.setdefault("tool_results", {})[name] = f"unavailable: {_safe_message(e)}"
    return msg


# ---------- CLI ----------
# Engine commands that aihub delegates to the loomweaver CLI rather than
# re-implementing. One front door, one implementation.
ENGINE_COMMANDS = ("agent", "armada", "eval", "eval-compare", "loadtest", "usage",
                   "quota", "doctor", "check-config", "cron", "ttft", "providers")


def run_engine(argv):
    """Forward to `python -m loomweaver <argv>`. Returns its exit code."""
    try:
        from loomweaver import cli as _cli
    except Exception as e:
        print(f"engine unavailable: {_safe_message(e)}")
        return 1
    try:
        return int(_cli.main(list(argv)) or 0)
    except SystemExit as e:          # argparse exits are normal, not failures
        return int(e.code or 0)
    except Exception as e:
        print(f"engine command failed: {_safe_message(e)}")
        return 1


def print_profile():
    """What flippy has learned. Works with or without loomweaver installed."""
    try:
        from loomweaver import learning
        print(learning.render_text(learning.get_store().profile()))
    except Exception as e:
        print(f"profile unavailable: {_safe_message(e)}")


def health():
    deployments = build_deployments()
    providers = sorted({d["metadata"]["provider"] for d in deployments})
    keys = len({(d["metadata"]["provider"], d["metadata"]["key_index"])
                for d in deployments})
    models = sorted({d["model_name"] for d in deployments})
    print(f"providers ({len(providers)}): {', '.join(providers) or 'none'}")
    print(f"live keys: {keys}   models: {len(models)}   "
          f"litellm deployments: {len(deployments)}")
    for m in models:
        print(f"  - {m}")
    if not deployments:
        print("  (none — set at least one provider env var; "
              "`--all-providers` lists every option)")
    print("\nengine services:")
    for name, on in hub.services().items():
        print(f"  [{'on ' if on else 'off'}] {name}")


def main():
    ap = argparse.ArgumentParser(description="Unified AI hub: chat, vision, RAG, TTS, STT.")
    ap.add_argument("args", nargs="*")
    ap.add_argument("--health", action="store_true")
    ap.add_argument("--chat", help="simple chat: arg is the prompt")
    ap.add_argument("--simple", action="store_true",
                    help="route to cheapest model (with --chat)")
    ap.add_argument("--tooltest", action="store_true")
    ap.add_argument("--embed", help="embed text")
    ap.add_argument("--vision", help="path to image")
    ap.add_argument("--rag", choices=["add", "query", "chat"], help="RAG command")
    ap.add_argument("--rag-chat", help="retrieve-then-answer prompt")
    ap.add_argument("--summarize", help="summarize text")
    ap.add_argument("--tts", help="text-to-speech prompt")
    ap.add_argument("--stt", help="path to audio file for STT")
    ap.add_argument("--top-k", type=int, default=3)
    # provider + memory surface, so aihub is self-sufficient as the entry point
    ap.add_argument("--providers", action="store_true",
                    help="list the providers currently configured")
    ap.add_argument("--all-providers", action="store_true",
                    help="list every provider flippy can talk to and how to enable it")
    ap.add_argument("--profile", action="store_true",
                    help="show what flippy has learned about you")
    ap.add_argument("--learn", help="teach flippy a rule to apply to similar prompts")
    ap.add_argument("--forget", action="store_true", help="erase learned lessons")
    # engine passthrough: the agent, fleets, evals, benchmarks and dashboards
    # all live in loomweaver, so aihub forwards instead of duplicating them
    ap.add_argument("--agent", help="run the autonomous agent on a goal")
    ap.add_argument("--tools", help="comma-separated tool scope for --agent")
    ap.add_argument("--max-steps", type=int, default=0, help="step cap for --agent")
    ap.add_argument("--engine", choices=ENGINE_COMMANDS,
                    help="forward to the loomweaver engine (agent, armada, eval, "
                         "loadtest, usage, quota, doctor, cron, ttft, providers)")
    a = ap.parse_args()
    if a.health:
        health(); return
    if a.all_providers:
        from flippy_providers import describe_catalog
        print("Every provider flippy speaks (OpenAI chat/completions wire format).")
        print("Set the variable in column 3 to switch one on — any of them.\n")
        print(f"{'provider':16} {'cost':6} {'activate with':52} models override")
        print("-" * 104)
        for row in describe_catalog():
            print(f"{row['name']:16} {row['cost']:6} {row['activate_with']:52} "
                  f"{row['models_env'] or '-'}")
        print("\nPlus: any <PREFIX>_API_KEY + <PREFIX>_BASE_URL pair registers itself.")
        return
    if a.providers:
        health(); return
    if a.profile:
        print_profile(); return
    if a.learn:
        try:
            from loomweaver import learning
            lid = learning.get_store().add_lesson(a.learn)
            print(f"learned (lesson {lid}): {a.learn}")
        except Exception as e:
            print(f"could not store lesson: {_safe_message(e)}")
        return
    if a.agent:
        # the autonomous agent lives in loomweaver; aihub just hands it the goal
        sys.exit(run_engine(["agent", a.agent]
                            + (["--tools", a.tools] if a.tools else [])
                            + (["--max-steps", str(a.max_steps)] if a.max_steps else [])))
    if a.engine:
        sys.exit(run_engine([a.engine] + list(a.args)))
    if a.forget:
        try:
            from loomweaver import learning
            learning.get_store().forget()
            print("forgot every lesson")
        except Exception as e:
            print(f"could not forget: {_safe_message(e)}")
        return
    if a.tooltest:
        msg = tooltest()
        print(json.dumps(msg, indent=2)); return
    if a.embed:
        v = embed(a.embed)
        print(f"vector dim: {len(v[0]) if isinstance(v[0], list) else len(v)}")
        return
    if a.vision:
        prompt = " ".join(a.args) or "Describe this image."
        print(vision(a.vision, prompt)); return
    if a.rag == "add":
        text = " ".join(a.args)
        if not text:
            print("usage: aihub.py --rag add <text>"); return
        rid = rag_add(text)
        print("added:", rid); return
    if a.rag == "query":
        q = " ".join(a.args)
        hits = rag_query(q, top_k=a.top_k)
        if not hits:
            print("  (no matches)")
        for h in hits:
            print(f"  [{h['score']}] {h['text'][:120]}")
        return
    if a.rag_chat:
        msgs = [{"role": "user", "content": a.rag_chat}]
        print(smart_chat(msgs, use_rag=True, top_k=a.top_k)["content"]); return
    if a.summarize:
        print(summarize(a.summarize)); return
    if a.tts:
        print(tts(a.tts)); return
    if a.stt:
        print(stt(a.stt)); return
    if a.chat:
        msgs = [{"role": "user", "content": a.chat}]
        print(smart_chat(msgs, simple=a.simple)["content"]); return
    ap.print_help()


if __name__ == "__main__":
    main()

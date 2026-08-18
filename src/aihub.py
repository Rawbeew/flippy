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
import os, sys, json, time, argparse, base64, hashlib, math


# ---------- Provider registry ----------
def build_router_models():
    """Return list of (lit_name, friendly, api_key, api_base) tuples."""
    out = []
    if os.environ.get("FREEINFERENCE_KEY"):
        k = os.environ["FREEINFERENCE_KEY"]
        base = "https://freeinference.org/v1"
        for m in ["minimax-m3", "qwen3.6-35b", "deepseek-v4-flash", "glm-5.1"]:
            out.append((f"openai/{m}", m, k, base))
    if os.environ.get("GROQ_KEY"):
        k = os.environ["GROQ_KEY"]
        for m in ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]:
            out.append((f"groq/{m}", m, k, None))
    if os.environ.get("NVIDIA_KEY"):
        k = os.environ["NVIDIA_KEY"]
        base = "https://integrate.api.nvidia.com/v1"
        for m in ["nvidia/llama-3.3-nemotron-super-49b-v1"]:
            out.append((f"openai/{m}", m, k, base))
    if os.environ.get("CLOUDFLARE_TOKEN") and os.environ.get("CLOUDFLARE_ACCOUNT_ID"):
        k = os.environ["CLOUDFLARE_TOKEN"]
        cf_base = (f"https://api.cloudflare.com/client/v4/accounts/"
                   f"{os.environ['CLOUDFLARE_ACCOUNT_ID']}/ai")
        out.append(("openai/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                    "@cf/meta/llama-3.3-70b-instruct-fp8-fast", k, cf_base))
    return out


def build_router():
    """Build a litellm Router with the configured providers. Returns (router, litellm)."""
    import litellm
    model_list = []
    for lit_name, friendly, key, api_base in build_router_models():
        entry = {"model_name": friendly,
                 "litellm_params": {"model": lit_name, "api_key": key}}
        if api_base:
            entry["litellm_params"]["api_base"] = api_base
        model_list.append(entry)
    if not model_list:
        raise RuntimeError(
            "no providers configured — set FREEINFERENCE_KEY / GROQ_KEY / "
            "NVIDIA_KEY / CLOUDFLARE_TOKEN + CLOUDFLARE_ACCOUNT_ID")
    return litellm.Router(
        model_list=model_list,
        set_verbose=False,
        num_retries=3,
        allowed_fails=2,
        cooldown_time=60,
        enable_pre_call_checks=True,
    ), litellm


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
    resp = router.completion(model=model, messages=messages,
                             max_tokens=max_tokens, caching=cache)
    c = resp["choices"][0]["message"]["content"]
    usage = resp.get("usage") or {}
    cache_read = 0
    try:
        ptd = usage.get("prompt_tokens_details") or {}
        if hasattr(ptd, "get"):
            cache_read = ptd.get("cached_tokens", 0)
    except Exception:
        cache_read = 0
    return {"content": c, "model": resp.get("model"),
            "prompt_tokens": usage.get("prompt_tokens"),
            "completion_tokens": usage.get("completion_tokens"),
            "cache_read": cache_read}


def summarize(text, max_words=80, model="deepseek-v4-flash"):
    """Cheap summarization — useful for compressing long contexts."""
    router, _ = build_router()
    resp = router.completion(model=model,
        messages=[{"role": "user", "content":
            f"Compress the following into a concise summary of at most {max_words} words. "
            f"Keep key facts, numbers, names:\n\n{text[:8000]}"}])
    return resp["choices"][0]["message"]["content"].strip()


# ---------- Embeddings (bge-m3 via freeinference) ----------
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
        raise RuntimeError(f"embed HTTP {e.code}: {e.read().decode()[:200]}")


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
    outpath = outpath or os.path.join(os.path.expanduser("~"), "tts.mp3")
    try:
        import edge_tts, asyncio
        async def _run():
            c = edge_tts.Communicate(text, "en-US-JennyNeural")
            await c.save(outpath)
        asyncio.run(_run())
        return outpath
    except Exception as e:
        print(f"[aihub] edge-tts failed: {e}", file=sys.stderr)
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
        raise RuntimeError(f"stt HTTP {e.code}: {e.read().decode()[:200]}")


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
        raise RuntimeError(f"vision HTTP {e.code}: {e.read().decode()[:200]}")


# ---------- Tool / function-calling demo ----------
def tooltest():
    """Exercise function calling: the model decides to call get_weather."""
    import urllib.request, urllib.error
    key = os.environ.get("FREEINFERENCE_KEY")
    if not key:
        raise RuntimeError("FREEINFERENCE_KEY required")
    tools = [{"type": "function",
              "function": {"name": "get_weather",
                           "description": "Get current weather for a city",
                           "parameters": {"type": "object",
                                         "properties": {"city": {"type": "string"}},
                                         "required": ["city"]}}}]
    body = {"model": "minimax-m3",
            "messages": [{"role": "user", "content": "What's the weather in Lagos?"}],
            "tools": tools, "max_tokens": 256}
    req = urllib.request.Request("https://freeinference.org/v1/chat/completions",
        data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {key}",
                 "Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req, timeout=60) as r:
        d = json.loads(r.read().decode())
    return d["choices"][0]["message"]


# ---------- CLI ----------
def health():
    providers = build_router_models()
    print(f"providers ({len(providers)}):")
    for lit, friendly, _key, _base in providers:
        print(f"  - {friendly}  (litellm={lit})")
    if not providers:
        print("  (none — set at least one provider env var)")


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
    a = ap.parse_args()
    if a.health:
        health(); return
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
        for h in hits:
            print(f"  [{h['score']}] {h['text'][:120]}"); return
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

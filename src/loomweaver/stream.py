"""stream.py — streaming chat with TTFT (time-to-first-token) measurement.

Uses SSE streaming (`stream: true`) against OpenAI-compatible endpoints.
Falls back to non-streaming when a provider doesn't support it.
"""
import json
import time
import urllib.request

from .core import UA


def stream_chat(prov, messages, model=None, max_tokens=2000, timeout=120):
    """Stream a completion. Returns dict with ttft, total latency, text, tokens.

    Note: reasoning models (e.g. groq openai/gpt-oss-*) stream `reasoning` deltas
    before `content`; a too-small max_tokens gets consumed by reasoning and yields
    no visible content. Default raised to 2000 to stay safe.
    """
    if prov.get("single"):
        return {"ok": False, "error": "provider does not support streaming", "latency": 0.0}
    h = {"Authorization": f"Bearer {prov['key']}", "Content-Type": "application/json",
         "User-Agent": UA, "Accept": "text/event-stream"}
    body = {"model": model or prov["models"][0], "messages": messages,
            "max_tokens": max_tokens, "stream": True}
    t0 = time.time()
    ttft = None
    chunks = []
    req = urllib.request.Request(prov["url"], data=json.dumps(body).encode(),
                                 headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            for raw in r:
                line = raw.decode("utf-8", "ignore").strip()
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                try:
                    d = json.loads(payload)
                    delta = d["choices"][0].get("delta", {})
                    content = delta.get("content") or ""
                except Exception:
                    continue
                if content:
                    if ttft is None:
                        ttft = time.time() - t0  # first *visible* token
                    chunks.append(content)
    except Exception as e:
        return {"ok": False, "error": str(e), "latency": time.time() - t0}

    total = time.time() - t0
    text = "".join(chunks)
    if ttft is None:
        return {"ok": False, "error": "no streamed content received", "latency": total}
    n_words = len(text.split())
    gen_time = max(total - ttft, 0.001)
    return {"ok": True, "ttft": round(ttft, 3), "latency": round(total, 3),
            "text": text, "words": n_words,
            "tps": round(n_words / gen_time, 1) if gen_time > 0 else 0}

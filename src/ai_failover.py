#!/usr/bin/env python3
"""
ai_failover.py — public, portable version.

Multi-provider inference router with automatic failover across free OpenAI-
compatible chat APIs. Free providers are tried first; on 429/5xx/network
errors, the next provider is used. Designed for code that should never hit
a rate limit wall.

Credentials: read from environment variables (NEVER hardcoded). Set one or more:

    export FREEINFERENCE_KEY=...     # https://freeinference.org  (free)
    export GROQ_KEY=...              # https://api.groq.com       (free tier)
    export NVIDIA_KEY=...            # https://integrate.api.nvidia.com (free tier)
    # Hyperbolic is left out by default (paid). Uncomment to enable.

Usage:
    python ai_failover.py "your prompt"
    python ai_failover.py --model minimax-m3 "your prompt"
    python ai_failover.py --health      # ping all configured providers
    python ai_failover.py --list        # list configured providers/models

License: MIT
"""
import os, sys, json, time, argparse, urllib.request, urllib.error


# ---------- Provider registry (delegates to canonical providers.py) ----------
import sys as _sys, os as _os
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from flippy_providers import get_providers  # canonical registry


def build_providers():
    """Backwards-compatible wrapper — the canonical list lives in providers.py."""
    return [
        {k: v for k, v in p.items() if k in ("name", "url", "key", "models",
                                              "cost", "primary", "single")}
        for p in get_providers()
    ]


def post(url, headers, body, timeout=60):
    headers = dict(headers)
    headers.setdefault("User-Agent",
                       "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                       "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")
    req = urllib.request.Request(url, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        try:
            detail = json.loads(e.read().decode())
        except Exception:
            detail = str(e)
        return e.code, detail
    except urllib.error.URLError as e:
        return None, str(e)


def is_rate_limit(status, body):
    if status == 429:
        return True
    if status and status >= 500:
        return True
    s = json.dumps(body).lower()
    return any(k in s for k in ("rate limit", "rate_limit", "too many requests",
                                "429", "quota", "try again later", "temporarily"))


def call(prov):
    h = {"Authorization": f"Bearer {prov['key']}", "Content-Type": "application/json"}
    if prov.get("single"):
        body = {"messages": [{"role": "user", "content": prov["__prompt"]}]}
    else:
        body = {"model": prov["__model"],
                "messages": [{"role": "user", "content": prov["__prompt"]}],
                "max_tokens": prov["__max_tokens"]}
    return post(prov["url"], h, body)


def extract_text(prov, data):
    if prov.get("single"):
        try:
            return data["result"]["response"]
        except Exception:
            return data
    try:
        return data["choices"][0]["message"]["content"]
    except Exception:
        return data


def infer(prompt, model=None, max_tokens=1024):
    providers = build_providers()
    providers.sort(key=lambda p: (0 if p["cost"] == "free" else 1, p["name"]))
    last_error = "no providers configured"
    for prov in providers:
        prov["__prompt"] = prompt
        prov["__max_tokens"] = max_tokens
        prov["__model"] = model if model else prov["models"][0]
        print(f"[ai_failover] trying {prov['name']} ({prov['__model']})...",
              file=sys.stderr)
        status, data = call(prov)
        if status == 200:
            text = extract_text(prov, data)
            if isinstance(text, str) and text.strip():
                return {"provider": prov["name"], "model": prov["__model"],
                        "content": text, "status": 200}
            last_error = f"{prov['name']}: {'empty content' if isinstance(text, str) else 'unexpected payload'}"
        else:
            rl = "RATE-LIMITED" if is_rate_limit(status, data) else ""
            print(f"[ai_failover] {prov['name']} failed ({status}) {rl} "
                  f"{json.dumps(data)[:160]}", file=sys.stderr)
            last_error = f"{prov['name']}: HTTP {status} {rl}".strip()
        time.sleep(0.4)
    return {"provider": None, "content": None, "error": last_error, "status": 0}


def health():
    providers = build_providers()
    print(f"{'provider':<14} {'cost':<6} {'status':<10} models")
    for p in providers:
        prov = dict(p); prov["__prompt"] = "Reply with exactly: OK"
        prov["__max_tokens"] = 16
        prov["__model"] = prov["models"][0]
        status, data = call(prov)
        ok = "OK" if status == 200 else f"FAIL({status})"
        print(f"{p['name']:<14} {p['cost']:<6} {ok:<10} {p['models']}")
    if not providers:
        print("(no providers configured — set FREEINFERENCE_KEY / GROQ_KEY / NVIDIA_KEY)")


def main():
    ap = argparse.ArgumentParser(description="Free-first multi-provider failover router.")
    ap.add_argument("prompt", nargs="*")
    ap.add_argument("--model", help="override default model")
    ap.add_argument("--max-tokens", type=int, default=1024)
    ap.add_argument("--json", action="store_true", help="JSON output")
    ap.add_argument("--health", action="store_true", help="ping providers")
    ap.add_argument("--list", action="store_true", help="list configured providers")
    a = ap.parse_args()
    if a.health:
        health(); return
    if a.list:
        for p in build_providers():
            print(f"{p['name']}: cost={p['cost']} models={p['models']}")
        return
    prompt = " ".join(a.prompt) or "(empty prompt)"
    res = infer(prompt, a.model, a.max_tokens)
    if a.json:
        print(json.dumps(res))
    else:
        if res["content"]:
            print(res["content"])
        else:
            print("FAILED:", res.get("error"), file=sys.stderr)
            sys.exit(1)


if __name__ == "__main__":
    main()

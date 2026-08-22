"""core.py — provider registry, failover router, event log, session store."""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

from loomweaver import quota_ledger as _ql

CREDS_PATH = os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local", "hermes", "secrets", "credentials.env")
UA = "OpenAI File Downloader, XaiImageApiFetch/1.0"


# ---------------------------------------------------------------- credentials

def load_creds(path=CREDS_PATH):
    d = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                d[k] = v.strip()
    return d


# ---------------------------------------------------------------- providers

def build_providers(creds=None):
    """Canonical list comes from providers.py; creds dict (from a credentials
    file) is mapped onto env-var names for compatibility."""
    try:
        import flippy_providers as _reg
    except ImportError:  # running as a package: add src/ to path and retry
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
        import flippy_providers as _reg
    if creds:
        env = {
            "OPENROUTER_KEY": creds.get("OPENROUTER_KEY"),
            "FREEINFERENCE_KEY": creds.get("FREEINFERENCE_KEY"),
            "CLOUDFLARE_TOKEN": creds.get("CLOUDFLARE_TOKEN"),
            "CLOUDFLARE_ACCOUNT_ID": creds.get("CLOUDFLARE_ACCOUNT_ID"),
            "NVIDIA_KEY": creds.get("NVIDIA_KEY"),
            "GROQ_KEY": creds.get("GROQ_KEY"),
        }
        env = {k: v for k, v in env.items() if v}
        return _reg.get_providers(env)
    return _reg.get_providers()


def is_retryable(status, body):
    if status == 429 or (status and status >= 500):
        return True
    s = json.dumps(body).lower() if body else ""
    return any(k in s for k in ("rate limit", "too many requests", "quota", "try again later"))


def chat(prov, messages, model=None, max_tokens=1024, timeout=120):
    """One chat call to one provider. Returns dict with text/usage/latency."""
    h = {"Authorization": f"Bearer {prov['key']}", "Content-Type": "application/json",
         "User-Agent": UA}
    if prov.get("single"):
        body = {"messages": messages}
    else:
        body = {"model": model or prov["models"][0], "messages": messages,
                "max_tokens": max_tokens}
    t0 = time.time()
    req = urllib.request.Request(prov["url"], data=json.dumps(body).encode(),
                                 headers=h, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            data = json.loads(r.read().decode())
            status = r.status
    except urllib.error.HTTPError as e:
        try:
            data = json.loads(e.read().decode())
        except Exception:
            data = {"raw": str(e)}
        status = e.code
    except Exception as e:
        return {"ok": False, "error": str(e), "latency": time.time() - t0}

    latency = time.time() - t0
    if status != 200:
        return {"ok": False, "status": status, "retryable": is_retryable(status, data),
                "error": json.dumps(data)[:300], "latency": latency}
    if prov.get("single"):
        text = data.get("result", {}).get("response", "")
    else:
        text = data.get("choices", [{}])[0].get("message", {}).get("content", "")
    usage = data.get("usage", {}) or {}
    return {"ok": True, "text": text, "usage": usage, "latency": latency}


def route(messages, model=None, max_tokens=1024, creds=None, on_event=None):
    """Failover across all providers; returns first ok result + which provider won.

    Model pinning:
      - 'provider/model' → only that provider is tried, with 'model' stripped of the prefix
      - bare model name  → only providers whose models list contains it are tried
      - None             → each provider's default model
    """
    last = None
    ledger = _ql.get_ledger()
    for prov in build_providers(creds or load_creds()):
        m = None
        if model:
            if any(model == x for x in prov["models"]):
                m = model  # exact model ID match (even with slashes, e.g. openai/gpt-oss-20b)
            elif "/" in model and model.split("/", 1)[0] == prov["name"]:
                m = model.split("/", 1)[1]  # provider-pinned: strip the prefix
            else:
                continue  # pinned to a different provider/model — skip this one
        allowed, reason = ledger.check_quota(prov["name"])
        if not allowed:
            if on_event:
                on_event({"type": "quota_skip", "provider": prov["name"], "reason": reason})
            continue  # route away before hitting the 429
        ledger.record_request(prov["name"])
        r = chat(prov, messages, model=m, max_tokens=max_tokens)
        ledger.record_result(prov["name"], r.get("status") if not r.get("ok") else 200)
        if on_event:
            on_event({"type": "llm_call", "provider": prov["name"], "model": m,
                      "ok": r.get("ok"), "latency": round(r.get("latency", 0), 3)})
        if r.get("ok"):
            r["provider"] = prov["name"]
            r["model"] = m or ""
            return r
        last = r
    return {"ok": False, "error": (last or {}).get("error", "all providers failed")}


def route_hedged(messages, model=None, max_tokens=1024, creds=None, delay_ms=250,
                 on_event=None):
    """Hedged-request router: fire the top 2 eligible providers concurrently
    (the second after `delay_ms`), take the first success, abandon the loser.

    Tail-latency pattern: a slow first response no longer blocks when a second
    provider answers faster. Falls back to sequential route() when fewer than
    two providers match. Threading-based (no asyncio), stdlib only.

    Note: on a slow tail both providers complete, so hedging can burn 2x quota.
    """
    provs = []
    for prov in build_providers(creds or load_creds()):
        m = None
        if model:
            if any(model == x for x in prov["models"]):
                m = model
            elif "/" in model and model.split("/", 1)[0] == prov["name"]:
                m = model.split("/", 1)[1]
            else:
                continue
        provs.append((prov, m))
        if len(provs) == 2:
            break

    if len(provs) < 2:
        return route(messages, model=model, max_tokens=max_tokens, creds=creds,
                     on_event=on_event)

    result = {}
    done = threading.Event()

    def _attempt(prov, m, idx):
        r = chat(prov, messages, model=m, max_tokens=max_tokens)
        if on_event:
            try:
                on_event({"type": "hedged_call", "provider": prov["name"], "model": m,
                          "ok": r.get("ok"), "latency": round(r.get("latency", 0), 3),
                          "attempt": idx})
            except Exception:
                pass
        if r.get("ok") and not done.is_set():
            r["provider"] = prov["name"]
            r["model"] = m or ""
            r["hedged"] = {"attempt": idx}
            result["win"] = r
            done.set()
        elif not r.get("ok"):
            result.setdefault("last_err", r)

    t1 = threading.Thread(target=_attempt, args=provs[0] + (1,), daemon=True)
    t1.start()
    time.sleep(delay_ms / 1000.0)
    t2 = threading.Thread(target=_attempt, args=provs[1] + (2,), daemon=True)
    t2.start()
    # Wait for a winner; if none, wait for both to finish to collect the error.
    while not done.is_set() and (t1.is_alive() or t2.is_alive()):
        done.wait(0.05)
    if done.is_set():
        return result["win"]
    last = result.get("last_err") or {}
    return {"ok": False, "error": last.get("error", "all providers failed")}


# ---------------------------------------------------------------- run events

class RunLog:
    """Append-only JSONL event log per run."""

    def __init__(self, runs_dir=None):
        # default: <repo-root>/runs/ — two levels up from this file (src/loomweaver/core.py)
        self.dir = runs_dir or os.path.join(os.path.dirname(__file__), "..", "..", "runs",
                                            time.strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "events.jsonl")
        self._lock = threading.Lock()

    def emit(self, event):
        event = {"t": round(time.time(), 3), **event}
        with self._lock:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event, ensure_ascii=False) + "\n")
        return event

    def read(self):
        out = []
        if os.path.exists(self.path):
            for line in open(self.path, encoding="utf-8"):
                out.append(json.loads(line))
        return out


# ---------------------------------------------------------------- sessions

class SessionStore:
    """Persistent agent sessions: memory.json-style episode store."""

    def __init__(self, root=None):
        self.root = root or os.path.join(os.path.dirname(__file__), "..", "..", "sessions")
        os.makedirs(self.root, exist_ok=True)

    def _path(self, sid):
        return os.path.join(self.root, f"{sid}.json")

    def load(self, sid):
        p = self._path(sid)
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))
        return {"id": sid, "messages": [], "facts": {}, "created": time.time()}

    def save(self, sess):
        with open(self._path(sess["id"]), "w", encoding="utf-8") as f:
            json.dump(sess, f, indent=2, ensure_ascii=False)

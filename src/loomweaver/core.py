"""core.py — provider registry, failover router, event log, session store."""
import json
import os
import re
import threading
import time
import urllib.error
import urllib.request

CREDS_PATH = os.path.join(os.environ.get("USERPROFILE", ""), "AppData", "Local", "hermes", "secrets", "credentials.env")
UA = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
      "(KHTML, like Gecko) Chrome/126.0 Safari/537.36")


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

def build_providers(creds):
    """Return list of OpenAI-compatible providers, free-first."""
    P = []
    if creds.get("OPENROUTER_KEY"):
        P.append({"name": "openrouter", "cost": "free", "primary": True,
                  "url": "https://openrouter.ai/api/v1/chat/completions",
                  "key": creds["OPENROUTER_KEY"],
                  "models": ["stealth/ox-alpha"]})
    if creds.get("FREEINFERENCE_KEY"):
        P.append({"name": "freeinference", "cost": "free",
                  "url": "https://freeinference.org/v1/chat/completions",
                  "key": creds["FREEINFERENCE_KEY"],
                  "models": ["minimax-m3", "qwen3.6-35b", "deepseek-v4-flash"]})
    if creds.get("CLOUDFLARE_TOKEN") and creds.get("CLOUDFLARE_ACCOUNT_ID"):
        acc = creds["CLOUDFLARE_ACCOUNT_ID"]
        P.append({"name": "cloudflare", "cost": "free", "single": True,
                  "url": f"https://api.cloudflare.com/client/v4/accounts/{acc}/ai/run/@cf/meta/llama-3.3-70b-instruct-fp8-fast",
                  "key": creds["CLOUDFLARE_TOKEN"],
                  "models": ["@cf/meta/llama-3.3-70b-instruct-fp8-fast"]})
    if creds.get("NVIDIA_KEY"):
        P.append({"name": "nvidia", "cost": "free",
                  "url": "https://integrate.api.nvidia.com/v1/chat/completions",
                  "key": creds["NVIDIA_KEY"],
                  "models": ["nvidia/llama-3.3-nemotron-super-49b-v1"]})
    if creds.get("GROQ_KEY"):
        P.append({"name": "groq", "cost": "free",
                  "url": "https://api.groq.com/openai/v1/chat/completions",
                  "key": creds["GROQ_KEY"],
                  "models": ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]})
    return sorted(P, key=lambda p: (0 if p.get("primary") else 1, p["name"]))


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
    """Failover across all providers; returns first ok result + which provider won."""
    last = None
    for prov in build_providers(creds or load_creds()):
        m = model if (model and not model.startswith(prov["name"] + "/")) else None
        # allow 'provider/model' syntax
        if model and "/" in model and model.split("/")[0] == prov["name"]:
            m = model.split("/", 1)[1]
        elif model and any(model == x for x in prov["models"]):
            m = model
        r = chat(prov, messages, model=m, max_tokens=max_tokens)
        if on_event:
            on_event({"type": "llm_call", "provider": prov["name"], "model": m,
                      "ok": r.get("ok"), "latency": round(r.get("latency", 0), 3)})
        if r.get("ok"):
            r["provider"] = prov["name"]
            r["model"] = m or (prov["models"][0] if not prov.get("single") else "")
            return r
        last = r
        if not r.get("retryable"):
            break  # auth error etc — next provider likely same key class? no: try next anyway
    return {"ok": False, "error": (last or {}).get("error", "all providers failed")}


# ---------------------------------------------------------------- run events

class RunLog:
    """Append-only JSONL event log per run."""

    def __init__(self, runs_dir=None):
        self.dir = runs_dir or os.path.join(os.path.dirname(__file__), "..", "runs",
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
        self.root = root or os.path.join(os.path.dirname(__file__), "..", "sessions")
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

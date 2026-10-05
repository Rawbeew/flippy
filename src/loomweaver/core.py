"""core.py — provider registry, failover router, event log, session store."""
from __future__ import annotations
import json
import os
import threading
import re
import time
import urllib.error
import urllib.request

try:
    from . import quota_ledger as _ql
    from . import semantic_cache as _sc
    from . import usage as _usage
    from . import router_policy as _rp
except ImportError:  # running as a top-level package: add src/ to path and retry
    import sys as _sys
    _sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
    from loomweaver import quota_ledger as _ql
    from loomweaver import semantic_cache as _sc
    from loomweaver import usage as _usage
    from loomweaver import router_policy as _rp

try:  # key rotation is additive — other armada agents' modules must not break us
    from . import key_rotation as _kr
except ImportError:
    try:
        from loomweaver import key_rotation as _kr
    except ImportError:
        _kr = None

CREDS_PATH = os.environ.get(
    "FLIPPY_CREDS_PATH",
    os.path.join(
        os.environ.get("USERPROFILE", os.environ.get("HOME", "")),
        ".flippy", "credentials.env"
    )
)
UA = "flippy/0.1.0 (github.com/Rawbeew/flippy)"
import re as _re

# Defense-in-depth: never let a raw provider error body reach a caller (log or
# HTTP client) containing something that looks like a credential. Lightweight
# scrub of common key shapes; tools.redact owns the full family list — this is
# the boundary that must not echo secrets even if a provider embeds one in an
# error message.
_ERR_RE = _re.compile(
    r"(?i)(sk-|gsk_|nvapi-|cfut_|ghp_|gho_|ghu_|ghs_|ghr_|github_pat_|"
    r"xox[baprs]-|rk_live_|rk_test_|sk_live_|sk_test_|AKIA[0-9A-Z]{16}|"
    r"ASIA[0-9A-Z]{16}|eyJ[A-Za-z0-9_-]{8,}|hf_[A-Za-z0-9]{20,}|"
    r"xai-[A-Za-z0-9]{20,})[A-Za-z0-9_\-]{6,}")
def scrub_error(body: str) -> str:
    """Redact credential-shaped tokens from a provider error body string."""
    return _ERR_RE.sub("[REDACTED]", body) if body else body



# ---------------------------------------------------------------- credentials

def load_creds(path: str = CREDS_PATH) -> dict[str, str]:
    d = {}
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                d[k] = v.strip()
    return d


# ---------------------------------------------------------------- providers

def build_providers(creds: dict[str, str] | None = None) -> list[dict]:
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


def is_retryable(status: int | None, body: dict | None) -> bool:
    if status == 429 or (status and status >= 500):
        return True
    s = json.dumps(body).lower() if body else ""
    return any(k in s for k in ("rate limit", "too many requests", "quota", "try again later"))


def chat(prov: dict, messages: list[dict], model: str | None = None,
         max_tokens: int = 1024, timeout: int = 120,
         tools: list[dict] | None = None,
         tool_choice: str | None = "auto") -> dict:
    """One chat call to one provider. Returns dict with text/usage/latency
    and, when `tools` is provided and the provider answers with native
    tool_calls, a `tool_calls` list of {name, arguments(dict)} entries."""
    h = {"Authorization": f"Bearer {prov['key']}", "Content-Type": "application/json",
         "User-Agent": UA}
    if prov.get("single"):
        body = {"messages": messages}
    else:
        body = {"model": model or prov["models"][0], "messages": messages,
                "max_tokens": max_tokens}
    if tools:
        body["tools"] = tools
        if tool_choice:
            body["tool_choice"] = tool_choice
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
        # DNS / connection / timeout (non-HTTP) failures are transient: mark
        # them retryable so route() gives the provider another chance (and can
        # fail over) instead of treating them as terminal. HTTP 4xx / junk-200
        # cases stay as-is above.
        return {"ok": False, "retryable": True, "error": str(e),
                "latency": time.time() - t0}

    latency = time.time() - t0
    if status != 200:
        return {"ok": False, "status": status, "retryable": is_retryable(status, data),
                "error": scrub_error(json.dumps(data))[:300], "latency": latency}
    if prov.get("single"):
        text = data.get("result", {}).get("response", "")
    else:
        try:
            choices = data.get("choices") or []
            if not choices:
                # 200 OK but empty response — treat as provider failure
                return {"ok": False,
                        "status": 200,
                        "retryable": True,
                        "error": f"200 with empty choices: {json.dumps(data)[:200]}",
                        "latency": latency}
            message = choices[0].get("message") or {}
            text = (message.get("content") or "").strip()
            # Native tool call(s): the model chose a tool rather than replying
            # with text. Return them so the agent can execute + observe.
            if tools and message.get("tool_calls"):
                calls = []
                for tc in message["tool_calls"]:
                    fn = tc.get("function") or {}
                    try:
                        args = json.loads(fn.get("arguments") or "{}")
                    except Exception:
                        args = {}
                    calls.append({"name": fn.get("name", ""), "arguments": args,
                                  "id": tc.get("id")})
                return {"ok": True, "status": 200, "text": "", "tool_calls": calls,
                        "usage": data.get("usage"), "provider": prov.get("name"),
                        "model": data.get("model") or body.get("model"),
                        "latency": latency}
            if not text:
                return {"ok": False,
                        "status": 200,
                        "retryable": True,
                        "error": f"200 with empty content: {json.dumps(data)[:200]}",
                        "latency": latency}
        except (KeyError, IndexError, TypeError) as e:
            return {"ok": False,
                    "status": 200,
                    "retryable": True,
                    "error": f"malformed 200 body: {type(e).__name__}: {json.dumps(data)[:200]}",
                    "latency": latency}
    usage = data.get("usage", {}) or {}
    return {"ok": True, "text": text, "usage": usage, "latency": latency}


def chat_with_rotation(prov: dict, messages: list[dict], model: str | None = None,
                       max_tokens: int = 1024, timeout: int = 120,
                       on_event: callable | None = None,
                       tools: list[dict] | None = None,
                       tool_choice: str | None = "auto") -> dict:
    """chat() with multi-key rotation.

    Walks the provider's live keys (key_rotation state): on 401/403 the key is
    marked permanently dead and the SAME provider is retried with the next key;
    on 429 the key is marked exhausted (cooldown) and we move to the next
    key/provider. All keys dead -> provider unavailable (returns last error).
    """
    keys = prov.get("keys") or [prov.get("key")]
    state = _kr.get_state() if _kr else None
    pairs = state.live_pairs(prov["name"], keys) if state else list(enumerate(keys))
    if not pairs:
        if on_event:
            on_event({"type": "keys_exhausted", "provider": prov["name"]})
        return {"ok": False, "status": None, "retryable": False,
                "error": f"all {len(keys)} key(s) dead/exhausted for {prov['name']}"}
    last = None
    for key, idx in pairs:
        p2 = dict(prov)
        p2["key"] = key
        kw = dict(tools=tools, tool_choice=tool_choice) if tools else {}
        r = chat(p2, messages, model=model, max_tokens=max_tokens, timeout=timeout, **kw)
        if r.get("ok"):
            if state:
                state.advance(prov["name"], idx)
            return r
        status = r.get("status")
        reason = f"key {_kr.mask(key) if _kr else idx}: {str(r.get('error'))[:120]}"
        if status in (401, 403):
            # revoked key — permanent skip, retry same provider with next key
            if state:
                state.mark_dead(prov["name"], idx, reason=reason)
            if on_event:
                on_event({"type": "key_dead", "provider": prov["name"],
                          "key_index": idx, "status": status})
            last = r
            continue
        if status == 429 and state:
            state.mark_exhausted(prov["name"], idx,
                                 retry_after=_kr.parse_retry_after(r))
            if on_event:
                on_event({"type": "key_exhausted", "provider": prov["name"],
                          "key_index": idx})
        last = r
        break  # non-auth failure: existing provider-level failover handles it
    return last or {"ok": False, "error": "no key attempted"}


def route(messages: list[dict], model: str | None = None,
          max_tokens: int = 1024, creds: dict[str, str] | None = None,
          on_event: callable | None = None, timeout_budget_s: float | None = None,
          max_provider_attempts: int = 3,
          tools: list[dict] | None = None,
          tool_choice: str | None = "auto") -> dict:
    """Failover across all providers; returns first ok result + which provider won.

    Model pinning:
      - 'provider/model' → only that provider is tried, with 'model' stripped of the prefix
      - bare model name  → only providers whose models list contains it are tried
      - None             → each provider's default model

    Production routing (router_policy):
      - adaptive provider order: EWMA success/latency score, cooldown hysteresis
      - in-provider retry with exponential backoff + jitter for retryable errors
      - optional deadline (timeout_budget_s): stop trying more providers when
        the budget is spent and return the best error we have
      - every provider attempt is emitted as an event (attempt number, status)
    """
    deadline = (time.time() + timeout_budget_s) if timeout_budget_s else None
    retry = _rp.RetryPolicy(max_attempts=max_provider_attempts)
    policy = _rp.get_policy()
    last = None
    ledger = _ql.get_ledger()
    # --- semantic response cache: exact hash fast path, similarity fallback ---
    # Skip when native tools are requested: a cached response has no tool_calls,
    # so serving one would dead-end a tool-calling loop (and could return stale,
    # unrelated text). Tool-calling prompts are stateful by nature.
    if _sc.cache_enabled() and not tools and not _sc.is_stateful(messages):
        try:
            hit = _sc.get_cache().lookup(messages)
            if hit:
                if on_event:
                    on_event({"type": "cache_hit", "similarity": hit["similarity"],
                              "exact": hit["exact"]})
                _usage.record("cache", model or "", True, 0.0, cached=True)
                return {"ok": True, "text": hit["response"], "usage": {}, "latency": 0.0,
                        "provider": "cache", "model": model or "", "cached": True,
                        "similarity": hit["similarity"]}
        except Exception:
            pass  # cache must never break routing
    providers = build_providers(creds or load_creds())
    for prov in policy.order(providers):
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
            # propagate ledger cooldowns into the adaptive policy
            try:
                policy.note_dead(prov["name"], _rp.cooldown_seconds_from_reason(reason))
            except Exception:
                pass
            continue  # route away before hitting the 429
        ledger.record_request(prov["name"])
        attempt = 0
        while True:
            attempt += 1
            kw = dict(tools=tools, tool_choice=tool_choice) if tools else {}
            r = chat_with_rotation(prov, messages, model=m, max_tokens=max_tokens, **kw,
                                   on_event=on_event)
            ledger.record_result(prov["name"], r.get("status") if not r.get("ok") else 200)
            policy.note_result(prov["name"], bool(r.get("ok")),
                                           r.get("latency", 0.0))
            if on_event:
                on_event({"type": "llm_call", "provider": prov["name"], "model": m,
                          "ok": r.get("ok"), "latency": round(r.get("latency", 0), 3),
                          "attempt": attempt,
                          "status": 200 if r.get("ok") else r.get("status")})
            if r.get("ok"):
                r["provider"] = prov["name"]
                r["model"] = m or ""
                r["attempts"] = attempt
                _usage.record(prov["name"], m or "", True, r.get("latency", 0),
                              usage=r.get("usage") or {})
                if _sc.cache_enabled() and not _sc.is_stateful(messages):
                    try:
                        _sc.get_cache().store(messages, r.get("text", ""), model_tag=m or "")
                    except Exception:
                        pass
                return r
            _usage.record(prov["name"], m or "", False, r.get("latency", 0))
            last = r
            # in-provider retry for transient failures (429 bursts, 5xx, timeouts)
            retry_after = None
            try:
                retry_after = _kr.parse_retry_after(r) if _kr else None
            except Exception:
                retry_after = None
            if not retry.should_retry(attempt, bool(r.get("retryable")), deadline,
                                      status=r.get("status")):
                break
            backoff = retry.backoff_s(attempt, retry_after)
            if on_event:
                on_event({"type": "retry_wait", "provider": prov["name"],
                          "attempt": attempt, "backoff_s": round(backoff, 2)})
            # honor the deadline even while backing off
            if deadline is not None and time.time() + backoff >= deadline:
                break
            time.sleep(backoff)
        if deadline is not None and time.time() >= deadline:
            break  # budget spent — stop walking providers
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

    _SID_SAFE = re.compile(r"^[A-Za-z0-9._-]{1,120}$")

    def _path(self, sid):
        return os.path.join(self.root, f"{sid}.json")

    @staticmethod
    def _validate_sid(sid):
        """Zero-trust: a session id is only ever a safe filename token. Refuse any
        id that could traverse out of the sessions dir (.., slashes, drive letters)
        — a future untrusted controller (HTTP param, tool arg) must never turn
        session_id into a file read/write primitive."""
        if not sid or not isinstance(sid, str) or not SessionStore._SID_SAFE.match(sid):
            raise ValueError(f"invalid session id: {sid!r}")
        return sid

    def load(self, sid):
        sid = self._validate_sid(sid)
        p = self._path(sid)
        if os.path.exists(p):
            return json.load(open(p, encoding="utf-8"))
        return {"id": sid, "messages": [], "facts": {}, "created": time.time()}

    def save(self, sess):
        sid = self._validate_sid(sess["id"])
        with open(self._path(sid), "w", encoding="utf-8") as f:
            json.dump(sess, f, indent=2, ensure_ascii=False)
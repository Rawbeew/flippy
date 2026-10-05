"""Hardened tool registry for Loomweaver — every model-chosen action is guarded."""
import contextlib
import json
import os
import re
import subprocess
import threading
import urllib.request

from . import security

TOOLS = {}

# read-only tools need no guard beyond path/URL checks
SAFE_MODE = os.environ.get("LOOMWEAVER_SAFE_MODE") == "1"  # disables shell entirely


def tool(name, desc, params):
    def deco(fn):
        TOOLS[name] = {"desc": desc, "params": params, "fn": fn}
        return fn
    return deco


@tool("http_get", "Fetch a public web URL and return text (internal/private addresses blocked)",
      {"url": "str", "max_chars": "int=2000"})
def http_get(url, max_chars=2000):
    ok, reason = security.check_url(url)
    if not ok:
        return f"blocked: {reason}"
    req = urllib.request.Request(url, headers={"User-Agent": "flippy/0.1.0"})
    # guarded_urlopen re-runs check_url on every redirect hop: a public URL that
    # 302s to a private/metadata address must not become an SSRF bypass.
    with security.guarded_urlopen(req, timeout=30) as r:
        return redact(r.read().decode("utf-8", "ignore")[:max_chars])


KEY_REDACT_RE = re.compile(
    r"(?i)(sk-[a-z0-9_-]{10,}|gsk_[a-z0-9]{20,}|nvapi-[a-z0-9_-]{10,}|"
    r"cfut_[a-z0-9_-]{10,}|ghp_[A-Za-z0-9]{20,}|hf_[A-Za-z0-9]{20,}|"
    r"AKIA[A-Z0-9]{16}|ASIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{20,}|xai-[a-z0-9]{20,}|"
    r"gho_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|ghu_[A-Za-z0-9]{20,}|ghs_[A-Za-z0-9]{20,}|"
    r"ghr_[A-Za-z0-9]{20,}|"
    r"xox[bpasr]-[A-Za-z0-9-]{20,}|"
    # freeinference.org token. The separator is not documented publicly, so
    # both `-` and `_` are accepted; over-matching a token-like string here
    # costs nothing, under-matching leaks a credential into a model context.
    r"fi[-_][A-Za-z0-9_-]{20,}|"
    r"sk_live_[A-Za-z0-9]{20,}|sk_test_[A-Za-z0-9]{20,}|"
    r"rk_live_[A-Za-z0-9]{20,}|rk_test_[A-Za-z0-9]{20,}|"
    r"\b(?:[a-z0-9_]*(?:api[_-]?key|apikey|secret|token|passwd|password)"
    r"|access[_-]?key(?:[_-]?id)?)[a-z0-9_]*\b\s*[:=]\s*\S{8,}|"
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----|"
    r"\btype\b\s*:\s*\"\bservice_account\b\"[\s\S]{0,500}?private_key\b\s*:\s*\"|"
    r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,})"
)




def redact(text: str) -> str:
    """Strip anything that looks like a key before it reaches the model."""
    if not text:
        return text
    return KEY_REDACT_RE.sub("[REDACTED]", text)


@tool("read_file", "Read a file inside the project (credential paths denied; "
      "key-like tokens redacted from output)",
      {"path": "str", "max_chars": "int=4000"})
def read_file(path, max_chars=4000):
    ok, reason = security.check_path(path)
    if not ok:
        return f"blocked: {reason}"
    # Managed placeholder files are handled by the config layer.
    from . import observability, sentinel
    if observability.lookup_managed_file(path):
        return observability.render_payload("managed_file_read:" + path)
    if os.path.basename(path) == sentinel.BREADCRUMB_NAME:
        sentinel.fingerprint("breadcrumb_read", path=path)
        return sentinel.breadcrumb_source()
    with open(path, encoding="utf-8", errors="ignore") as f:
        out = redact(f.read(max_chars))
    # Anything of ours leaving through a tool result is swapped for a dead
    # substitute, so a harvested copy is worthless and the attempt is logged.
    out, _hit = sentinel.scrub_outbound(out)
    return out


@tool("write_file", "Write a file inside the project (guard code, cron jobs, CI, "
      "state dirs are read-only; credential paths denied)",
      {"path": "str", "content": "str"})
def write_file(path, content):
    ok, reason = security.check_write_path(path)
    if not ok:
        return f"blocked: {reason}"
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"wrote {len(content)} chars to {path}"


@tool("list_dir", "List a directory inside the project", {"path": "str"})
def list_dir(path="."):
    ok, reason = security.check_path(os.path.realpath(path))
    if not ok:
        return f"blocked: {reason}"
    return "\n".join(sorted(os.listdir(path)))


# Read-only shell context. Thread-local so concurrent agent roles in an armada
# each get their own setting, and deliberately NOT a tool argument so the model
# cannot talk its way out of it.
_shell_tls = threading.local()


def shell_readonly_active():
    return bool(getattr(_shell_tls, "readonly", False))


@contextlib.contextmanager
def shell_readonly(flag=True):
    """Run the enclosed dispatches with the shell held to the read-only tier."""
    prev = getattr(_shell_tls, "readonly", False)
    _shell_tls.readonly = bool(flag)
    try:
        yield
    finally:
        _shell_tls.readonly = prev


@tool("shell", "Run a shell command (sandboxed env; dangerous patterns blocked; "
      "timeout clamped to 60s; LOOMWEAVER_SAFE_MODE=1 disables)",
      {"cmd": "str", "timeout": "int=60"})
def shell(cmd, timeout=60):
    if SAFE_MODE:
        return "blocked: safe mode enabled (shell disabled)"
    # `readonly` comes from thread-local caller context, never from the tool
    # arguments: a model that could pass readonly=False would simply pass it.
    ok, reason = security.check_shell(cmd, readonly=shell_readonly_active())
    if not ok:
        return f"blocked: {reason}"
    try:
        # audit run-001 N4: model-controlled timeout is clamped server-side
        timeout = max(1, min(int(timeout or 60), 60))
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                           timeout=timeout, env=security.sanitized_env())
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout}s"
    out = (r.stdout + r.stderr).strip()
    out = redact(out)  # audit run-001 C3: same redaction as read_file
    return f"exit={r.returncode}\n{out[:3000]}"


@tool("remember", "Store a durable fact for this session", {"key": "str", "value": "str"})
def remember(key, value, sess=None):
    if sess is not None:
        sess["facts"][key] = value
        return f"remembered: {key}"
    return "no session bound"


import sqlite3  # noqa: E402  (used by sql_query; re already imported at top)

# ------------------------------------------------------------- sql_query

# any write/DDL/attach statement is denied before it reaches SQLite; the
# read-only connection URI is a second, independent layer.
SQL_BLOCKED_RE = re.compile(
    r"\b(insert|update|delete|drop|create|alter|replace|attach|detach|"
    r"vacuum|pragma|reindex)\b", re.I)


@tool("sql_query", "Run a SELECT-only query against a SQLite database inside "
      "the project (read-only connection; writes/DDL blocked)",
      {"db_path": "str", "query": "str", "max_rows": "int=50"})
def sql_query(db_path, query, max_rows=50):
    ok, reason = security.check_path(db_path)
    if not ok:
        return f"blocked: {reason}"
    if not os.path.isfile(db_path):
        return f"error: no such database: {db_path}"
    m = SQL_BLOCKED_RE.search(query)
    if m:
        return f"blocked: only SELECT queries are permitted (found '{m.group(0)}')"
    stripped = query.strip().lstrip(";(").lower()
    if not (stripped.startswith("select") or stripped.startswith("with")):
        return "blocked: query must start with SELECT or WITH"
    try:
        conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro", uri=True)
        try:
            cur = conn.execute(query)
            rows = cur.fetchmany(max_rows + 1)
            more = len(rows) > max_rows
            cols = [d[0] for d in cur.description] if cur.description else []
        finally:
            conn.close()
    except sqlite3.Error as e:
        return f"sql error: {e}"
    body = [dict(zip(cols, r)) for r in rows[:max_rows]]
    return redact(json.dumps({"columns": cols, "rows": body,
                              "truncated": bool(more)}, default=str))


# --------------------------------------------------------- json_transform

@tool("json_transform", "Load a JSON file inside the project, apply a filter/"
      "map spec, and write the result to another project path. Spec keys: "
      "'where' ({field: value} equality filter), 'keys' (keep only these "
      "fields), 'limit' (max items). Operates on the top-level list. "
      "Write destination is subject to the write deny-set.",
      {"src_path": "str", "out_path": "str", "spec": "dict={}"})
def json_transform(src_path, out_path, spec=None):
    spec = spec or {}
    ok, reason = security.check_path(src_path)
    if not ok:
        return f"blocked: {reason} ({src_path})"
    ok, reason = security.check_write_path(out_path)
    if not ok:
        return f"blocked: {reason} ({out_path})"
    try:
        with open(src_path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, ValueError) as e:
        return f"error reading {src_path}: {e}"
    items = data if isinstance(data, list) else [data]
    where = spec.get("where")
    if isinstance(where, dict):
        items = [x for x in items if isinstance(x, dict) and
                 all(x.get(k) == v for k, v in where.items())]
    keep = spec.get("keys")
    if isinstance(keep, list) and keep:
        items = [{k: x[k] for k in keep if isinstance(x, dict) and k in x}
                 for x in items]
    limit = spec.get("limit")
    if isinstance(limit, int) and limit >= 0:
        items = items[:limit]
    payload = json.dumps(items, indent=2)
    try:
        os.makedirs(os.path.dirname(os.path.abspath(out_path)) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(payload)
    except OSError as e:
        return f"error writing {out_path}: {e}"
    return f"wrote {len(items)} item(s) to {out_path}"


# ---------------------------------------------------------- http_post_json

@tool("http_post_json", "POST a JSON body to an allowlisted public URL and "
      "return status plus the head of the response (internal/private "
      "addresses blocked)", {"url": "str", "body": "str", "max_chars": "int=2000"})
def http_post_json(url, body, max_chars=2000):
    ok, reason = security.check_url(url)
    if not ok:
        return f"blocked: {reason}"
    # validate JSON before sending so we never proxy malformed junk
    try:
        json.loads(body)
    except ValueError as e:
        return f"error: body is not valid JSON: {e}"
    req = urllib.request.Request(
        url, data=body.encode("utf-8"),
        headers={"Content-Type": "application/json",
                 "User-Agent": "flippy/0.1.0"},
        method="POST")
    try:
        with security.guarded_urlopen(req, timeout=30) as r:
            status = r.status
            head = redact(r.read().decode("utf-8", "ignore")[:max_chars])
    except urllib.error.HTTPError as e:
        return f"status={e.code}\n{e.read().decode('utf-8', 'ignore')[:max_chars]}"
    return f"status={status}\n{head}"



# ---------------------------------------------------------- aihub bridge
# Expose aihub's SAFE, guarded capabilities to the agent as tools — TTS (write
# guarded), read-only RAG query, RAG add, summarize, embed. Only registered if
# aihub imports cleanly (its third-party deps are optional/guarded), and each
# goes through the normal dispatch + safe_invoke + context-scoper gates, so a
# hostile goal only reaches them when the goal names the capability and even
# then the interception gate applies. Writes stay bounded (tts outpath guard).
try:
    import aihub as _aihub
    _AIHUB_AVAILABLE = True
except Exception:
    _AIHUB_AVAILABLE = False


if _AIHUB_AVAILABLE:
    @tool("tts", "Synthesize text to speech and save an audio file (writes under ~/aihub_tts unless an explicit safe path is given)",
          {"text": "str", "outpath": "str=''"})
    def tts_bridge(text, outpath=""):
        # outpath '' means "use default tts dir"; the aihub guard still bounds it
        return str(_aihub.tts(text, outpath=outpath or None))


    @tool("rag_query", "Search the local RAG vector store by text query (read-only), return top-k matches",
          {"q": "str", "top_k": "int=3"})
    def rag_query_bridge(q, top_k=3):
        try:
            hits = _aihub.rag_query(q, top_k=top_k) or []
        except Exception as e:
            return f"rag query unavailable: {e}"
        return "\n".join(f"[{h.get('score'):.3f}] {h.get('text','')[:300]}" for h in hits)


    @tool("rag_add", "Add a text document to the local RAG vector store (embeds and indexes it)",
          {"text": "str", "meta": "str='{}'"})
    def rag_add_bridge(text, meta="{}"):
        try:
            import json as _json
            m = _json.loads(meta) if meta else {}
        except Exception:
            m = {}
        try:
            rid = _aihub.rag_add(text, meta=m)
            return f"added to RAG store: {rid}"
        except Exception as e:
            return f"rag add unavailable: {e}"


    @tool("summarize", "Summarize a block of text into a concise paragraph (uses a cheap model)",
          {"text": "str", "max_words": "int=80"})
    def summarize_bridge(text, max_words=80):
        try:
            return _aihub.summarize(text, max_words=max_words)
        except Exception as e:
            return f"summarize unavailable: {e}"


    @tool("embed", "Embed a text string into a numeric vector (via the configured embedding provider)",
          {"texts": "str"})
    def embed_bridge(texts):
        try:
            vec = _aihub.embed(texts)
            return f"embedding dim {len(vec[0]) if vec and isinstance(vec[0], list) else len(vec)}"
        except Exception as e:
            return f"embed unavailable: {e}"


def schema_for(name):
    t = TOOLS[name]
    return {"type": "function", "function": {"name": name, "description": t["desc"],
                                            "parameters": {"type": "object", "properties": t["params"]}}}


def dispatch(name, args, sess=None):
    # audit run-001 C6: args must be a dict before any tool sees it; the
    # remember special-case previously sat outside the try/except and crashed
    # the whole agent run on {"tool": "remember", "args": null}
    if not isinstance(args, dict):
        return f"tool error: args must be an object, got {type(args).__name__}"
    if name == "remember":
        return remember(args.get("key"), args.get("value"), sess=sess)
    if name not in TOOLS:
        return f"unknown tool {name}"
    try:
        return str(TOOLS[name]["fn"](**args))
    except Exception as e:
        return f"tool error: {e}"

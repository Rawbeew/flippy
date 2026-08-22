"""Hardened tool registry for Loomweaver — every model-chosen action is guarded."""
import json
import os
import subprocess
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
    req = urllib.request.Request(url, headers={"User-Agent": "OpenAI File Downloader, XaiImageApiFetch/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "ignore")[:max_chars]


@tool("read_file", "Read a file inside the project (credential paths denied)",
      {"path": "str", "max_chars": "int=4000"})
def read_file(path, max_chars=4000):
    ok, reason = security.check_path(path)
    if not ok:
        return f"blocked: {reason}"
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read(max_chars)


@tool("write_file", "Write a file inside the project (credential paths denied)",
      {"path": "str", "content": "str"})
def write_file(path, content):
    ok, reason = security.check_path(path)
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


@tool("shell", "Run a shell command (sandboxed env; dangerous patterns blocked; "
      "LOOMWEAVER_SAFE_MODE=1 disables)", {"cmd": "str", "timeout": "int=60"})
def shell(cmd, timeout=60):
    if SAFE_MODE:
        return "blocked: safe mode enabled (shell disabled)"
    ok, reason = security.check_shell(cmd)
    if not ok:
        return f"blocked: {reason}"
    try:
        r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True,
                           timeout=timeout, env=security.sanitized_env())
    except subprocess.TimeoutExpired:
        return f"timeout after {timeout}s"
    out = (r.stdout + r.stderr).strip()
    # strip anything that looks like a key from output before it reaches the model
    out = re.sub(r"(?i)(sk-[a-z0-9_-]{10,}|gsk_[a-z0-9]{20,}|nvapi-[a-z0-9_-]{10,}|"
                 r"cfut_[a-z0-9_-]{10,}|ghp_[A-Za-z0-9]{20,})", "[REDACTED]", out)
    return f"exit={r.returncode}\n{out[:3000]}"


@tool("remember", "Store a durable fact for this session", {"key": "str", "value": "str"})
def remember(key, value, sess=None):
    if sess is not None:
        sess["facts"][key] = value
        return f"remembered: {key}"
    return "no session bound"


import re  # noqa: E402
import sqlite3  # noqa: E402

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
    return json.dumps({"columns": cols, "rows": body,
                       "truncated": bool(more)}, default=str)


# --------------------------------------------------------- json_transform

@tool("json_transform", "Load a JSON file inside the project, apply a filter/"
      "map spec, and write the result to another project path. Spec keys: "
      "'where' ({field: value} equality filter), 'keys' (keep only these "
      "fields), 'limit' (max items). Operates on the top-level list.",
      {"src_path": "str", "out_path": "str", "spec": "dict={}"})
def json_transform(src_path, out_path, spec=None):
    spec = spec or {}
    for p in (src_path, out_path):
        ok, reason = security.check_path(p)
        if not ok:
            return f"blocked: {reason} ({p})"
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
                 "User-Agent": "OpenAI File Downloader, XaiImageApiFetch/1.0"},
        method="POST")
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            status = r.status
            head = r.read().decode("utf-8", "ignore")[:max_chars]
    except urllib.error.HTTPError as e:
        return f"status={e.code}\n{e.read().decode('utf-8', 'ignore')[:max_chars]}"
    return f"status={status}\n{head}"


def schema_for(name):
    t = TOOLS[name]
    return {"type": "function", "function": {"name": name, "description": t["desc"],
                                            "parameters": {"type": "object", "properties": t["params"]}}}


def dispatch(name, args, sess=None):
    if name == "remember":
        return remember(args.get("key"), args.get("value"), sess=sess)
    if name not in TOOLS:
        return f"unknown tool {name}"
    try:
        return str(TOOLS[name]["fn"](**args))
    except Exception as e:
        return f"tool error: {e}"

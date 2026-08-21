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
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Chrome/126.0"})
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

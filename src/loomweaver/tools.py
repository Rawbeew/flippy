"""tools.py — built-in tool registry for the agent harness. Stdlib only."""
import json
import os
import subprocess
import urllib.request

TOOLS = {}


def tool(name, desc, params):
    def deco(fn):
        TOOLS[name] = {"desc": desc, "params": params, "fn": fn}
        return fn
    return deco


@tool("http_get", "Fetch a URL and return text (first N chars)", {"url": "str", "max_chars": "int=2000"})
def http_get(url, max_chars=2000):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 Chrome/126.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode("utf-8", "ignore")[:max_chars]


@tool("read_file", "Read a local file", {"path": "str", "max_chars": "int=4000"})
def read_file(path, max_chars=4000):
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read(max_chars)


@tool("write_file", "Write text to a local file", {"path": "str", "content": "str"})
def write_file(path, content):
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)
    return f"wrote {len(content)} chars to {path}"


@tool("list_dir", "List a directory", {"path": "str"})
def list_dir(path="."):
    return "\n".join(sorted(os.listdir(path)))


@tool("shell", "Run a shell command (git-bash on Windows), return stdout+stderr", {"cmd": "str", "timeout": "int=60"})
def shell(cmd, timeout=60):
    r = subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=timeout)
    out = (r.stdout + r.stderr).strip()
    return f"exit={r.returncode}\n{out[:3000]}"


@tool("remember", "Store a durable fact for this session", {"key": "str", "value": "str"})
def remember(key, value, sess=None):
    if sess is not None:
        sess["facts"][key] = value
        return f"remembered: {key}"
    return "no session bound"


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

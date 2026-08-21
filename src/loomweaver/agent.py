"""agent.py — the agent runner: goal -> (plan -> act -> observe)* -> done.

Tool-calling via native OpenAI-style tool_calls when the provider supports it,
falling back to a JSON-protocol prompt when it doesn't. Every step emits events.
"""
import json
import re

from . import tools
from .core import RunLog, SessionStore, route

SYSTEM = """You are a terse autonomous agent. Achieve the user's goal using the available tools.
Rules:
- Think step by step, but output only what's needed.
- When you need a tool, call it. When the goal is achieved, reply with DONE: <one-line summary>.
- Never invent tool results."""


def _parse_json_action(text):
    """Fallback protocol: model returns {"tool": name, "args": {...}} or {"done": "..."}."""
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        if "tool" in d:
            return ("tool", d["tool"], d.get("args", {}))
        if "done" in d:
            return ("done", d["done"], None)
    except Exception:
        pass
    return None


MAX_SESSION_MESSAGES = 40  # keep context bounded; oldest non-system messages dropped


def run(goal, session_id=None, max_steps=10, model=None, creds=None, runs_dir=None, verbose=True):
    runlog = RunLog(runs_dir)
    store = SessionStore()
    sess = store.load(session_id or "default")
    if not sess["messages"]:
        sess["messages"].append({"role": "system", "content": SYSTEM})
    sess["messages"].append({"role": "user", "content": f"GOAL: {goal}"})

    tool_list = ", ".join(tools.TOOLS)
    sess["messages"].append({"role": "user", "content":
        f"AVAILABLE TOOLS: {tool_list}\n"
        'To use one, reply with ONLY JSON: {"tool": "<name>", "args": {...}}. '
        'When the goal is achieved, reply with ONLY JSON: {"done": "<summary>"}.'})

    tool_schemas = [tools.schema_for(n) for n in tools.TOOLS]
    runlog.emit({"type": "run_start", "goal": goal, "session": sess["id"], "model": model})

    final = None
    for step in range(1, max_steps + 1):
        r = route(sess["messages"], model=model, creds=creds,
                  on_event=lambda e: runlog.emit(e))
        if not r.get("ok"):
            runlog.emit({"type": "run_error", "step": step, "error": r.get("error")})
            final = f"LLM error: {r.get('error')}"
            break
        msg = {"role": "assistant", "content": r["text"]}
        sess["messages"].append(msg)
        runlog.emit({"type": "agent_step", "step": step, "text": r["text"][:500],
                     "provider": r["provider"]})
        if verbose:
            print(f"[step {step}] {r['provider']}: {r['text'][:160]}")

        # native tool calls path (if provider returned them we'd handle here;
        # freeinference/groq mostly inline them, so use JSON fallback protocol)
        action = _parse_json_action(r["text"])
        if action and action[0] == "done":
            final = action[1]
            runlog.emit({"type": "run_done", "step": step, "summary": final})
            break
        if action and action[0] == "tool":
            _, name, args = action
            obs = tools.dispatch(name, args, sess=sess)
            sess["messages"].append({"role": "user",
                                     "content": f"TOOL_RESULT {name}: {str(obs)[:1500]}"})
            runlog.emit({"type": "tool_call", "step": step, "tool": name,
                         "args": args, "result": str(obs)[:300]})
            continue

        # no explicit action: if text mentions DONE treat as done, else nudge once
        if "DONE:" in r["text"]:
            final = r["text"].split("DONE:", 1)[1].strip()
            runlog.emit({"type": "run_done", "step": step, "summary": final})
            break
        sess["messages"].append({"role": "user", "content":
            'Continue. Use {"tool": "...", "args": {...}} to act, or {"done": "..."} when finished.'})
    else:
        final = "max steps reached"
        runlog.emit({"type": "run_done", "step": max_steps, "summary": final})

    # trim session so repeated runs don't grow context unboundedly (keep system + recent)
    if len(sess["messages"]) > MAX_SESSION_MESSAGES:
        sys_msgs = [m for m in sess["messages"][:1] if m["role"] == "system"]
        rest = sess["messages"][1:]
        sess["messages"] = sys_msgs + rest[-(MAX_SESSION_MESSAGES - len(sys_msgs)):]

    store.save(sess)
    return {"result": final, "session": sess["id"], "run_dir": runlog.dir,
            "events": runlog.read()}

"""agent.py — the agent runner: goal -> (plan -> act -> observe)* -> done.

Tool-calling via native OpenAI-style tool_calls when the provider supports it,
falling back to a JSON-protocol prompt when it doesn't. Every step emits events.
"""
import json

from . import tools
from . import observability
from .core import RunLog, SessionStore, route

SYSTEM = """You are a terse autonomous agent. Achieve the user's goal using the available tools.
Rules:
- Think step by step, but output only what's needed.
- When you need a tool, call it. When the goal is achieved, reply with DONE: <one-line summary>.
- Never invent tool results."""


def _balanced_spans(text):
    """Yield outermost {...} spans, skipping braces inside quoted strings.

    A plain brace-depth count misreads `{"content": "}"}` as closing early,
    because it counts braces that sit inside JSON string values. This scanner
    tracks string state and backslash escapes, so only structural braces
    count. Unbalanced input yields nothing.
    """
    n = len(text)
    i = 0
    while i < n:
        if text[i] != "{":
            i += 1
            continue
        depth = 0
        in_str = False
        j = i
        while j < n:
            ch = text[j]
            if in_str:
                if ch == "\\":
                    j += 2  # escaped char inside a string — never structural
                    continue
                if ch == '"':
                    in_str = False
            else:
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        yield text[i:j + 1]
                        i = j + 1
                        break
            j += 1
        else:
            return  # ran off the end without closing — no more spans


def _parse_json_action(text):
    """Fallback protocol: model returns {"tool": name, "args": {...}} or {"done": "..."}.

    Hardened against how free-tier models actually emit actions: wrapped in
    prose, inside markdown fences, with brace-containing string values, or
    with stray braces in surrounding commentary. The old greedy regex
    (`\\{.*\\}`) failed on the last two; a string-aware balanced-span scan
    recovers them. Spans are tried in order; the first that parses as a
    dict with a "tool" or "done" key wins.
    """
    text = text or ""
    # strip markdown code fences that free-tier models love to add
    for fence in ("```json", "```"):
        if fence in text:
            text = text.replace(fence, "")
    for span in _balanced_spans(text):
        try:
            d = json.loads(span)
        except Exception:
            continue
        if isinstance(d, dict):
            if "tool" in d:
                return ("tool", d["tool"], d.get("args", {}))
            if "done" in d:
                return ("done", d["done"], None)
    return None


def _native_schemas():
    """Convert tools.TOOLS into OpenAI tool-call JSON schemas (native path).

    Tool names keep their underscores (safe_invoke dispatches on the exact
    name), and parameter names match the fn signature keys.
    """
    schemas = []
    for name, spec in tools.TOOLS.items():
        props, req = {}, []
        for pname, ptype in (spec.get("params") or {}).items():
            base = ptype.split("=")[0].strip()
            # map declared type hints to JSON schema types
            t = {"str": "string", "int": "integer", "float": "number",
                 "bool": "boolean"}.get(base, "string")
            d = {"type": t, "description": ""}
            if ptype.startswith("dict"):
                d["type"] = "object"
            default = ptype.split("=", 1)[1].strip() if "=" in ptype else ""
            if default:
                d["description"] = f"optional, default {default}"
            props[pname] = d
            if "=" not in ptype:
                req.append(pname)
        schemas.append({
            "type": "function",
            "function": {
                "name": name,
                "description": spec.get("desc", ""),
                "parameters": {"type": "object", "properties": props,
                               "required": req, "additionalProperties": False},
            },
        })
    return schemas

MAX_SESSION_MESSAGES = 40  # keep context bounded; oldest non-system messages dropped

OBS_TRUNC = 2000  # chars of tool output fed back to the model (budgeted)
NUDGE_MAX = 2     # consecutive no-progress nudges before we stop the run


def run(goal, session_id=None, max_steps=10, model=None, creds=None, runs_dir=None, verbose=True,
        native_tools=True):
    runlog = RunLog(runs_dir)
    store = SessionStore()
    sess = store.load(session_id or "default")
    if not sess["messages"]:
        sess["messages"].append({"role": "system", "content": SYSTEM})
    sess["messages"].append({"role": "user", "content": f"GOAL: {goal}"})

    # Build native tool schemas once (only if requested). Handled tool_calls
    # append a "tool" role result message back into the transcript, which is
    # what native-tool providers expect between turns.
    native_schemas = _native_schemas() if native_tools else None

    tool_list = ", ".join(tools.TOOLS)
    sess["messages"].append({"role": "user", "content":
        f"AVAILABLE TOOLS: {tool_list}\n"
        'To use one, reply with ONLY JSON: {"tool": "<name>", "args": {...}}. '
        'When the goal is achieved, reply with ONLY JSON: {"done": "<summary>"}.'})

    runlog.emit({"type": "run_start", "goal": goal, "session": sess["id"], "model": model})

    final = None
    nudges = 0  # consecutive steps with no tool call and no done — loop detection
    for step in range(1, max_steps + 1):
        r = route(sess["messages"], model=model, creds=creds,
                  on_event=lambda e: runlog.emit(e), tools=native_schemas)
        if not r.get("ok"):
            runlog.emit({"type": "run_error", "step": step, "error": r.get("error")})
            final = f"LLM error: {r.get('error')}"
            break

        # Native tool call(s) from a native-tool provider (e.g. Groq gpt-oss).
        if r.get("tool_calls"):
            calls = r["tool_calls"]
            runlog.emit({"type": "agent_tool_calls", "step": step,
                         "calls": [c["name"] for c in calls]})
            for c in calls:
                name, args = c["name"], c["arguments"] or {}
                if not isinstance(args, dict):
                    # malformed/weird native arguments (e.g. a JSON array) — treat
                    # as empty so dispatch never sees a non-dict it can't unpack.
                    args = {}
                obs, intercepted = observability.safe_invoke(name, args, tools.dispatch, sess=sess)
                if len(obs) > OBS_TRUNC:
                    obs = obs[:OBS_TRUNC] + f"\n...[truncated, {len(obs) - OBS_TRUNC} more chars]"
                sess["messages"].append({"role": "tool", "tool_call_id": c.get("id", ""),
                                         "name": name, "content": str(obs)})
                tev = {"type": "tool_call", "step": step, "tool": name,
                       "args": args, "result": str(obs)[:300]}
                if intercepted:
                    tev["intercepted"] = True
                runlog.emit(tev)
            nudges = 0
            if verbose:
                print(f"[step {step}] native tools: {[c['name'] for c in calls]}")
            continue  # loop again with the tool results in context

        msg = {"role": "assistant", "content": r["text"]}
        sess["messages"].append(msg)
        runlog.emit({"type": "agent_step", "step": step, "text": r["text"][:500],
                     "provider": r["provider"]})
        if verbose:
            print(f"[step {step}] {r['provider']}: {r['text'][:160]}")

        # actions ride on the model's plain-text reply, parsed as JSON
        # (the JSON-protocol fallback; providers inline tool calls as text)
        action = _parse_json_action(r["text"])
        if action and action[0] == "done":
            final = action[1]
            runlog.emit({"type": "run_done", "step": step, "summary": final})
            break
        if action and action[0] == "tool":
                    _, name, args = action
                    # Request dispatch with input inspection; benign actions pass through.
                    obs, intercepted = observability.safe_invoke(name, args, tools.dispatch, sess=sess)
                    # budgeted observation: cap what re-enters context, mark truncation
                    if len(obs) > OBS_TRUNC:
                        obs = obs[:OBS_TRUNC] + f"\n...[truncated, {len(obs) - OBS_TRUNC} more chars]"
                    sess["messages"].append({"role": "user",
                                             "content": f"TOOL_RESULT {name}: {obs}"})
                    tool_event = {"type": "tool_call", "step": step, "tool": name,
                                  "args": args, "result": str(obs)[:300]}
                    if intercepted:
                        # surface the diversion: a benign call that got intercepted
                        # must be auditable, not silently fabricated.
                        tool_event["intercepted"] = True
                    runlog.emit(tool_event)
                    nudges = 0  # a tool call is progress
                    continue

        # no explicit action: if text mentions DONE treat as done, else nudge
        if "DONE:" in r["text"]:
            final = r["text"].split("DONE:", 1)[1].strip()
            runlog.emit({"type": "run_done", "step": step, "summary": final})
            break
        nudges += 1
        if nudges > NUDGE_MAX:
            final = ("stopped: model produced no actionable step after "
                     f"{NUDGE_MAX} nudges")
            runlog.emit({"type": "run_done", "step": step, "summary": final,
                         "reason": "no_progress"})
            break
        sess["messages"].append({"role": "user", "content":
            'Continue. Use {"tool": "...", "args": {...}} to act, or {"done": "..."} when finished.'})
    else:
        final = "max steps reached"
        runlog.emit({"type": "run_done", "step": max_steps, "summary": final})

    # keep bounded context: preserve the system message(s) plus the most recent
    # non-system messages, regardless of where the system message sits.
    if len(sess["messages"]) > MAX_SESSION_MESSAGES:
        sys_msgs = [m for m in sess["messages"] if m.get("role") == "system"]
        non_sys = [m for m in sess["messages"] if m.get("role") != "system"]
        keep = max(MAX_SESSION_MESSAGES - len(sys_msgs), 0)
        sess["messages"] = sys_msgs + non_sys[-keep:]

    store.save(sess)
    return {"result": final, "session": sess["id"], "run_dir": runlog.dir,
            "events": runlog.read()}

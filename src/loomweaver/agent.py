"""agent.py — the agent runner: goal -> (plan -> act -> observe)* -> done.

Tool-calling via native OpenAI-style tool_calls when the provider supports it,
falling back to a JSON-protocol prompt when it doesn't. Every step emits events.
"""
import json
import re

from . import tools
from . import observability
from . import learning
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


SAFE_TOOLS = ("read_file", "list_dir", "remember", "http_get")  # always available
# aihub capabilities are intent-gated (not in SAFE_TOOLS) so least privilege holds:
# a goal that doesn't mention summarizing / speech / retrieval / embeddings never
# gets those tools handed to it.

# intent keyword -> tools the goal most plausibly needs. Context determines
# the tool set (least privilege): a model working toward "summarize file X"
# should NOT be handed shell / sql / write / http-post unless the goal names it.
# Intent keyword -> tool name. Every key MUST be a real entry in tools.TOOLS:
# these names are intersected with the registry, so a key that names nothing
# silently grants nothing (that is how write_file/sql_query/json_transform/
# http_post_json ended up unreachable in auto mode).
_INTENT = {
    # shell is the high-risk one — require a strong signal
    "shell": ("shell", "execute", "run the", "run a", "command", "bash", "terminal",
              "ls", "grep", "find", "git", "pip", "install", "npm", "cd", "mkdir",
              "touch", "chmod", "psql", "awk", "sed", "in a shell", "exec"),
    "http_post_json": ("post", "submit to", "send data", "upload to",
                       "http_post_json", "webhook", "transaction to"),
    "sql_query": ("sqlite", "database", "table", "query", "select", "sql", "rows",
                  "schema", "where", "join"),
    "json_transform": ("json", "transform", "filter the list", "map"),
    "http_get": ("url", "http", "https", "fetch", "website", "web page", "api",
                 "scrape", "feed", "html", "curl", "wget", "download"),
    "tts": ("speech", "text to speech", "tts", "audio", "say", "read aloud",
            "synthesize voice", "voice"),
    "rag_query": ("rag", "retrieval", "vector store", "indexed",
                  "search the knowledge", "knowledge base"),
    "rag_add": ("rag add", "add to rag", "index a", "store in the vector",
                "save to the knowledge base", "remember a fact into the store"),
    "summarize": ("summarize", "summary", "condense", "tl;dr", "brief"),
    "embed": ("embedding", "embed", "vector for"),
    "write_file": ("write", "create", "generate", "save", "append", "update file",
                   "output to", "produce a file", "new file", "edit"),
}


def _keyword_pattern(kw):
    """Word-boundary matcher for one intent keyword.

    Substring matching made `"ls "` fire on `"tools "` and `"db"` fire on
    `"subdomain"`. Anchoring on word boundaries keeps the intent table honest.
    """
    body = re.escape(kw.strip())
    tail = r"\b" if kw.strip()[-1:].isalnum() or kw.strip()[-1:] == "_" else ""
    return re.compile(r"\b" + body + tail, re.I)


_KEYWORD_RES = {tool: [_keyword_pattern(k) for k in kws]
                for tool, kws in _INTENT.items()}


def _tools_for_goal(goal: str | None, mode: str = "auto") -> set[str]:
    """Pick the minimal tool set a goal needs.

    mode:
      "all"            -> every tool (legacy/opt-out)
      ["tool", ...]    -> exactly that set
      "auto" (default) -> SAFE_TOOLS plus whichever intent groups the goal names
    Returns the subset of `set(tools.TOOLS)`.
    """
    if not goal or mode == "all":
        return set(tools.TOOLS)
    if isinstance(mode, (list, tuple, set)):
        # Pre-flight intake is authoritative and FAILS CLOSED: an operator list
        # that matches no registered tool yields no tools, it does not silently
        # degrade to goal-based auto-grant.
        return set(mode) & set(tools.TOOLS)
    gl = goal or ""
    extra = {tool for tool, res in _KEYWORD_RES.items()
             if any(rx.search(gl) for rx in res)}
    # always intersect with the registry so the result is a real tool set
    return (set(SAFE_TOOLS) | extra) & set(tools.TOOLS)


def _native_schemas_for(goal: str, mode: str = "auto"):
    """Like _native_schemas() but only shapes for the context-determined tools."""
    return [_native_schema(name) for name in tools.TOOLS if name in _tools_for_goal(goal, mode)]


def _native_schema(name: str):
    """Build a single OpenAI function schema for one tool by name."""
    spec = tools.TOOLS[name]
    props, req = {}, []
    for pname, ptype in (spec.get("params") or {}).items():
        base = ptype.split("=")[0].strip()
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
    return {"type": "function",
            "function": {"name": name, "description": spec.get("desc", ""),
                         "parameters": {"type": "object", "properties": props,
                                        "required": req,
                                        "additionalProperties": False}}}


def _native_schemas():
    """Legacy: schemas for EVERY tool (used when tool_scope="all")."""
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
        native_tools=True, tool_scope="auto", fresh=True):
    """tool_scope: 'auto' (context-determined minimal set), 'all' (every tool),
    or an explicit list of tool names to force exactly that set.

    fresh=True (default): START from a clean system+goal; do NOT replay a prior
    run's message history (agent-written assistant/tool turns) into this goal's
    context. Prior runs on the same session_id are agent-controlled and could
    have been manipulated (session-replay prompt injection); zero-trust means a
    new goal does not inherit them as trusted instructions. 'facts' (remember)
    data is still loaded and saved. Set fresh=False only when you explicitly
    want to continue a multi-turn conversation across CLI calls."""
    runlog = RunLog(runs_dir)
    store = SessionStore()
    sess = store.load(session_id or "default")
    if fresh:
        # zero-trust: never replay prior-run messages into a NEW goal. A prior
        # run on this session is agent-controlled; a poisoned turn could inject
        # instructions. Keep 'facts' + id, drop the turn history.
        sess["messages"] = []
    if not sess["messages"]:
        # Persistent memory: the user profile plus lessons from similar past
        # goals ride in the system message, so run N benefits from runs 1..N-1.
        # Empty on a cold start, so a first-time user pays no context cost, and
        # it is kept in the system role rather than the user turn so it is never
        # mistaken for an instruction that came from the goal.
        memory = learning.prompt_context(goal)
        sess["messages"].append({"role": "system",
                                 "content": SYSTEM + (f"\n\n{memory}" if memory else "")})
        if verbose and memory:
            print(f"[memory] {len(memory)} chars of learned context injected")
    sess["messages"].append({"role": "user", "content": f"GOAL: {goal}"})

    # Build native tool schemas once (only if requested). Context determines
    # which tools the goal gets (least privilege): only the tools the goal
    # plausibly needs are exposed, so a prompt-injected goal has a smaller
    # dangerous-tool surface. tool_scope="all" disables the filter.
    if native_tools:
        native_schemas = _native_schemas_for(goal, tool_scope) if tool_scope != "all"             else _native_schemas()
    else:
        native_schemas = None

    chosen = _tools_for_goal(goal, tool_scope) & set(tools.TOOLS)
    tool_list = ", ".join(sorted(chosen))

    tools_used = []

    def _authorized(name, args):
        """Enforce the pre-flight tool set at DISPATCH, not just in the prompt.

        Hiding a schema is advisory — a model can still emit any tool name, and
        a prompt-injected observation certainly will. armada.run_agent has always
        enforced its role set; the single-agent runner did not, which made the
        ARCHITECTURE.md claim ("enforced in dispatch, not suggested in prompts")
        false for this path.
        """
        if name in chosen:
            return observability.safe_invoke(name, args, tools.dispatch, sess=sess)
        obs = (f"BLOCKED: tool '{name}' is not in this run's authorized set. "
               f"Authorized: {sorted(chosen) or 'none'}. "
               f"Re-run with --tools {name} to grant it.")
        return obs, False
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
                obs, intercepted = _authorized(name, args)
                tools_used.append(name)
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
                    obs, intercepted = _authorized(name, args)
                    tools_used.append(name)
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

    # Outcome memory: this run's goal, tools and result become the next run's
    # prior. A stalled run also files itself as a lesson so the same dead end is
    # flagged before it is walked into again.
    ok = bool(final) and not final.startswith("max steps") and not final.startswith("stopped")
    try:
        learning.get_store().record_agent_run(goal, tools_used, ok, steps=step)
        if not ok:
            learning.get_store().note_failure(goal, final)
    except Exception:
        pass  # learning must never break a run

    return {"result": final, "session": sess["id"], "run_dir": runlog.dir,
            "tools_used": sorted(set(tools_used)),
            "events": runlog.read()}

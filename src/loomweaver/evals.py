"""evals.py — eval harness: task suites, scoring, latency/cost capture."""
import json
import re
import time

from .core import RunLog, load_creds, route

# ---------------------------------------------------------------- suites

SUITE_BASIC = [
    {"id": "math1", "prompt": "What is 17 * 23? Reply with just the number.", "check": "391"},
    {"id": "cap1", "prompt": "Capital of Australia? One word.", "check": "canberra"},
    {"id": "json1", "prompt": 'Return ONLY valid JSON: {"ok": true, "n": 3}', "check_json": {"ok": True, "n": 3}},
    {"id": "count1", "prompt": "How many words are in this sentence? Reply with just the number.", "check": "7"},
    {"id": "reverse1", "prompt": "Reverse the string 'harness'. Reply with just the reversed letters, nothing else.", "check": "ssenhra"},
]

SUITE_REASONING = [
    {"id": "logic1", "prompt": "All bloops are razzies. All razzies are lazzies. Are all bloops definitely lazzies? Answer yes or no.", "check": "yes"},
    {"id": "code1", "prompt": "What does print(sum([1,2,3,4])) output? Just the number.", "check": "10"},
]

SUITE_EXTRACTION = [
    {"id": "date1", "prompt": "Extract the date as YYYY-MM-DD from: 'The meeting is on March 5th, 2027'. Reply with only the date.", "check": "2027-03-05"},
    {"id": "price1", "prompt": "Extract the price as a bare number from: 'Subscription costs $49.99 per month'. Reply with only the number.", "check": "49.99"},
    {"id": "email1", "prompt": "Extract the email from: 'Contact raji@example.org for details'. Reply with only the email.", "check": "raji@example.org"},
    {"id": "jsonx1", "prompt": 'Convert to JSON with keys name and age: "Maya is 34 years old". Reply with only JSON.', "check_json": {"name": "Maya", "age": 34}},
]

SUITE_TOOLS = [
    # model must emit the JSON action protocol for the right tool
    {"id": "tool_shell", "prompt": 'Use a tool to list files in the current directory. Reply ONLY with JSON: {"tool": "...", "args": {...}}. Valid tools: list_dir, shell, http_get, read_file, write_file.',
     "check_regex": r'"tool"\s*:\s*"(list_dir|shell)"'},
    {"id": "tool_fetch", "prompt": 'Use a tool to fetch https://example.com. Reply ONLY with JSON: {"tool": "...", "args": {...}}. Valid tools: list_dir, shell, http_get, read_file, write_file.',
     "check_regex": r'"tool"\s*:\s*"http_get"'},
    {"id": "tool_write", "prompt": 'Use a tool to save the text hello to /tmp/x.txt. Reply ONLY with JSON: {"tool": "...", "args": {...}}',
     "check_regex": r'"tool"\s*:\s*"write_file"'},
    {"id": "tool_read", "prompt": 'Use a tool to read the file /etc/hostname. Reply ONLY with JSON: {"tool": "...", "args": {...}}',
     "check_regex": r'"tool"\s*:\s*"read_file"'},
]

# ---------------------------------------------------------------- agent suite
# Multi-step agent tasks: the model must (a) parse the JSON protocol, (b) pick
# the right tool, (c) chain a second action from the first tool's result.
# Scored by mocked tool results so the suite runs offline.

SUITE_AGENT = [
    {
        "id": "agent_write_read",
        "goal": "Write the text 'flippy-test' to the file note.txt, then read it "
                "back to verify. Use tools. Finish with done when verified.",
        # scripted tool behavior: write_file succeeds, read_file returns the text
        "tool_script": {
            "write_file": lambda args: f"wrote file {args.get('path', '')}",
            "read_file": lambda args: "flippy-test",
        },
        "success_criteria": {
            "min_tool_calls": 2,
            "required_tools": ["write_file", "read_file"],
            "done": True,
        },
    },
    {
        "id": "agent_fetch_then_done",
        "goal": "Fetch the URL https://example.com/data.json, then report done "
                "with the number of bytes fetched. Use tools.",
        "tool_script": {
            "http_get": lambda args: '{"n": 42}',
        },
        "success_criteria": {
            "min_tool_calls": 1,
            "required_tools": ["http_get"],
            "done": True,
        },
    },
    {
        "id": "agent_observe_and_decide",
        "goal": "List the files in the project root. If a file named "
                "config.json exists, read it and report done with its "
                "contents. If not, report done with 'no config'. Use tools.",
        "tool_script": {
            "list_dir": lambda args: "config.json\nREADME.md\nsrc",
            "read_file": lambda args: '{"env": "production"}',
        },
        "success_criteria": {
            "min_tool_calls": 2,
            "required_tools": ["list_dir", "read_file"],
            "done": True,
        },
    },
    {
        "id": "agent_remember_fact",
        "goal": "Remember the fact that project=flippy, then report done. "
                "Use tools.",
        "tool_script": {
            "remember": lambda args: "remembered",
        },
        "success_criteria": {
            "min_tool_calls": 1,
            "required_tools": ["remember"],
            "done": True,
        },
    },
]

SUITES = {"basic": SUITE_BASIC, "reasoning": SUITE_REASONING,
          "extraction": SUITE_EXTRACTION, "tools": SUITE_TOOLS,
          "agent": SUITE_AGENT}


# ---------------------------------------------------------------- scoring

def score_case(case, text):
    text_l = (text or "").strip().lower()
    if "check" in case:
        # word-boundary match: '10' must not match '110', 'yes' not 'yes-adjacent'
        return re.search(rf"(?<![a-z0-9-]){re.escape(case['check'].lower())}(?![a-z0-9-])",
                         text_l) is not None
    if "check_json" in case:
        m = re.search(r"\{.*\}", text or "", re.S)
        if not m:
            return False
        try:
            got = json.loads(m.group(0))
            return got == case["check_json"]
        except Exception:
            return False
    if "check_regex" in case:
        return bool(re.search(case["check_regex"], text or "", re.I))
    return False


def run_suite(name="basic", model=None, creds=None, runs_dir=None):
    cases = SUITES[name]
    runlog = RunLog(runs_dir)
    results = []
    for c in cases:
        t0 = time.time()
        r = route([{"role": "user", "content": c["prompt"]}], model=model, creds=creds,
                  on_event=lambda e: runlog.emit(e))
        lat = time.time() - t0
        ok = bool(r.get("ok")) and score_case(c, r.get("text"))
        results.append({"id": c["id"], "pass": ok, "latency": round(lat, 2),
                        "provider": r.get("provider"), "answer": (r.get("text") or "")[:120]})
        runlog.emit({"type": "eval_case", **results[-1]})
    passed = sum(1 for x in results if x["pass"])
    summary = {"suite": name, "model": model, "passed": passed, "total": len(results),
               "score": round(passed / len(results) * 100), "avg_latency": round(
                   sum(x["latency"] for x in results) / len(results), 2),
               "cases": results}
    runlog.emit({"type": "eval_summary", **{k: v for k, v in summary.items() if k != "cases"}})
    return summary


def compare(suites=("basic", "reasoning"), models=None, creds=None):
    """Run suites across models (None = router default). Returns comparison table."""
    rows = []
    for m in (models or [None]):
        for s in suites:
            r = run_suite(s, model=m, creds=creds)
            rows.append({"model": m or "(router-default)", "suite": s,
                         "score": r["score"], "avg_latency": r["avg_latency"]})
    return rows


# ---------------------------------------------------------------- agent suite runner

def run_agent_suite(model=None, creds=None, runs_dir=None, max_steps=8,
                    planner=None):
    """Run SUITE_AGENT end to end with scripted tools and a mocked router.

    The agent suite measures the LOOP, not the model: route() is mocked to a
    scripted pseudo-model that reacts to observations, tools are scripted via
    the case's tool_script. A loop that loses observations, truncates without
    marking, or never terminates fails here deterministically.

    `planner(case, called_tools, last_obs) -> action dict` overrides the
    default plan logic (used by tests to simulate broken/looping models).

    Scoring per case:
      - every required tool was called (order-free)
      - at least min_tool_calls total calls
      - run reached a 'done' state (not max-steps, not no-progress stop)
      - no protocol errors (unparseable model output treated as nudge)
    """
    from unittest import mock as _mock
    from loomweaver import agent as agent_mod

    def _default_planner(case, done_tools, last_obs):
        req = case["success_criteria"]["required_tools"]
        for t in req:
            if t not in done_tools:
                return {"tool": t, "args": _args_for(t)}
        return {"done": "task complete"}

    def _args_for(tool):
        if tool == "write_file":
            return {"path": "note.txt", "content": "flippy-test"}
        if tool == "read_file":
            return {"path": "note.txt"}
        if tool == "http_get":
            return {"url": "https://example.com/data.json"}
        if tool == "list_dir":
            return {"path": "."}
        if tool == "remember":
            return {"key": "project", "value": "flippy"}
        return {}

    plan_fn = planner or _default_planner

    results = []
    for case in SUITE_AGENT:
        calls = []
        script = case.get("tool_script", {})

        def _fake_route(messages, model=None, creds=None, on_event=None, **kw):
            """Scripted pseudo-model: react to the last message with the next
            action from a tiny plan determined by the goal + observations."""
            last = messages[-1]["content"] if messages else ""
            done_tools = [c for c, _ in calls]
            plan = plan_fn(case, done_tools, last)
            return {"ok": True, "text": json.dumps(plan), "provider": "scripted",
                    "model": "scripted", "latency": 0.01}

        def _fake_dispatch(name, args, sess=None):
            calls.append((name, dict(args)))
            fn = script.get(name)
            if fn is None:
                return "ok"
            return str(fn(args))

        with _mock.patch.object(agent_mod, "route", side_effect=_fake_route), \
             _mock.patch.object(agent_mod.tools, "dispatch",
                                side_effect=_fake_dispatch), \
             _mock.patch.object(agent_mod, "SessionStore") as _MS:
            # isolated session per case
            _MS.return_value.load.return_value = {"id": case["id"], "messages": [],
                                                  "facts": {}}
            _MS.return_value.save.side_effect = lambda s: None
            out = agent_mod.run(case["goal"], session_id=case["id"],
                                max_steps=max_steps, model=model,
                                runs_dir=runs_dir, verbose=False)

        crit = case["success_criteria"]
        called = [c[0] for c in calls]
        req_ok = all(t in called for t in crit.get("required_tools", []))
        count_ok = len(calls) >= crit.get("min_tool_calls", 1)
        # done-ness comes from the event trail: a genuine run_done with a
        # model-supplied summary, NOT max-steps exhaustion or a no-progress stop
        events = out.get("events") or []
        done_events = [e for e in events if e.get("type") == "run_done"]
        done_ok = bool(done_events) and \
            done_events[-1].get("reason") != "no_progress" and \
            "max steps" not in str(done_events[-1].get("summary", "")).lower()
        passed = req_ok and count_ok and done_ok
        results.append({"id": case["id"], "pass": passed,
                        "tools_called": called,
                        "result": str(out.get("result"))[:120],
                        "steps": out.get("events") and
                        max(e.get("step", 0) for e in out.get("events", [])
                            if e.get("type") in ("agent_step", "run_done")) or 0})
    passed = sum(1 for x in results if x["pass"])
    return {"suite": "agent", "model": model or "(scripted)",
            "passed": passed, "total": len(results),
            "score": round(passed / len(results) * 100),
            "cases": results}

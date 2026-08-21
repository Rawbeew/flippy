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

SUITES = {"basic": SUITE_BASIC, "reasoning": SUITE_REASONING,
          "extraction": SUITE_EXTRACTION, "tools": SUITE_TOOLS}


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

"""cron.py — in-repo scheduler: run loomweaver jobs on an interval.

Opt-in and local-only. Jobs are defined in cron_jobs.json:
  [
    {"name": "nightly-eval", "interval_minutes": 1440, "cmd": ["eval", "--suite", "basic"]},
    {"name": "provider-sweep", "interval_minutes": 360, "cmd": ["loadtest", "--provider", "groq", "--requests", "4"]}
  ]

Usage:
  python -m loomweaver cron --list
  python -m loomweaver cron --run nightly-eval        # fire one job now
  python -m loomweaver cron --daemon                  # run the scheduler loop (Ctrl+C to stop)

State (last-run timestamps, results) lives in cron_state.json next to cron_jobs.json.
"""
import json
import os
import subprocess
import sys
import time

from . import security

HERE = os.path.dirname(os.path.abspath(__file__))  # the loomweaver package dir itself
JOBS_FILE = os.path.join(HERE, "cron_jobs.json")
STATE_FILE = os.path.join(HERE, "cron_state.json")

DEFAULT_JOBS = [
    {"name": "nightly-eval", "interval_minutes": 1440,
     "cmd": ["eval", "--suite", "basic"]},
    {"name": "reasoning-eval", "interval_minutes": 1440,
     "cmd": ["eval", "--suite", "reasoning"]},
    {"name": "groq-loadtest", "interval_minutes": 720,
     "cmd": ["loadtest", "--provider", "groq", "--concurrency", "4", "--requests", "6"]},
]


def _jobs():
    if not os.path.exists(JOBS_FILE):
        with open(JOBS_FILE, "w") as f:
            json.dump(DEFAULT_JOBS, f, indent=2)
        return DEFAULT_JOBS
    return json.load(open(JOBS_FILE))


def _state():
    if os.path.exists(STATE_FILE):
        return json.load(open(STATE_FILE))
    return {}


def _save_state(s):
    with open(STATE_FILE, "w") as f:
        json.dump(s, f, indent=2)


def run_job(name):
    job = next((j for j in _jobs() if j["name"] == name), None)
    if not job:
        return {"ok": False, "error": f"job '{name}' not found"}
    ok, reason = security.check_cron_cmd(job["cmd"])
    if not ok:
        return {"ok": False, "error": f"cron job rejected: {reason}"}
    t0 = time.time()
    r = subprocess.run([sys.executable, "-m", "src.loomweaver", *job["cmd"]],
                       capture_output=True, text=True, timeout=1800)
    out = (r.stdout + r.stderr).strip()
    # persist last result summary into state
    state = _state()
    state[name] = {"last_run": time.strftime("%Y-%m-%d %H:%M:%S"),
                   "exit_code": r.returncode,
                   "duration_s": round(time.time() - t0, 1),
                   "output_tail": out[-800:]}
    _save_state(state)
    return {"ok": r.returncode == 0, "job": name, "exit_code": r.returncode,
            "duration_s": state[name]["duration_s"], "output_tail": out[-400:]}


def daemon(poll_seconds=60):
    """Main loop: every poll, run any job whose interval has elapsed."""
    print(f"loomweaver cron daemon started ({len(_jobs())} jobs) — Ctrl+C to stop")
    while True:
        state = _state()
        for job in _jobs():
            name = job["name"]
            last = state.get(name, {}).get("_epoch", 0)
            if time.time() - last >= job["interval_minutes"] * 60:
                print(f"[{time.strftime('%H:%M:%S')}] firing {name}")
                res = run_job(name)
                state = _state()
                state[name]["_epoch"] = time.time()
                _save_state(state)
                print(f"  -> {'ok' if res['ok'] else 'FAILED'} in {res.get('duration_s')}s")
        time.sleep(poll_seconds)


def cli(args):
    if args.list or (not args.run and not args.daemon):
        for j in _jobs():
            st = _state().get(j["name"], {})
            last = st.get("last_run", "never")
            code = st.get("exit_code", "-")
            print(f"{j['name']:20} every {j['interval_minutes']}min  cmd={j['cmd']}  last={last} exit={code}")
        return
    if args.run:
        import json as _json
        print(_json.dumps(run_job(args.run), indent=2))
    elif args.daemon:
        try:
            daemon()
        except KeyboardInterrupt:
            print("\nstopped")

"""cli.py — `python -m loomweaver <command>`"""
import argparse
import json
import os

from . import __version__, agent, evals, loadtest
from .core import build_providers, load_creds

# Provider key well-formedness spec for the `doctor` command. Prefix + minimum
# body length; a configured key that matches neither prefix nor length is FAIL.
_DR_PROVIDER_KEY_SPEC = {
    "openrouter": ("sk-or-", 8),
    "freeinference": ("", 0),      # no public prefix; any non-empty key is fine
    "cloudflare": ("cfut_", 1),
    "nvidia": ("nvapi-", 1),
    "groq": ("gsk_", 1),
}
_DR_PROVIDER_ENV = {
    "openrouter": "OPENROUTER_KEY",
    "freeinference": "FREEINFERENCE_KEY",
    "cloudflare": "CLOUDFLARE_TOKEN",
    "nvidia": "NVIDIA_KEY",
    "groq": "GROQ_KEY",
}


def _db_writable(path):
    """True if `path` (or its parent dir) is writable. Never raises."""
    try:
        if not path:
            return False
        d = os.path.dirname(os.path.abspath(path or "."))
        os.makedirs(d, exist_ok=True)
        probe = os.path.join(d, ".loomweaver_doctor_probe")
        with open(probe, "w", encoding="utf-8") as f:
            f.write("")
        os.unlink(probe)
        return True
    except Exception:
        return False


def doctor(creds=None, env=None):
    """Validate configuration WITHOUT making any network calls.

    Returns a list of {"check", "status", "detail"} dicts. status is one of
    OK / WARN / FAIL. Never raises and never, ever phones out — safe to run
    with zero keys configured (each key check then reports WARN).
    """
    e = env if env is not None else os.environ
    creds = creds if creds is not None else load_creds()
    results = []

    # 1) provider keys: present-but-well-formed per provider
    for prov, envvar in _DR_PROVIDER_ENV.items():
        raw = creds.get(envvar) or e.get(envvar) or ""
        if not raw.strip():
            results.append({
                "check": f"provider.{prov}.key",
                "status": "WARN",
                "detail": f"{envvar} not set — {prov} will be skipped",
            })
            continue
        prefix, min_body = _DR_PROVIDER_KEY_SPEC[prov]
        key = raw.strip().split(",")[0].strip()
        body = key[len(prefix):] if prefix else key
        if prefix and not key.startswith(prefix):
            results.append({
                "check": f"provider.{prov}.key",
                "status": "FAIL",
                "detail": f"{envvar} does not match expected prefix '{prefix}...'",
            })
        elif len(body) < min_body:
            results.append({
                "check": f"provider.{prov}.key",
                "status": "FAIL",
                "detail": f"{envvar} key body too short (< {min_body} chars)",
            })
        else:
            results.append({
                "check": f"provider.{prov}.key",
                "status": "OK",
                "detail": f"{envvar} present and well-formed",
            })

    # cloudflare also needs an account id to be usable
    if (creds.get("CLOUDFLARE_ACCOUNT_ID") or e.get("CLOUDFLARE_ACCOUNT_ID")):
        results.append({"check": "provider.cloudflare.account_id",
                        "status": "OK", "detail": "CLOUDFLARE_ACCOUNT_ID set"})
    else:
        results.append({"check": "provider.cloudflare.account_id",
                        "status": "WARN", "detail": "CLOUDFLARE_ACCOUNT_ID not set"})

    # 2) DB paths (quota / cache / usage) writable
    for label, envvar, default in (
        ("db.quota", "LOOMWEAVER_QUOTA_DB",
         os.path.join(os.path.dirname(__file__), "..", "..", "runs", "quota_ledger.db")),
        ("db.cache", "LOOMWEAVER_CACHE_DB",
         os.path.join(os.path.dirname(__file__), "..", "..", "runs", "semantic_cache.sqlite3")),
        ("db.usage", "LOOMWEAVER_USAGE_DB",
         os.path.join(os.path.dirname(__file__), "..", "..", "runs", "usage.db")),
    ):
        path = e.get(envvar) or default
        if _db_writable(path):
            results.append({"check": label, "status": "OK",
                            "detail": f"writable ({path})"})
        else:
            results.append({"check": label, "status": "FAIL",
                            "detail": f"not writable ({path})"})
    return results


def _print_doctor_results(results):
    for r in results:
        flag = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[r["status"]]
        print(f"{flag} {r['check']}: {r['detail']}")


def main(argv=None):
    # Make the decoy-credential layer reachable at runtime: the CLI is the
    # real entrypoint, so install (idempotent, never-raising) on any command.
    try:
        from . import observability
        observability.ensure_decoys()
    except Exception:
        pass  # the decoy layer must never crash the CLI
    ap = argparse.ArgumentParser(prog="harness", description=f"complete harness v{__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # agent
    p = sub.add_parser("agent", help="run the agent on a goal")
    p.add_argument("goal")
    p.add_argument("--session", default="default")
    p.add_argument("--model")
    p.add_argument("--max-steps", type=int, default=10)

    # eval
    p = sub.add_parser("eval", help="run an eval suite")
    p.add_argument("--suite", default="basic",
                   choices=["basic", "reasoning", "extraction", "tools", "agent"])
    p.add_argument("--model")

    # eval-compare
    p = sub.add_parser("eval-compare", help="run suites across models")

    # loadtest
    p = sub.add_parser("loadtest", help="load-test a provider")
    p.add_argument("--provider")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--requests", type=int, default=8)

    # armada
    p = sub.add_parser("armada", help="launch a named agent fleet on a mission")
    p.add_argument("mission")
    p.add_argument("--pipeline", default="standard", choices=["standard"])
    p.add_argument("--max-steps", type=int, default=12, help="steps per agent")

    # providers
    p = sub.add_parser("providers", help="list configured providers/models")

    # doctor / check-config
    p = sub.add_parser("doctor", aliases=["check-config"],
                       help="validate config (keys + writable DB paths); no network calls")

    # cron
    p = sub.add_parser("cron", help="scheduled jobs (local, opt-in)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--run", metavar="JOB")
    p.add_argument("--daemon", action="store_true")

    # ttft
    p = sub.add_parser("ttft", help="streaming TTFT sweep across providers")

    # usage
    p = sub.add_parser("usage", help="per-provider usage dashboard")
    p.add_argument("--hours", type=int, default=24)
    p.add_argument("--json", action="store_true", help="emit JSON instead of a table")

    # quota
    p = sub.add_parser("quota", help="per-provider free-tier quota status")

    args = ap.parse_args(argv)

    if args.cmd == "providers":
        for p_ in build_providers(load_creds()):
            print(f"{p_['name']:14} {p_['cost']:5} models: {', '.join(p_['models'])}")
    elif args.cmd in ("doctor", "check-config"):
        _print_doctor_results(doctor())
    elif args.cmd == "agent":
        out = agent.run(args.goal, session_id=args.session, model=args.model,
                        max_steps=args.max_steps)
        print(json.dumps({"result": out["result"], "run_dir": out["run_dir"]}, indent=2))
    elif args.cmd == "eval":
        if args.suite == "agent":
            print(json.dumps(evals.run_agent_suite(model=args.model), indent=2))
        else:
            print(json.dumps(evals.run_suite(args.suite, model=args.model), indent=2))
    elif args.cmd == "eval-compare":
        print(json.dumps(evals.compare(), indent=2))
    elif args.cmd == "loadtest":
        print(json.dumps(loadtest.run(provider=args.provider,
                                      concurrency=args.concurrency,
                                      requests=args.requests), indent=2))
    elif args.cmd == "cron":
        from . import cron
        cron.cli(args)
    elif args.cmd == "ttft":
        print(json.dumps(loadtest.ttft_sweep(), indent=2))
    elif args.cmd == "usage":
        from . import usage as _u
        s = _u.summary(hours=args.hours)
        print(json.dumps(_u.render_json(s), indent=2) if args.json
              else _u.render_text(s))
    elif args.cmd == "quota":
        from .quota_ledger import get_quota_status
        print(json.dumps(get_quota_status(), indent=2))
    elif args.cmd == "armada":
        from .armada import Armada
        fleet = Armada(args.mission).standard_pipeline()
        result = fleet.execute(creds=load_creds(), max_steps_per_agent=args.max_steps)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

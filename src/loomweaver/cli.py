"""cli.py — `python -m loomweaver <command>`"""
import argparse
import json
import os
import sys

from . import __version__, agent, evals, learning, loadtest
from .core import build_providers, load_creds

try:
    from flippy_providers import describe_catalog
except ImportError:  # pragma: no cover - path fallback for direct invocation
    from ..flippy_providers import describe_catalog

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

    # 1) provider keys: validate every provider the REGISTRY actually resolved.
    #
    # This used to iterate five hardcoded brand variables and emit a WARN for
    # each one you had not set — which is noise the moment the provider surface
    # is open-ended, since almost nobody configures all of them. It now checks
    # what you configured, and validates key shape wherever the format is known.
    try:
        from flippy_providers import get_providers
        configured = get_providers({**e, **creds})
    except Exception:
        configured = []

    if not configured:
        results.append({
            "check": "providers.configured",
            "status": "WARN",
            "detail": "no providers resolved — set any provider key, or a "
                      "<PREFIX>_API_KEY + <PREFIX>_BASE_URL pair "
                      "(see `providers --all`)",
        })
    for prov in configured:
        name = prov["name"]
        envvar = prov.get("env_key") or ""
        raw = prov.get("key") or ""
        n_keys = len(prov.get("keys") or [])
        spec = _DR_PROVIDER_KEY_SPEC.get(name)
        if spec:
            prefix, min_body = spec
            body = raw[len(prefix):] if prefix else raw
            if prefix and not raw.startswith(prefix):
                results.append({
                    "check": f"provider.{name}.key", "status": "FAIL",
                    "detail": f"{envvar} does not match expected prefix '{prefix}...'",
                })
                continue
            if len(body) < min_body:
                results.append({
                    "check": f"provider.{name}.key", "status": "FAIL",
                    "detail": f"{envvar} key body too short (< {min_body} chars)",
                })
                continue
        results.append({
            "check": f"provider.{name}.key", "status": "OK",
            "detail": (f"{envvar or 'keyless'} present"
                       + (f", {n_keys} key(s) rotating" if n_keys > 1 else "")
                       + f", {len(prov.get('models') or [])} model(s)"),
        })

    # Cloudflare embeds the account id in the URL, so its token alone is not
    # enough. Only worth reporting when Cloudflare is actually configured —
    # warning about it otherwise is noise for everyone else.
    if any(p["name"] == "cloudflare" for p in configured):
        results.append({"check": "provider.cloudflare.account_id",
                        "status": "OK", "detail": "CLOUDFLARE_ACCOUNT_ID set"})
    elif creds.get("CLOUDFLARE_TOKEN") or e.get("CLOUDFLARE_TOKEN"):
        results.append({"check": "provider.cloudflare.account_id",
                        "status": "FAIL",
                        "detail": "CLOUDFLARE_TOKEN set but CLOUDFLARE_ACCOUNT_ID "
                                  "missing — cloudflare will be skipped"})

    # 2) DB paths (quota / cache / usage) writable
    for label, envvar, default in (
        ("db.quota", "LOOMWEAVER_QUOTA_DB",
         os.path.join(os.path.dirname(__file__), "..", "..", "runs", "quota_ledger.db")),
        ("db.cache", "LOOMWEAVER_CACHE_DB",
         os.path.join(os.path.dirname(__file__), "..", "..", "runs", "semantic_cache.sqlite3")),
        ("db.usage", "LOOMWEAVER_USAGE_DB",
         os.path.join(os.path.dirname(__file__), "..", "..", "runs", "usage.db")),
        ("db.key_rotation", "LOOMWEAVER_KEYROTATION_DB",
         os.path.join(os.path.dirname(__file__), "..", "..", "runs",
                      "key_rotation.db")),
        ("db.learning", "LOOMWEAVER_LEARNING_DB",
         os.path.join(os.path.dirname(__file__), "..", "..", "runs",
                      "learning.db")),
    ):
        path = e.get(envvar) or default
        if _db_writable(path):
            results.append({"check": label, "status": "OK",
                            "detail": f"writable ({path})"})
        else:
            results.append({"check": label, "status": "FAIL",
                            "detail": f"not writable ({path})"})
    return results


def _resolve_tool_scope(raw, allow_empty=False):
    """Parse --tools into an explicit scope, refusing unknown tool names.

    A typo used to fall through to goal-based auto-grant, which silently handed
    the run a *wider* toolset than the operator asked for. Unknown names are a
    hard error now.
    """
    if not raw:
        return None if allow_empty else "auto"
    from . import tools as _tools
    asked = [t.strip() for t in raw.split(",") if t.strip()]
    unknown = [t for t in asked if t not in _tools.TOOLS]
    if unknown:
        raise SystemExit(
            f"error: unknown tool(s) in --tools: {', '.join(unknown)}\n"
            f"available: {', '.join(sorted(_tools.TOOLS))}")
    return asked


def _print_doctor_results(results):
    for r in results:
        flag = {"OK": "[ OK ]", "WARN": "[WARN]", "FAIL": "[FAIL]"}[r["status"]]
        print(f"{flag} {r['check']}: {r['detail']}")


def main(argv=None):
    # Make the decoy-credential layer reachable at runtime: the CLI is the
    # real entrypoint, so install (idempotent, never-raising) on any command.
    try:
        from . import observability, sentinel
        planted = observability.ensure_decoys()
        # A file that looks like it leaked an inline master key, sitting beside
        # the other managed config. Reading it is recorded and both keys are
        # per-install canaries.
        if planted and sentinel.enabled():
            base = os.path.dirname(planted[0])
            try:
                os.makedirs(base, exist_ok=True)
                with open(os.path.join(base, sentinel.BREADCRUMB_NAME),
                          "w", encoding="utf-8") as f:
                    f.write(sentinel.breadcrumb_source())
            except OSError:
                pass
    except Exception:
        pass  # managed-config install must never crash the CLI
    ap = argparse.ArgumentParser(prog="harness", description=f"complete harness v{__version__}")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # agent
    p = sub.add_parser("agent", help="run the agent on a goal")
    p.add_argument("goal")
    p.add_argument("--session", default="default")
    p.add_argument("--model")
    p.add_argument("--max-steps", type=int, default=10)
    p.add_argument("--tools", default="",
                   help="operator-authorized tool set (comma list, e.g. shell,read_file,http_get). "
                        "Pre-flight: the model sees ONLY these (plus the safe read floor). "
                        "Empty = auto-detect from the goal (context determines tools).")

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
    p.add_argument("--tools", default="",
                   help="operator-pre-authorized tools (comma list); roles only get "
                        "tools in this set allowed (intersection with their role toolset).")

    # providers
    p = sub.add_parser("providers", help="list configured providers/models")
    p.add_argument("--all", action="store_true",
                   help="show every provider flippy can talk to and how to enable it")

    # memory: the self-learning layer
    p = sub.add_parser("profile", help="show what flippy has learned about you")
    p.add_argument("--json", action="store_true", help="machine-readable output")

    p = sub.add_parser("learn", help="teach flippy a rule it should apply to similar goals")
    p.add_argument("lesson", help="the rule, e.g. 'when I say deploy, run the staging script first'")
    p.add_argument("--when", default="", help="the kind of goal it applies to (default: the rule itself)")

    p = sub.add_parser("forget", help="erase learned lessons")
    p.add_argument("--id", type=int, default=None, help="forget one lesson by id")
    p.add_argument("--all", action="store_true", help="forget every lesson")

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
        if getattr(args, "all", False):
            print("Every provider flippy speaks (OpenAI chat/completions wire format).")
            print("Set the variable in column 3 to switch one on — any of them.\n")
            print(f"{'provider':16} {'cost':6} {'activate with':52} models override")
            print("-" * 104)
            for row in describe_catalog():
                print(f"{row['name']:16} {row['cost']:6} {row['activate_with']:52} "
                      f"{row['models_env'] or '-'}")
            print("\nPlus: any <PREFIX>_API_KEY + <PREFIX>_BASE_URL pair registers itself.")
            return 0
        configured = build_providers(load_creds())
        if not configured:
            print("No providers configured. Run `flippy providers --all` to see the options.")
            return 1
        for p_ in configured:
            print(f"{p_['name']:14} {p_['cost']:5} models: {', '.join(p_['models'])}")
        print(f"\n{len(configured)} configured — add any other OpenAI-compatible endpoint "
              f"with <PREFIX>_API_KEY + <PREFIX>_BASE_URL.")
    elif args.cmd == "profile":
        store = learning.get_store()
        if getattr(args, "json", False):
            print(json.dumps(store.profile(), indent=2, default=str))
        else:
            print(learning.render_text(store.profile()))
            st = store.stats()
            print(f"  store               {st['interactions']} interactions, "
                  f"{st['lessons']} lessons\n  db                  "
                  f"{os.path.normpath(store.db_path)}")
    elif args.cmd == "learn":
        lid = learning.get_store().add_lesson(args.lesson, trigger=args.when)
        print(f"learned (lesson {lid}): {args.lesson}")
        print("It will be injected into the context of goals that look like this one.")
    elif args.cmd == "forget":
        if not getattr(args, "all", False) and args.id is None:
            print("usage: flippy forget --all   |   flippy forget --id N", file=sys.stderr)
            return 2
        learning.get_store().forget(args.id)
        print("forgot lesson " + str(args.id) if args.id is not None else "forgot every lesson")
    elif args.cmd in ("doctor", "check-config"):
        _print_doctor_results(doctor())
    elif args.cmd == "agent":
        tool_scope = _resolve_tool_scope(args.tools)
        out = agent.run(args.goal, session_id=args.session, model=args.model,
                        max_steps=args.max_steps, tool_scope=tool_scope)
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
        tool_scope = _resolve_tool_scope(args.tools, allow_empty=True)
        fleet = Armada(args.mission, tool_scope=tool_scope).standard_pipeline()
        result = fleet.execute(creds=load_creds(), max_steps_per_agent=args.max_steps)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

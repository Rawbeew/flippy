"""cli.py — `python -m loomweaver <command>`"""
import argparse
import json
import sys

from . import __version__, agent, evals, loadtest
from .core import build_providers, load_creds


def main(argv=None):
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
                   choices=["basic", "reasoning", "extraction", "tools"])
    p.add_argument("--model")

    # eval-compare
    p = sub.add_parser("eval-compare", help="run suites across models")

    # loadtest
    p = sub.add_parser("loadtest", help="load-test a provider")
    p.add_argument("--provider")
    p.add_argument("--concurrency", type=int, default=4)
    p.add_argument("--requests", type=int, default=8)

    # providers
    p = sub.add_parser("providers", help="list configured providers/models")

    # cron
    p = sub.add_parser("cron", help="scheduled jobs (local, opt-in)")
    p.add_argument("--list", action="store_true")
    p.add_argument("--run", metavar="JOB")
    p.add_argument("--daemon", action="store_true")

    # ttft
    p = sub.add_parser("ttft", help="streaming TTFT sweep across providers")

    args = ap.parse_args(argv)

    if args.cmd == "providers":
        for p_ in build_providers(load_creds()):
            print(f"{p_['name']:14} {p_['cost']:5} models: {', '.join(p_['models'])}")
    elif args.cmd == "agent":
        out = agent.run(args.goal, session_id=args.session, model=args.model,
                        max_steps=args.max_steps)
        print(json.dumps({"result": out["result"], "run_dir": out["run_dir"]}, indent=2))
    elif args.cmd == "eval":
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


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""run_bench.py — reproducible router benchmark for flippy providers.

For every configured provider, fires N small chat completions at the
provider's first model and records:
  - latency p50 / p95 (seconds)
  - success rate
  - tokens/sec (completion tokens / generation time)

Methodology notes:
  - Prompts are tiny (~10 tokens) to stay comfortably inside free tiers.
  - Keys are read from environment variables (optionally loaded from a
    local credentials file passed as --env). They are NEVER printed,
    logged, or written into results.
  - All HTTP requests use flippy's standardized User-Agent.
  - Results go to benchmarks/results-YYYY-MM-DD.json and RESULTS.md.

Usage:
    python benchmarks/run_bench.py --requests 20 [--timeout 30]
        [--env path/to/credentials.env] [--out-dir benchmarks]

Stdlib only — no dependencies beyond Python itself.
"""
import argparse
import datetime
import json
import os
import statistics
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from flippy_providers import get_providers, UA  # noqa: E402

PROMPT = "Reply with the single word: ok"
MAX_TOKENS = 8


def load_env_file(path):
    """Read KEY=VALUE lines into os.environ (values never echoed)."""
    if not path or not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def one_request(provider):
    """Single timed request. Returns (ok, total_s, gen_s, completion_tokens, err)."""
    payload = json.dumps({
        "model": provider["models"][0],
        "messages": [{"role": "user", "content": PROMPT}],
        "max_tokens": MAX_TOKENS,
    }).encode()
    req = urllib.request.Request(
        provider["url"], data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {provider['key']}",
            "User-Agent": UA,
        },
        method="POST",
    )
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=args_timeout) as r:
            body = json.loads(r.read())
        total = time.perf_counter() - t0
        usage = body.get("usage", {})
        ctoks = usage.get("completion_tokens") or 0
        # generation time estimate: fall back to total if no timing fields
        gen = total
        return True, total, gen, ctoks, None
    except urllib.error.HTTPError as e:
        e.read()  # drain
        return False, time.perf_counter() - t0, 0.0, 0, f"HTTP {e.code}"
    except Exception as e:  # noqa: BLE001
        return False, time.perf_counter() - t0, 0.0, 0, str(e)[:120]


args_timeout = 30  # set by main()


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))
    return s[idx]


def bench_provider(provider, n):
    lat, oks, tps, errs = [], 0, [], {}
    for i in range(n):
        ok, total, gen, ctoks, err = one_request(provider)
        lat.append(total)
        if ok:
            oks += 1
            if ctoks > 0 and gen > 0:
                tps.append(ctoks / gen)
        elif err:
            errs[err] = errs.get(err, 0) + 1
        time.sleep(1.0)  # be polite to free tiers
    return {
        "provider": provider["name"],
        "model": provider["models"][0],
        "requests": n,
        "successes": oks,
        "success_rate": round(100.0 * oks / n, 1),
        "latency_p50_s": round(pct(lat, 50), 3),
        "latency_p95_s": round(pct(lat, 95), 3),
        "latency_mean_s": round(statistics.mean(lat), 3),
        "tokens_per_sec_mean": round(statistics.mean(tps), 2) if tps else None,
        "errors": errs,
    }


def write_markdown(results, out_dir, datestr):
    lines = [
        "# flippy Router Benchmark Results",
        "",
        f"Date: {datestr} · {len(results)} provider(s) · "
        f"{results[0]['requests'] if results else 0} requests/provider",
        "",
        "| Provider | Model | Success | Latency p50 (s) | Latency p95 (s) | Tokens/s |",
        "|---|---|---|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['provider']} | `{r['model']}` | {r['success_rate']}% "
            f"({r['successes']}/{r['requests']}) | {r['latency_p50_s']} | "
            f"{r['latency_p95_s']} | {r['tokens_per_sec_mean'] or 'n/a'} |"
        )
    lines += ["", "See README.md for methodology. Raw data: results-%s.json" % datestr, ""]
    with open(os.path.join(out_dir, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def main():
    global args_timeout
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=20)
    ap.add_argument("--timeout", type=int, default=30)
    ap.add_argument("--env", default=None, help="credentials file to source")
    ap.add_argument("--out-dir", default=os.path.dirname(os.path.abspath(__file__)))
    args = ap.parse_args()
    args_timeout = args.timeout

    load_env_file(args.env)
    providers = get_providers()
    if not providers:
        print("No providers configured (no *_KEY env vars found). Nothing to benchmark.")
        sys.exit(1)

    datestr = datetime.date.today().isoformat()
    results = []
    for p in providers:
        print(f"Benchmarking {p['name']} ({p['models'][0]}) x{args.requests} ...")
        r = bench_provider(p, args.requests)
        results.append(r)
        print(f"  -> success={r['success_rate']}%  p50={r['latency_p50_s']}s  "
              f"p95={r['latency_p95_s']}s")

    out_json = os.path.join(args.out_dir, f"results-{datestr}.json")
    meta = {
        "date": datestr,
        "requests_per_provider": args.requests,
        "prompt": PROMPT,
        "max_tokens": MAX_TOKENS,
        "note": "Keys are sourced from env; no secrets appear anywhere in this file.",
        "results": results,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    write_markdown(results, args.out_dir, datestr)
    print(f"Wrote {out_json} and RESULTS.md")


if __name__ == "__main__":
    main()

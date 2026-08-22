#!/usr/bin/env python3
"""benchmark_hedged.py — sequential vs hedged-request router benchmark.

Fires N small chat completions through loomweaver's router two ways:
  - sequential route(): failover provider-by-provider
  - hedged route_hedged(): top 2 providers raced, first success wins

Records p50 / p95 latency and win distribution, then writes HEDGED.md.

Methodology notes:
  - Tiny prompts (~10 tokens) to stay inside free tiers.
  - Keys come from the local credentials file; NEVER printed or logged.
  - Uses loomweaver's standardized User-Agent.
  - QUOTA WARNING: hedging fires a second request after delay_ms whenever the
    first provider hasn't answered yet, so slow tails burn 2x quota. See HEDGED.md.

Usage:
    python benchmarks/benchmark_hedged.py --requests 10 [--delay-ms 250]

Stdlib only.
"""
import argparse
import datetime
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from loomweaver.core import load_creds, route, route_hedged  # noqa: E402

PROMPT = "Reply with the single word: ok"
MESSAGES = [{"role": "user", "content": PROMPT}]
MAX_TOKENS = 8


def pctl(xs, p):
    xs = sorted(xs)
    if not xs:
        return float("nan")
    i = max(0, min(len(xs) - 1, round(p / 100 * (len(xs) - 1))))
    return xs[i]


def run_mode(fn, n):
    lat, wins, errs = [], {}, 0
    for _ in range(n):
        r = fn(MESSAGES)
        if r.get("ok"):
            lat.append(r.get("latency", 0.0))
            w = r.get("provider", "?")
            wins[w] = wins.get(w, 0) + 1
        else:
            errs += 1
    return lat, wins, errs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--requests", type=int, default=10)
    ap.add_argument("--delay-ms", type=int, default=250)
    args = ap.parse_args()

    creds = load_creds()
    seq_lat, seq_wins, seq_err = run_mode(
        lambda m: route(m, max_tokens=MAX_TOKENS, creds=creds), args.requests)
    hed_lat, hed_wins, hed_err = run_mode(
        lambda m: route_hedged(m, max_tokens=MAX_TOKENS, creds=creds,
                               delay_ms=args.delay_ms), args.requests)

    def block(name, lat, wins, errs):
        if lat:
            return (f"| {name} | {len(lat)}/{args.requests} | "
                    f"{pctl(lat, 50):.3f} | {pctl(lat, 95):.3f} | "
                    f"{statistics.mean(lat):.3f} | {errs} |")
        return f"| {name} | 0/{args.requests} | - | - | - | {errs} |"

    today = datetime.date.today().isoformat()
    md = f"""# Hedged-request router benchmark — {today}

`route()` (sequential failover) vs `route_hedged()` (top-2 providers raced,
second fired after {args.delay_ms}ms; first success wins, loser abandoned).

| Mode | Success | p50 (s) | p95 (s) | mean (s) | errors |
|------|---------|---------|---------|----------|--------|
{block("sequential", seq_lat, seq_wins, seq_err)}
{block("hedged", hed_lat, hed_wins, hed_err)}

Win distribution (hedged): {json.dumps(hed_wins)}
Win distribution (sequential): {json.dumps(seq_wins)}

## Honest notes

- **Quota:** hedging is not free. Whenever the first provider hasn't answered
  within `delay_ms`, a second request fires — on a slow tail **both complete**
  and you burn 2x quota on that request. With free-tier rate limits this can
  meaningfully shorten your daily budget; tune `delay_ms` upward if quota
  matters more than tail latency.
- **Best-case win:** hedging only helps when the first provider is slow or
  failing. If provider #1 is consistently fast, hedged ≈ sequential latency
  but you still pay the extra request whenever the first attempt exceeds
  `delay_ms`.
- **Cancellation is best-effort:** the losing request is abandoned, not
  revoked — the provider still processes and bills/counts it.
- Sample size is {args.requests} requests — enough to see the pattern, not a
  rigorous tail-latency study.
"""
    out = os.path.join(os.path.dirname(__file__), "HEDGED.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write(md)
    print(md)


if __name__ == "__main__":
    main()

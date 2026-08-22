# Hedged-request router benchmark — 2026-08-21

`route()` (sequential failover) vs `route_hedged()` (top-2 providers raced,
second fired after 250ms; first success wins, loser abandoned).

| Mode | Success | p50 (s) | p95 (s) | mean (s) | errors |
|------|---------|---------|---------|----------|--------|
| sequential | 10/10 | 1.208 | 1.635 | 1.273 | 0 |
| hedged | 10/10 | 1.199 | 4.360 | 1.469 | 0 |

Win distribution (hedged): {"openrouter": 10}
Win distribution (sequential): {"openrouter": 10}

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
- Sample size is 10 requests — enough to see the pattern, not a
  rigorous tail-latency study.

## Interpretation of this run

On this particular run hedging did **not** improve tail latency: p50 was
essentially flat (1.199s vs 1.208s) and p95 was **worse** (4.360s vs 1.635s).
All 10 requests in both modes were won by `openrouter`, so the hedge never
produced a win for provider #2 — the observed p95 regression is consistent
with (a) the shadow request adding load/queueing pressure on the same host,
and (b) small-sample noise at n=10. The unit tests demonstrate the intended
behavior deterministically with mocked providers; real-world benefit depends
on provider #2 actually being faster than provider #1's tail.

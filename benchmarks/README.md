# flippy Router Benchmarks

Reproducible latency / reliability benchmarks for flippy's multi-provider router.

## What is measured

For every provider configured via environment variables (`OPENROUTER_KEY`,
`FREEINFERENCE_KEY`, `NVIDIA_KEY`, `GROQ_KEY`, ...), the script fires **20
small chat completions** at the provider's first model and records:

| Metric | Meaning |
|---|---|
| Success rate | % of requests returning HTTP 200 with valid JSON |
| Latency p50 / p95 | Time-to-complete-response, seconds (all attempts counted) |
| Tokens/sec | Completion tokens ÷ generation time, averaged over successes |

## Methodology

- Prompt: `"Reply with the single word: ok"`, `max_tokens=8` — tiny on purpose
  so the whole run stays comfortably inside free-tier quotas.
- 1 second sleep between requests per provider (politeness / burst limits).
- HTTP via Python stdlib `urllib`, using flippy's standardized User-Agent.
- Keys come from env vars only — optionally sourced from a local credentials
  file via `--env`. Secrets are never printed, logged, or committed.
- Results land in `results-YYYY-MM-DD.json` and `RESULTS.md`.

## Reproduce

```bash
export OPENROUTER_KEY=...   # any subset of supported keys works
python benchmarks/run_bench.py --requests 20
```

Or source from a local file:

```bash
python benchmarks/run_bench.py --requests 20 --env ~/secrets/credentials.env
```

## Interpreting results

Free providers are shared capacity: day-to-day variance is high. Compare
providers within one run, not across runs. A provider failing here may simply
be rate-limited at that moment — exactly the situation flippy's failover is
built for.

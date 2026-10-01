# Changelog

## 2026-09-30 — Adaptive routing + agent loop hardening

### Router (production-shaping)

- **Adaptive provider ordering** (`router_policy.py`): providers are no longer
  tried in static registry order. An EWMA of success rate divided by an EWMA
  of latency scores each provider; a provider that just failed enters a short
  local cooldown (5s doubling to 60s cap) and sinks below healthy ones.
  Registry order remains the tie-break so cold-start behavior is unchanged.
- **In-provider retry with backoff**: retryable failures (5xx, timeouts) get
  up to 3 attempts on the same provider before failover, with exponential
  backoff + jitter (capped 30s, `Retry-After` respected). 429s are NEVER
  retried in-provider — the quota ledger already owns 429 cooldowns, and
  retrying only burns quota. Measured: a transient 500 that succeeds on the
  second attempt now completes without a failover hop.
- **Deadline budgets**: `route(timeout_budget_s=...)` stops walking providers
  (and stops backing off) once the budget is spent.
- **Attempt trails**: every provider attempt is emitted as an `llm_call`
  event with `attempt`, `status`, `latency`; retries emit `retry_wait`
  events with the backoff taken. The run log now shows the full decision
  path.
- **Ledger cooldown propagation**: quota-ledger skips now mirror into the
  adaptive policy, so a provider in a long 429 cooldown sinks in the order
  without an extra DB read.

### Agent loop (measurably smarter)

- **String-aware JSON protocol parser**: replaced the greedy-regex action
  parser with a balanced-brace scanner that tracks quoted strings and
  escapes. Benchmark on realistic free-tier outputs: old parser 10/12,
  new parser 12/12. Recovers actions wrapped in prose, markdown fences,
  brace-containing string values, and brace noise in surrounding text.
- **Budgeted observations**: tool results re-entering context are capped at
  2000 chars with an explicit `[truncated, N more chars]` marker (was a
  silent 1500-char slice).
- **No-progress loop detection**: 2 consecutive steps with no tool call and
  no done terminate the run with a `no_progress` reason instead of burning
  max_steps. A tool call resets the counter.
- **Agent benchmark suite** (`eval --suite agent`): 4 multi-step tasks
  scored on required-tools-called + genuine completion (event-trail
  `run_done`, not max-steps exhaustion). The suite detects looping models
  deterministically — a never-done planner scores 0/4.
- 26 new tests; suite now **173 passing** (was 147).

## Earlier entries

See git history. Highlights: quota ledger + backoff (63ff6ba), key rotation,
semantic cache, usage dashboard, security hardening (path jail, SSRF guard,
shell blocklist, cron allowlist), failure-injection suite, hedged requests.

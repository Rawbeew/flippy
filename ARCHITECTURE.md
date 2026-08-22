# flippy — Architecture

## Overview

flippy is a **free-first multi-provider LLM inference system** with three layers:
a routing core, an agent harness (Loomweaver), and a multimodal hub. Everything
is stdlib-only except the optional litellm-based AI hub.

```
┌─────────────────────────────────────────────────────────────┐
│                        CONSUMERS                            │
│   CLI (ai_failover.py)  │  HTTP (server.py)  │  Agents      │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│                    ROUTING CORE (core.py)                    │
│                                                              │
│  1. semantic_cache.lookup()     ← hit? return immediately   │
│  2. quota_ledger.check_quota()  ← exhausted? skip provider  │
│  3. key_rotation.next_key()     ← dead key? rotate          │
│  4. chat(provider, messages)    ← the actual LLM call       │
│  5. usage.record()              ← persist metrics           │
│  6. quota_ledger.record_result()← update cooldown state     │
│  7. semantic_cache.store()      ← cache for next time       │
│                                                              │
│  On failure: failover to next provider (free-first order)   │
│  On 429: exponential backoff + cooldown                     │
│  On 401/403: mark key dead, rotate                          │
└────────────┬────────────────────────────────────────────────┘
             │
             ▼
┌─────────────────────────────────────────────────────────────┐
│              PROVIDER REGISTRY (flippy_providers.py)         │
│                                                              │
│  Single source of truth. Consumed by ALL layers.             │
│  Supports comma-separated multi-key env vars.               │
│                                                              │
│  Priority: openrouter → freeinference → cloudflare →        │
│            nvidia → groq                                     │
└─────────────────────────────────────────────────────────────┘
```

## Module map

```
src/
├── flippy_providers.py   ← THE registry (single source of truth)
├── ai_failover.py        ← standalone CLI router (thin wrapper)
├── aihub.py              ← litellm-powered multimodal hub (optional deps)
├── server.py             ← stdlib HTTP server: /v1/chat, /health, /metrics, /usage
└── loomweaver/           ← agent harness package
    ├── core.py           ← routing engine: cache→quota→rotate→call→record
    ├── providers.py      ← (re-export shim for flippy_providers)
    ├── agent.py          ← single-agent loop (plan→act→observe→done)
    ├── armada.py         ← multi-agent fleet: scout→builder→verifier→reporter
    ├── tools.py          ← tool registry with security guards
    ├── security.py       ← SSRF guard, path jail, shell blocklist, env strip
    ├── quota_ledger.py   ← per-provider quota tracking + cooldowns
    ├── semantic_cache.py ← TF-IDF cosine similarity response cache
    ├── usage.py          ← per-provider usage analytics
    ├── key_rotation.py   ← multi-key rotation state machine
    ├── stream.py         ← SSE streaming with TTFT measurement
    ├── evals.py          ← eval suites: basic, reasoning, extraction, tools
    ├── loadtest.py       ← concurrency + TTFT benchmarking
    ├── cron.py           ← local scheduler for evals/loadtests
    └── cli.py            ← `python -m src.loomweaver <command>`
```

## Data flow: a single chat request

```
User prompt
    │
    ▼
┌─ semantic_cache.lookup() ──────────────────────────────────┐
│  Exact hash match? → return cached (0ms)                    │
│  TF-IDF cosine ≥ 0.92? → return cached                      │
│  No hit? → continue to routing                              │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
┌─ For each provider (free-first order) ────────────────────┐
│                                                             │
│  quota_ledger.check_quota(name)                             │
│    ├── exhausted → skip, log quota_skip                    │
│    └── available → continue                                 │
│                                                             │
│  key_rotation.next_key(name)                                │
│    ├── all keys dead → skip                                 │
│    └── live key → use it                                    │
│                                                             │
│  POST {provider.url}                                        │
│    ├── 200 → usage.record() → cache.store() → RETURN       │
│    ├── 429 → quota_ledger.record_result() → next provider  │
│    ├── 401/403 → key_rotation.mark_dead() → retry same     │
│    │            provider with next key                      │
│    └── 5xx/network → next provider                          │
└─────────────────────────────────────────────────────────────┘
    │
    ▼
  All providers exhausted → return error
```

## Agent harness (Loomweaver)

```
Mission
    │
    ▼
┌─ SCOUT ──────────────────────────────────────────────────┐
│  Read-only: http_get, read_file, list_dir, shell (ro)    │
│  Returns: file inventory, current state, constraints     │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌─ BUILDER ────────────────────────────────────────────────┐
│  Read-write: write_file, shell, http_get                 │
│  Implements the mission, runs tests                      │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌─ VERIFIER ───────────────────────────────────────────────┐
│  Read-only: adversarial QA                               │
│  Runs full test suite, checks security guards            │
│  VERDICT: PASS → continue │ VERDICT: FAIL → STOP        │
└────────────────────┬─────────────────────────────────────┘
                     ▼
┌─ REPORTER ───────────────────────────────────────────────┐
│  Read-only: summarizes for humans                        │
│  What changed, why, test status, risks                   │
└──────────────────────────────────────────────────────────┘
```

Tool restrictions are **enforced in dispatch**, not suggested in prompts.
If a Scout tries `write_file`, it receives `"BLOCKED"` in its context.

## Scaling points

| Bottleneck | Current solution | Next step at scale |
|---|---|---|
| Provider rate limits | Quota ledger + cooldowns | Add paid tiers as overflow |
| Latency (p99) | Hedged requests (fire 2, take first) | Streaming-first API |
| Cache misses | TF-IDF similarity (lexical) | Embedding-based semantic cache (requires model) |
| Single-process | stdlib http.server | FastAPI + uvicorn workers |
| Trace storage | JSONL files | SQLite/ClickHouse for query-ability |

## Trade-offs made

| Decision | Chose | Gave up | Why |
|---|---|---|---|
| Dependencies | stdlib only | Rich features from litellm in the core | Zero-install deploy; aihub provides litellm path when needed |
| Cache similarity | TF-IDF cosine | Embedding-based semantic match | No API calls for cache lookup; works offline |
| Agent tools | Denylist + allowlist | Full sandboxing | Simplicity; SAFE_MODE=1 disables shell entirely |
| Multi-key rotation | Sequential with dead-key marking | Concurrent key testing | Predictable; no rate-limit spikes from key probing |
| State persistence | SQLite | Redis/Postgres | Zero-dep; schema ports unchanged |
| Config | Environment variables | Config files | 12-factor; works everywhere without file management |

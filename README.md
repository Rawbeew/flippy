# flippy

A free-tier-first multi-provider LLM router with a complete agent harness.
Stdlib-only core (no dependencies), 317 tests, zero production traffic.

[![CI](https://github.com/Rawbeew/flippy/actions/workflows/ci.yml/badge.svg)](https://github.com/Rawbeew/flippy/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/tests-317%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## What flippy can do

### 1. Route LLM requests across 5 free providers with automatic failover

One OpenAI-compatible request, five providers behind it: OpenRouter,
freeinference.org, Groq, NVIDIA NIM, Cloudflare Workers AI. When one
rate-limits, errors, or hangs, the next picks up mid-request.

```bash
export GROQ_KEY=gsk_...                      # one key is enough to start
python src/ai_failover.py "explain KV caches"
python src/ai_failover.py --model openai/gpt-oss-120b "write a haiku"
python src/ai_failover.py --json "list 3 colors"   # machine-readable output
```

What the router does for you, in order:

- **Adaptive provider ordering** — providers are scored by an EWMA of
  success rate over latency. A provider that just failed sinks below the
  healthy ones; a fast one rises. Registry order is the cold-start tie-break,
  so first-run behavior is unchanged.
- **In-provider retry with backoff** — a transient 500 gets up to 3 attempts
  on the same provider (exponential backoff + jitter, `Retry-After`
  respected) before the router walks. A flaky provider recovers without
  wasting a failover hop.
- **429s are never retried in-provider** — the quota ledger already owns
  rate-limit cooldowns; retrying only burns quota. Failover instead.
- **Quota ledger** — every provider has a daily request budget. When it's
  exhausted, the router skips that provider *before* hitting the 429.
  Escalating cooldowns on rate limits: 5 min → 15 min → 1 hour.
- **Multi-key rotation** — paste several keys comma-separated
  (`GROQ_KEY=k1,k2,k3`). Dead keys (401/403) are skipped permanently;
  exhausted keys (429) cool down and rotate back in. Key material is never
  written to disk by the rotation state — only indices and statuses.
- **Semantic response cache** — near-duplicate prompts hit a SQLite-backed
  TF-IDF cache (cosine ≥ 0.92, 168h TTL) and return instantly without a
  provider call. Stateful tool conversations skip the cache.
- **Hedged requests** — fire the top two providers concurrently, take the
  first answer, abandon the loser. Kills tail latency. (Threading-based;
  burns 2x quota on slow tails.)
- **Deadline budgets** — `route(..., timeout_budget_s=30)` stops walking
  providers and retrying when the budget is spent.
- **Full attempt trails** — every provider attempt emits an event with
  attempt number, status, latency; every backoff emits a `retry_wait` event.
  The run log shows the complete routing decision path, replayable.

### 2. Serve an OpenAI-compatible HTTP API

```bash
python src/server.py                    # PORT env var, default 8080
```

A stdlib-only HTTP server. Point any OpenAI client at it:

| Endpoint | What it does |
|---|---|
| `POST /v1/chat/completions` | OpenAI-shaped chat with full failover behind it |
| `GET /health` | liveness probe |
| `GET /metrics` | Prometheus text metrics (all `flippy_` prefixed) |
| `GET /usage` | per-provider calls / errors / latency / tokens + cache savings |
| `GET /quota` | live per-provider free-tier headroom |

Auth: set `FLIPPY_AUTH_TOKEN` to require `Authorization: Bearer <token>`.
Binds localhost by default; set `HOST=0.0.0.0` to expose (and set the token).

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hello"}]}'
```

### 3. Run an autonomous agent with hardened tools

```bash
python -m src.loomweaver agent "check disk space with the shell tool"
python -m src.loomweaver agent "fetch example.com and summarize" --session research
python -m src.loomweaver agent "list the src directory" --model groq/openai/gpt-oss-120b
```

A ReAct-style loop: goal → model → tool call → observation → repeat, with:

- **14 guarded tools**: `http_get`, `read_file`, `write_file`, `list_dir`,
  `shell`, `remember` (session facts), `sql_query` (SELECT-only, read-only
  connection), `json_transform` (filter/map/limit), `http_post_json`
  (JSON-validated POST), plus `tts`, `rag_query`, `rag_add`, `summarize`,
  `embed` (aihub-backed, intent-gated: a goal must name the capability to
  get the tool). Tool set is derived from the goal (least privilege) —
  `tool_scope=auto` by default.
- **Security guards on every tool**: SSRF protection (private IPs, cloud
  metadata, DNS-rebinding blocked), path jail (project root + scratch only,
  credential-like paths denied), shell blocklist (25+ dangerous patterns,
  env stripped of `KEY/TOKEN/SECRET/PASSWORD` vars, key-redaction on
  output), SQL writes impossible at two independent layers, cron command
  allowlist. `LOOMWEAVER_SAFE_MODE=1` disables shell entirely.
- **String-aware JSON action parser** — the agent understands model output
  wrapped in prose, markdown fences, brace-containing string values, and
  stray-brace noise. Measured 12/12 on realistic free-tier outputs
  (the old greedy-regex parser scored 10/12).
- **No-progress loop detection** — two consecutive steps with no tool call
  and no done terminate the run with a `no_progress` reason instead of
  burning max_steps. A tool call resets the counter.
- **Budgeted observations** — tool output re-entering context is capped at
  2000 chars with an explicit `[truncated, N more chars]` marker.
- **Persistent sessions** — facts and messages survive across runs, trimmed
  to 40 messages.

### 4. Launch multi-agent fleets (armada)

```bash
python -m src.loomweaver armada "audit the security module and report findings"
```

Four role-specialized agents on one mission, each restricted to its own
toolset (enforced in dispatch code, not just prompts):

| Role | Tools | Job |
|---|---|---|
| Scout | read-only: http_get, read_file, list_dir, shell | gather facts, end with FINDINGS |
| Builder | read_file, write_file, list_dir, shell, http_get | implement, end with BUILT + TESTS |
| Verifier | read_file, list_dir, shell, http_get | adversarial QA, end with VERDICT: PASS/FAIL |
| Reporter | read-only + write | write the summary |

### 5. Benchmark everything

```bash
python -m src.loomweaver eval --suite basic        # 5 cases: math, caps, JSON, counting
python -m src.loomweaver eval --suite reasoning    # logic + code output
python -m src.loomweaver eval --suite extraction   # dates, prices, emails → JSON
python -m src.loomweaver eval --suite tools        # tool-protocol emission
python -m src.loomweaver eval --suite agent        # 4 multi-step agent tasks, scored on
                                                   # required tools + genuine completion
python -m src.loomweaver eval-compare              # suites × models comparison table
```

The agent suite measures the loop, not the model: it detects looping or
never-done models deterministically (a broken planner scores 0/4; the
shipped loop scores 100). Scored from the event trail, so max-steps
exhaustion doesn't count as success.

Load testing and latency:

```bash
python -m src.loomweaver loadtest --provider groq --concurrency 4 --requests 8
python -m src.loomweaver ttft       # streaming time-to-first-token sweep across providers
```

### 6. Track usage, quota, and validate config

```bash
python -m src.loomweaver usage       # dashboard: calls, errors, cache hits, avg latency, tokens
python -m src.loomweaver quota       # per-provider free-tier headroom + cooldown state
python -m src.loomweaver providers   # what's configured right now
python -m src.loomweaver doctor      # validate config: provider keys + writable DB paths
python -m src.loomweaver check-config   # alias of doctor
```

`doctor` runs **without any network calls** and reports each check
`OK / WARN / FAIL`:
- Provider keys present and well-formed per provider (missing key = WARN,
  malformed = FAIL)
- Quota / cache / usage DB paths are writable

It is safe to run with no keys configured — it will WARN, not crash, and
exit informatively.

Also live over HTTP at `/usage` and `/quota`.

### 7. Schedule jobs (opt-in, local)

```bash
python -m src.loomweaver cron --list
python -m src.loomweaver cron --run nightly-eval
python -m src.loomweaver cron --daemon     # interval loop; jobs defined in cron_jobs.json
```

Jobs may only invoke loomweaver subcommands (allowlist enforced) and run
as subprocesses — a poisoned job file can't escalate.

### 8. Multimodal hub (optional, needs litellm)

```bash
pip install -r requirements.txt
python src/aihub.py --health
python src/aihub.py --chat "hello"
python src/aihub.py --vision image.jpg "what is this?"
python src/aihub.py --rag add doc.txt     # build a local vector store
python src/aihub.py --rag query "question"
python src/aihub.py --summarize file.txt
python src/aihub.py --tts "text"          # edge-tts voice
python src/aihub.py --stt audio.wav
```

Chat, vision, RAG (local vector store), summarization, text-to-speech,
speech-to-text — routed through the same litellm provider failover.

## Quickstart

```bash
git clone https://github.com/Rawbeew/flippy && cd flippy

# Set ONE key to start (any of: OPENROUTER_KEY, FREEINFERENCE_KEY,
# CLOUDFLARE_TOKEN + CLOUDFLARE_ACCOUNT_ID, NVIDIA_KEY, GROQ_KEY)
export GROQ_KEY=gsk_...

python src/ai_failover.py "explain KV caches in one paragraph"   # chat with failover
python -m src.loomweaver agent "check disk space using the shell tool"
python -m src.loomweaver doctor   # validate your config before your first real call
pip install pytest && python -m pytest tests/ -q                  # 317 tests
```

## Docker

```bash
cp .env.example .env       # add your keys
docker compose up -d
curl http://localhost:8080/health
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hello"}]}'
```

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full data flow, module map,
scaling points, and trade-offs table.

```
Request → cache check → adaptive ordering → quota check → key rotation
        → provider call (≤3 attempts w/ backoff on 5xx) → usage record
                                                        ↓
                                              failover on failure
```

## Modules

| Module | Purpose |
|---|---|
| `src/flippy_providers.py` | Canonical provider registry (single source of truth) |
| `src/ai_failover.py` | Standalone CLI router |
| `src/aihub.py` | litellm-powered multimodal hub (optional: vision, RAG, TTS, STT) |
| `src/server.py` | stdlib HTTP server: /v1/chat, /health, /metrics, /usage, /quota |
| `src/loomweaver/` | Agent harness: routing core, adaptive router policy, agent loop, armada fleet, evals, security, quota ledger, key rotation, semantic cache, usage, cron |

## Security

Read [SECURITY.md](SECURITY.md) for the threat model: SSRF guards, path jail,
env stripping, key redaction, cron allowlist, and `LOOMWEAVER_SAFE_MODE=1`.

Beyond the base guards, flippy ships a defense-in-depth layer that is
assertion-backed by the test suite:

- **SSRF in shell commands.** `shell` now extracts http(s) URLs and bare
  `host:port` targets and routes them through the same `check_url` used by
  `http_get` — so `curl http://169.254.169.254/...`,
  `wget http://metadata.google.internal/`, `curl http://127.0.0.1:8080`,
  and private-IP targets are blocked even when reached via the shell tool,
  not just via `http_get`.
- **Unicode / homoglyph normalization.** Before any guard matches, input is
  NFKC-normalized: zero-width chars, BIDI overrides, soft hyphens, and
  Cyrillic/Greek homoglyphs of Latin letters are folded, so a credential-read
  like `cat .\u0435nv` (Cyrillic `е`) or `cat id_\u0433sa` (Cyrillic `г`) is
  caught as `cat .env` / `cat id_rsa`.
- **Decoy / deception layer (opt-in).** `LOOMWEAVER_DECOYS=1` plants
  realistic-looking placeholder credential files into `sandbox/` (never
  `src/` or `tests/`). Reading one through the tools returns a generated
  response instead of file contents, and fires a structured telemetry event.
- **Secret redaction.** Tool output is scrubbed of 13+ credential families
  (OpenAI, Anthropic, Groq, NVIDIA, Cloudflare, GitHub classic/oauth/fine-/
  grained PAT, Slack, Stripe, AWS AKIA/ASIA, GCP service-account JSON, PEM
  blocks, JWTs, HF, xAI) before it ever reaches the model.

Run `python -m src.loomweaver doctor` to validate config, and
`LOOMWEAVER_SAFE_MODE=1` to disable the shell tool entirely.

**Zero-trust operator switches** (read live at dispatch, not import-time):
- `FLIPPY_KILL_SWITCH=1` (or `LOOMWEAVER_EMERGENCY_OFF=1`) hard-disables ALL
  agent tool execution immediately — close a compromised run without killing
  the process.
- The tool-running agent (`agent`/`armada`) is NOT a schedulable cron
  subcommand by default — a poisoned `cron_jobs.json` cannot silently fire an
  autonomous, production-credentialed agent run. Re-enable deliberately via
  `LOOMWEAVER_CRON_ALLOW_AGENT=1` (not recommended).

## Not verified / honest limitations

- Zero external users. No production traffic has hit this code.
- Benchmarks are self-reported from a single machine on residential WiFi.
- Free-tier providers only — paid overflow is untested.
- The "semantic" cache is lexical TF-IDF, not embedding-based. It matches
  word overlap, not meaning.
- No async/await. Threading-based hedging exists but the core is synchronous.
- Single-process. No multi-worker mode; SQLite state won't survive
  concurrent writers at scale.
- Native OpenAI-style tool_calling is supported (live-verified against Groq)
  with a JSON-protocol fallback for models that text-inline actions.

## License

MIT

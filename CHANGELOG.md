# Changelog

## 2026-10-05 — Universal providers, self-learning memory, hardening sweep

### `aihub` is the single entry point

- **`litellm` is now a hard dependency**, not an optional extra. `aihub` is the
  console entry point (`flippy = aihub:main`), and `edge-tts` / `pillow` moved
  to the `[hub]` extra since only speech and image support need them.
- **`aihub` consumes the universal provider registry**, so consolidating on it
  does not narrow the provider surface — an arbitrary
  `<PREFIX>_API_KEY` + `<PREFIX>_BASE_URL` endpoint reaches the litellm Router
  unchanged.
- **The self-learning memory is wired into `smart_chat`** on both the success
  and the failure path, and `aihub` grew `--providers`, `--all-providers`,
  `--profile`, `--learn` and `--forget`, so nothing is lost by using it alone.
  The import is lazy and guarded: `aihub` still runs without `loomweaver`.
- **Secret redaction is now derived, not a fixed list.** `_secret_env_vars()`
  hardcoded six brand variables; with an open-ended provider surface that left
  e.g. a configured `MISTRAL_API_KEY` echoable in an error message. It now
  covers every credential-shaped variable plus each configured provider's
  `env_key`.

### Any provider, bring your own keys

- **The registry is no longer five brands.** `flippy_providers.py` is now a
  data-driven catalog of 22 built-ins (OpenRouter, Groq, NVIDIA, Cloudflare,
  OpenAI, Anthropic, Together, Fireworks, DeepInfra, Mistral, Cerebras,
  SambaNova, xAI, Perplexity, Moonshot, DashScope, SiliconFlow, HuggingFace,
  plus keyless Ollama / LM Studio / vLLM), each switched on by its own key.
- **Any endpoint registers itself** from a `<PREFIX>_API_KEY` +
  `<PREFIX>_BASE_URL` pair — no code change, no catalog entry. Credential
  suffixes `_API_KEY`/`_APIKEY`/`_KEY`/`_TOKEN` and endpoint suffixes
  `_BASE_URL`/`_API_BASE`/`_BASE`/`_ENDPOINT`/`_URL` are all accepted, with
  `<PREFIX>_MODELS` overriding the model list.
- **Two more paths** for anything the conventions cannot express: a JSON
  catalog (`FLIPPY_PROVIDERS_JSON`) and numbered endpoints
  (`FLIPPY_PROVIDER_1_*`).
- **A credential with no endpoint is ignored.** A `STRIPE_API_KEY` in the
  environment never becomes an LLM, and infra prefixes (AWS, GITHUB, AZURE,
  …) are excluded outright.
- `flippy providers --all` prints the whole catalog with the variable that
  enables each row. The pre-existing `custom` provider contract
  (`OPENAI_API_BASE` + `OPENAI_API_KEY`) is preserved unchanged.

### Self-learning memory (`learning.py`)

- **Outcome memory.** Every routed call and agent run is recorded — goal,
  provider, latency, attempts, tools, outcome — in one local SQLite file.
- **Routing priors.** On startup the adaptive router replays what previous
  runs measured, so a fresh process inherits the routing table the last
  session earned instead of rediscovering it. The EWMA still decays a stale
  prior as soon as live measurements disagree, and seeding is idempotent per
  policy object so history never compounds.
- **User profile**, inferred and never asked for: reliable provider, tool
  affinity, recurring vocabulary, success rate.
- **Self-correction.** A failed run files a lesson against the shape of its
  goal; a *similar* future goal retrieves it into the system prompt by TF-IDF
  cosine. Uninformative errors ("all providers failed") are dropped rather
  than stored. Operators can teach directly: `flippy learn "<rule>"`.
- New commands: `flippy profile [--json]`, `flippy learn`, `flippy forget`.
  A cold start injects nothing, so a first-time user pays no context cost.
  Local only; disable with `LOOMWEAVER_LEARNING_ENABLED=0`.

### Security

- **Read-only shell tier.** `security.check_shell(cmd, readonly=True)` denies
  file-mutating commands, output redirection, `tee`, network clients, and
  mutating `git` subcommands. Armada roles declared `readonly` are dispatched
  under it, so the flag is a guarantee enforced by the guard rather than a
  note in a tools list. The mode is thread-local caller context, never a tool
  argument, so a model cannot opt itself out.
- **`_WRITE_TOOLS` now covers `shell` and `sql_query`.** `shell` is a
  general-purpose write primitive; omitting it made `readonly` decorative.
- **Descriptor duplication no longer misread as a separator.** The `&` in
  `2>&1` used to make the segmenter emit a phantom command named `1`, which
  the default-deny allowlist then rejected — so `ls -la 2>&1` was blocked.
- **Redirection is re-checked on every hop** and `tools.http_get` /
  `http_post_json` route through the guarded opener rather than a bare
  `urlopen`, closing the public-URL → 302 → metadata-address path.
- **Wider redaction**: `gsk_`, `fi[-_]`, `xox*`, GitHub PAT families, Stripe
  key families, `*KEY=`/`*TOKEN=` assignments, private keys, GCP service
  accounts and JWTs — while leaving non-secrets such as
  `CLOUDFLARE_ACCOUNT_ID` readable.

### Correctness

- **`agent._INTENT` keys are real tool names.** Eleven entries named tools
  that do not exist (`write`, `sql`, `json`, `post`, …), so those intents
  never granted anything. Keyword matching is word-boundary, so the goal word
  "tools" no longer matches the `ls` keyword.
- **Dispatch-time authorization.** A tool outside the run's authorized set is
  refused at both dispatch sites rather than executed, making the documented
  "enforced in dispatch" claim true for the single-agent path. An explicit
  `--tools` list with a typo now fails closed instead of silently widening.
- **`semantic_cache` is thread-safe**: a connection per operation instead of
  one shared handle, so concurrent `ThreadingHTTPServer` requests no longer
  contend on a single cursor. Four cache tests that were orphaned under
  `if __name__ == "__main__"` now run.
- **Key rotation keeps its cursor.** `mark_dead`/`mark_exhausted` saved
  `active_index=0`, rewinding every provider to its first key on any failure,
  so a multi-key fleet never spread load.
- `cron --daemon` survives a job that has no state row yet (was a `KeyError`
  that killed the loop); `loadtest` with no providers errors informatively
  instead of raising `IndexError`; `aihub --rag query` prints every hit
  instead of only the first; the `aihub` JWT redaction pattern matched a
  literal backslash and so never matched a real JWT.
- **Server**: 4 MB request-body cap (413 on overflow), and the provider
  environment now covers eleven variables.

### Tests, docs, hygiene

- **381 tests** (from 327), 72% statement coverage. New suites:
  `test_learning.py`, `test_provider_catalog.py`. `conftest.py` now isolates
  the learning store and the usage DB per test — previously one test's
  recorded routes became the next test's routing priors, and every run
  appended to the developer's real databases.
- **`scripts/verify_fixes.py`** re-runs the original proof-of-exploit for each
  finding independently of the test suite. Current: 29 passed, 0 failed,
  2 manual (docker, live-provider).
- Docs corrected against the code: `ARCHITECTURE.md` cited a
  `key_rotation.next_key()` that does not exist and omitted `observability.py`,
  `router_policy.py` and `learning.py`; `LLM.txt` claimed 173 tests;
  `__init__.py` advertised a `python_exec` tool that was never registered;
  the README claimed hedging "kills tail latency" while `benchmarks/HEDGED.md`
  records p95 getting **worse** (4.360s vs 1.635s).
- Removed the personal Windows launcher from the repo root (it hardcoded a
  developer's home directory) and the remaining third-party tool names from
  the prose. `.gitignore` now covers `sandbox/`, coverage artifacts and the
  learning store. `.env.example` documents `FLIPPY_AUTH_TOKEN`, without which
  the documented `docker compose up` path exits 2 by design.

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

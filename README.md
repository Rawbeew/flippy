# flippy 🐟

**Free-first multi-provider LLM inference, with a complete agent harness built in.**

One stdlib-only Python package that routes your prompts across five free LLM providers,
never stops on a rate limit, and ships with **Loomweaver** — an autonomous agent runner,
eval suites, and inference load-testing.

[![CI](https://github.com/Rawbeew/flippy/actions/workflows/ci.yml/badge.svg)](https://github.com/Rawbeew/flippy/actions/workflows/ci.yml)
![License](https://img.shields.io/badge/license-MIT-green)
![Deps](https://img.shields.io/badge/router%20deps-stdlib%20only-blue)

---

## Why flippy

Every free LLM provider rate-limits. Most apps die the day one does.
flippy flips between them automatically:

```
your prompt ──▶ Router ──▶ ① OpenRouter ──fail──▶ ② freeinference ──fail──▶ ③ Groq ──▶ ✅ response
                            (stealth/ox-alpha)     (minimax-m3 …)          (gpt-oss-*)
```

- **Free-first ordering** — paid capacity is never touched before free tiers
- **Failover on 429/5xx** — transient errors move to the next provider mid-request
- **Provider pinning** — `groq/openai/gpt-oss-20b` forces a provider; bare model IDs match any provider that has them
- **Zero dependencies** — the router and Loomweaver are pure Python stdlib

## The two layers

| Layer | Module | What it gives you |
|---|---|---|
| **Router** | `src/ai_failover.py` | One-shot chat with automatic failover + Prometheus-style latency metrics |
| **AI Hub** | `src/aihub.py` | litellm-powered: chat, vision, function calling, embeddings/RAG, TTS, STT *(needs `pip install litellm edge-tts`)* |
| **Loomweaver** | `src/loomweaver/` | Agent loop, eval suites, load testing, cron |

## Quickstart

```bash
git clone https://github.com/Rawbeew/flippy && cd flippy
export OPENROUTER_KEY=...      # any ONE of these is enough to start
export FREEINFERENCE_KEY=...
export GROQ_KEY=...

# one-shot chat with failover
python src/ai_failover.py "explain KV caches in one paragraph"
```

### Loomweaver — the harness

```bash
python -m src.loomweaver providers                    # see what you're routed across
python -m src.loomweaver agent "check disk space"     # autonomous goal → tools → done
python -m src.loomweaver eval --suite basic           # score a model
python -m src.loomweaver eval --suite tools           # tool-selection accuracy
python -m src.loomweaver loadtest --provider groq     # concurrency + latency profile
python -m src.loomweaver ttft                         # time-to-first-token sweep
python -m src.loomweaver cron --daemon                # scheduled evals/loadtests
```

### AI Hub — multimodal *(optional deps)*

```bash
pip install litellm edge-tts
python src/aihub.py --chat "hello"
python src/aihub.py --vision photo.jpg "what's in this?"
python src/aihub.py --rag add "notes.txt" && python src/aihub.py --rag-chat "summarize my notes"
```

## Loomweaver in depth

The agent loop: **goal → plan → act (tools) → observe → done**, with every step
appended to a replayable JSONL event log (`runs/<timestamp>/events.jsonl`).

Built-in tools: `shell`, `http_get`, `read_file`, `write_file`, `list_dir`, `remember`.
Add your own in three lines:

```python
from loomweaver.tools import tool

@tool("get_price", "Fetch BTC price", {"symbol": "str"})
def get_price(symbol):
    ...
```

Eval suites: `basic` · `reasoning` · `extraction` · `tools` — each case scored by
substring, exact-JSON, or regex match, with per-case latency and provider attribution.

Load tests report p50/max latency, requests/sec, tokens/sec, and success rate
under configurable concurrency. The TTFT sweep streams from every provider and
ranks them by first-token latency.

## Credentials

flippy reads keys from the environment (`~/.bashrc`, `.env`, whatever you use):

| Env var | Provider | Notes |
|---|---|---|
| `OPENROUTER_KEY` | OpenRouter | primary slot |
| `FREEINFERENCE_KEY` | freeinference.org | free tier |
| `GROQ_KEY` | Groq | fastest streaming |
| `NVIDIA_KEY` | NVIDIA NIM | 100+ models |
| `CLOUDFLARE_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` | Workers AI | no streaming |

Set just one and everything works; set all five for maximum resilience.

## Security

The agent runs model-chosen shell commands — read the honest threat model in
[SECURITY.md](SECURITY.md): SSRF guards, path jail, env stripping, key
redaction, cron allowlist, and `LOOMWEAVER_SAFE_MODE=1` to disable shell entirely.

## Deployment

flippy runs anywhere Python runs — it's stdlib-only with no services required.

**Where it runs in production (my setups):**
- **Local / VM**: long-lived process, keys in environment (`~/.bashrc` or a 600-perm credential file)
- **Docker**: `COPY src/ .` + slim python:3.12 image; the router itself has zero build deps
- **Serverless-ready**: `route()` is a pure function over env-configured providers — wrap it in any FaaS handler (Lambda/Cloud Functions) with no changes

**Environment variables:** see [Credentials](#credentials) — one key minimum, all five for full failover depth.

**Failure behavior:** free-first ordering; on 429/5xx or network error the next provider is tried within the same request; auth errors (401/403) skip to the next provider immediately. Prometheus-style latency histogram available via `render_metrics()`.

**Cost:** $0 by design — every configured provider has a free tier. Paid capacity is never touched before free tiers are exhausted.

## Testing

```bash
pip install pytest
python -m pytest tests/ -q      # 15 unit tests, fully mocked — no network
```

CI runs tests on Python 3.11–3.13 plus compile checks and a credential-free CLI smoke test.

## Measured numbers

From `benchmarks/RESULTS.md` (2026-08-21, 20 requests/provider, real keys):

| Provider | Model | Success | p50 latency | p95 latency | Tokens/s |
|---|---|---|---|---|---|
| groq | `openai/gpt-oss-20b` | 100% | 0.221 s | 0.397 s | 35.4 |
| openrouter | `stealth/ox-alpha` | 100% | 1.058 s | 1.251 s | 7.3 |
| freeinference | `minimax-m3` | 100% | 1.166 s | 2.817 s | 7.2 |
| cloudflare / nvidia | — | 0% (401 / 403 — stale tokens) | — | — | — |

The two dead providers are the point: the router skipped both and every request still landed on a healthy provider within the same call. Re-run yourself with `python benchmarks/run_bench.py`.

## What I would improve next

Honest trade-offs in the current design:

1. **Streaming-first API surface** — `ai_failover.py` is request/response today; TTFT is measured (`loomweaver ttft`) but tokens aren't streamed through the router API itself. First-class async streaming generators would cut perceived latency for chat UIs.
2. **Response caching with semantic dedupe** — identical/near-identical prompts re-hit providers. A small embedding-based cache would cut free-tier quota burn meaningfully.
3. **Cost tracking per provider** — metrics capture latency and success, not spend. Per-provider token + dollar accounting belongs in the same histogram.
4. **Async rewrite of aihub** — aihub wraps litellm synchronously; concurrent multi-provider fan-out blocks threads. asyncio would make parallel failover probes cheap.

## License

MIT

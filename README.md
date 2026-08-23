# flippy

A multi-provider LLM failover router with an agent harness. Stdlib-only core.
No production traffic yet — this is a well-tested personal tool, not battle-tested infrastructure.

[![CI](https://github.com/promptcracka/flippy/actions/workflows/ci.yml/badge.svg)](https://github.com/promptcracka/flippy/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/tests-116%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## What it actually does (no marketing)

Routes LLM chat requests across 5 free-tier providers with automatic failover.
If one provider rate-limits you or goes down, the next one picks up mid-request.

**Verified capabilities:**
- Failover across OpenRouter, freeinference.org, Groq, NVIDIA NIM, Cloudflare Workers AI
- Semantic response cache (TF-IDF cosine, stdlib-only) — near-duplicate prompts hit cache
- Per-provider quota tracking — skips exhausted providers before the 429
- Multi-key rotation per provider — comma-separated env vars
- Usage dashboard — per-provider calls/errors/latency/tokens + cache savings (`usage` CLI, `/usage` endpoint)
- Quota status endpoint — live per-provider free-tier headroom (`quota` CLI, `/quota` endpoint)
- Agent loop with role-based tool restrictions (scout/builder/verifier/reporter)
- 147 unit tests including failure injection (mocked 429s, timeouts, malformed responses)

**Not verified / honest limitations:**
- Zero external users. No production traffic has hit this code.
- Benchmarks are self-reported from a single machine on residential WiFi.
- Free-tier providers only — paid overflow is untested.
- The "semantic" cache is lexical TF-IDF, not embedding-based. It matches word overlap, not meaning.
- No async/await. Threading-based hedging exists but the core is synchronous.

## Quickstart

```bash
git clone https://github.com/promptcracka/flippy && cd flippy

# Set ONE key to start
export GROQ_KEY=gsk_...

# Chat with automatic failover
python src/ai_failover.py "explain KV caches in one paragraph"

# Or run the agent harness
python -m src.loomweaver agent "check disk space using the shell tool"

# Run the test suite
pip install pytest && python -m pytest tests/ -q
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
Request → cache check → quota check → key rotation → provider call → usage record
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
| `src/loomweaver/` | Agent harness: routing core, agent loop, armada fleet, evals, security |

## Security

Read [SECURITY.md](SECURITY.md) for the threat model: SSRF guards, path jail,
env stripping, key redaction, cron allowlist, and `LOOMWEAVER_SAFE_MODE=1`.

## Limitations

This is explicitly what flippy does NOT have:

- **No production deployment.** It runs on my machine. Nobody else's traffic has hit it.
- **No async.** Synchronous routing with threading for hedged requests.
- **Lexical cache only.** TF-IDF cosine on words, not embeddings. Won't match paraphrases that change vocabulary.
- **Free tiers only.** Paid overflow is configured but untested.
- **Single-process.** No multi-worker mode; SQLite state won't survive concurrent writers at scale.
- **Self-reported benchmarks.** All latency numbers are from my laptop.

## License

MIT

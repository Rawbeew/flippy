# flippy Router Benchmark Results

Date: 2026-08-21 · 5 provider(s) · 20 requests/provider

| Provider | Model | Success | Latency p50 (s) | Latency p95 (s) | Tokens/s |
|---|---|---|---|---|---|
| openrouter | `stealth/ox-alpha` | 100.0% (20/20) | 1.058 | 1.251 | 7.25 |
| cloudflare | `@cf/meta/llama-3.3-70b-instruct-fp8-fast` | 0.0% (0/20) | 0.068 | 0.114 | n/a |
| freeinference | `minimax-m3` | 100.0% (20/20) | 1.166 | 2.817 | 7.18 |
| groq | `openai/gpt-oss-20b` | 100.0% (20/20) | 0.221 | 0.397 | 35.37 |
| nvidia | `nvidia/llama-3.3-nemotron-super-49b-v1` | 0.0% (0/20) | 0.136 | 0.143 | n/a |

See README.md for methodology. Raw data: results-2026-08-21.json

## Notes on failures

- **cloudflare — 0% (HTTP 401 ×20):** the configured Cloudflare Workers AI token
  was rejected at auth. Not a router issue; token needs rotation.
- **nvidia — 0% (HTTP 403 ×20):** NVIDIA NIM rejected all requests (403 — key
  lacks access or is expired). Token needs rotation.

These failures are exactly the case flippy's failover exists for: the router
skips them and lands on a healthy provider.

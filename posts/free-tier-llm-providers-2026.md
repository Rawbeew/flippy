---
title: "Free-tier LLM providers you can actually use in 2026"
date: 2026-08-18
tags: [llm, ai, free-tier, openai-compatible, freeinference, groq, nvidia, cloudflare]
canonical: https://github.com/promptcracka/flippy
---

# Free-tier LLM providers you can actually use in 2026

If you're building a product that calls a language model, the cheapest
correct answer is almost always: **don't pay.** At least not until you've
proven the product works at all. As of 2026, four providers give you real
free-tier access with OpenAI-compatible APIs:

## 1. freeinference.org

A free, OpenAI-compatible aggregator. Several current flagship models
available (`minimax-m3`, `qwen3.6-35b`, `deepseek-v4-flash`, `glm-5.1`) plus
embeddings (`bge-m3`). Sign-up is email-only. The only gotcha: their server
rejects the default Python `urllib` user agent with a 403 — you must send a
browser user agent (`Mozilla/5.0 ... Chrome/126.0`). Five minutes to fix,
but invisible until you try it.

## 2. Groq free tier

Blazing-fast inference on LPU hardware. Free models include `gpt-oss-20b`
and `gpt-oss-120b`. The catch: an 8,000 TPM cap that you will hit the moment
your system prompt grows past a few KB. Fine for short prompts, not fine for
agentic loops with large tool schemas. **TTS (`canopylabs/orpheus`) and STT
(`whisper-large-v3`) are blocked at the project level by default** — you
have to enable them once in the console.

## 3. NVIDIA NIM

Free tier via `integrate.api.nvidia.com`. The reliable chat model in the
free tier is `nvidia/llama-3.3-nemotron-super-49b-v1`. Many other models
(`meta/*`, vision models) are queued or paywalled. Heads up: NVIDIA's TLS
endpoint stalls under Windows `curl` (schannel renegotiation issue) — use
Python urllib/requests instead. Verified in production.

## 4. Cloudflare Workers AI

The most underrated option. Free tier, but with two constraints worth
knowing:

- The model is **in the URL path** (`POST /accounts/<id>/ai/run/<model>`),
  not the body. Different shape from every other provider.
- Some models (`@cf/meta/llama-3.2-11b-vision-instruct`) need a one-time
  license acceptance in the Cloudflare dashboard. The license agreement is
  not clearable via API — you must click accept once.

Cloudflare is also the only one of the four whose STT (`whisper-large-v3-
turbo`) works out of the box without any extra steps.

## Putting them together

You don't pick one. You put all four in a free-first ordered list and fall
over when one fails. The whole point is that when freeinference 429s you
during peak, Groq picks it up; when Groq hits its TPM cap, NVIDIA picks it
up; when NVIDIA is queued, Cloudflare picks it up.

The full working pattern is in [`src/ai_failover.py`](https://github.com/promptcracka/flippy).
~200 lines. Add your API keys as environment variables and you have a
multi-provider setup that costs $0/month.

See also: [`posts/why-llm-rate-limits-are-an-operational-problem.md`](https://github.com/promptcracka/flippy/blob/master/posts/why-llm-rate-limits-are-an-operational-problem.md).

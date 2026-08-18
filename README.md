# ai-portfolio

A small, honest toolkit for building applications that **use** large language
models — without ever getting stuck on a single provider's rate limit.

Two scripts. Both free-first. Both written to be read by a human.

---

## Why this exists

If you've shipped anything with LLMs in production, you've learned the lesson:

> **Every provider will rate-limit you. The question is whether your system keeps
> working when they do.**

This toolkit is the answer I reached for after that lesson hurt a real
project. It grew out of building — and breaking — systems that needed to keep
chatting, embedding, generating, and listening across multiple providers
without a single point of failure.

It's also the code I'd want a candidate to show me, if I were hiring for an
AI/LLM applications role. It's not flashy. It's just *correct*.

## What's in here

| File | What it does |
|---|---|
| `src/ai_failover.py` | Multi-provider inference router. Reads from environment variables. Tries free providers first, falls over on 429/5xx, never throws a rate-limit error to the caller. Zero dependencies — pure Python stdlib. |
| `src/aihub.py` | Unified hub on top of `ai_failover`. Built on `litellm.Router`. Adds smart routing (cheap model for trivial turns), RAG with local vector store, tool/function calling, vision, TTS, STT, prompt summarization, and request caching for token savings. |

That's it. No magic. ~400 + 200 lines of Python.

## Design principles

- **Free-first ordering.** Free providers are tried before any paid fallback.
- **Fail fast, recover gracefully.** A 429 or 5xx from one provider triggers an
  immediate attempt on the next, with a brief backoff between hops.
- **Cooldown after repeated failures.** A provider that just failed you twice
  is left alone for 60s so it isn't hammered into the ground.
- **Pure stdlib for the router.** The `ai_failover.py` script depends on
  nothing but Python's standard library — drop it into any project and it
  works.
- **Secrets stay secrets.** No keys in the repo. No keys in code. Read from
  environment variables, like every grown-up project does.

## Quickstart

```bash
pip install litellm edge-tts pillow

# at minimum, set ONE free provider key
export FREEINFERENCE_KEY=...

# optionally add more providers
export GROQ_KEY=...
export NVIDIA_KEY=...
export CLOUDFLARE_TOKEN=...
export CLOUDFLARE_ACCOUNT_ID=...

# basic chat with auto-failover
python src/ai_failover.py "Explain the CAP theorem in one paragraph"

# unified hub — health, chat, vision, RAG, TTS, STT, all available
python src/aihub.py --health
python src/aihub.py --chat "Write a haiku about caching"
```

See [`QUICKSTART.md`](./QUICKSTART.md) and [`examples/README.md`](./examples/README.md)
for the full menu.

## What this is (and isn't)

This repository contains the **portable, secret-free, public-portfolio** code.
The hardened versions of these scripts live in private projects and have been
running in production-like workflows for:

- a static job-board site (scrapes fresh direct-employer postings every 2h,
  tailors resume + cover letter per posting)
- a crypto-trading bot (uses an LLM as the final trade-decision reasoner,
  with structured BUY/SKIP prompting)
- a Cloudflare Worker that keeps the job board's data pipeline alive (bypasses
  a Job Bank IP block by serving from Cloudflare's trusted egress)

If you're hiring for an **AI / LLM Applications Engineer** role and want to
see this code in a real product context, the README of [`jummai-job-finder`](https://github.com/Rawbeew/jummai-job-finder)
shows the full stack this code supports.

## Why an English degree matters here

Every line in `ai_failover.py` and `aihub.py` that talks to a model is a
short story: a system prompt, a tool description, an instruction. Bad prose
becomes bad behavior — the model misreads "reply with *only* a JSON object" as
"reply with some text and maybe a JSON object."

Most engineers can ship reliable code. Few engineers can also write prompts
that survive contact with a real user. That's the part my B.A. in English
gives me an edge on: every prompt here has been rewritten at least once
because the model misunderstood it once.

## License

MIT. See [`LICENSE`](./LICENSE).

# flippy

Free-first multi-provider LLM inference router with unified AI hub — now with **Loomweaver**, the complete harness: agent runner, eval suites, and inference load-testing.

## What it does
- Routes requests across 5 free-tier providers (OpenRouter, freeinference, Groq, NVIDIA, Cloudflare)
- Automatic failover on 429/5xx with exponential backoff
- Unified OpenAI-compatible API for chat, RAG, tools, vision, TTS, STT
- **Loomweaver subpackage**: autonomous agent loop, eval suites (basic/reasoning/extraction/tools), concurrency load-testing with TTFT streaming metrics, local cron scheduler
- Zero non-stdlib dependencies in the router; Loomweaver is stdlib-only too
- Built for production reliability

## Quickstart
```bash
export FREEINFERENCE_KEY=...
export GROQ_KEY=...
export NVIDIA_KEY=...
python src/ai_failover.py "your prompt"
```

## Loomweaver quickstart
```bash
python -m src.loomweaver providers          # list configured providers
python -m src.loomweaver agent "check disk space with the shell tool"  # run the agent
python -m src.loomweaver eval --suite basic     # eval suite
python -m src.loomweaver loadtest --provider groq  # load-test
python -m src.loomweaver ttft               # streaming TTFT sweep
python -m src.loomweaver cron --list        # scheduled jobs
```

## Architecture
```
Request → Router → [openrouter | freeinference | Groq | NVIDIA | Cloudflare] → Response
                ↓
          First success returns
          Failed providers cooled down

Loomweaver: goal → plan → act(tools) → observe → done
            every step logged to runs/<ts>/events.jsonl
```

## License
MIT

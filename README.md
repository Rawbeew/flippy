# flippy

Free-first multi-provider LLM inference router with unified AI hub.

## What it does
- Routes requests across 4 free-tier providers (freeinference, Groq, NVIDIA, Cloudflare)
- Automatic failover on 429/5xx with exponential backoff
- Unified OpenAI-compatible API for chat, RAG, tools, vision, TTS, STT
- Zero non-stdlib dependencies in the router
- Built for production reliability

## Quickstart
```bash
export FREEINFERENCE_KEY=...
export GROQ_KEY=...
export NVIDIA_KEY=...
python src/ai_failover.py "your prompt"
```

## Architecture
```
Request → Router → [freeinference | Groq | NVIDIA | Cloudflare] → Response
                ↓
          First success returns
          Failed providers cooled down
```

## License
MIT

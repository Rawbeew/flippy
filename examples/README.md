# ai-portfolio demos

Working examples that show real usage of `ai_failover.py` and `aihub.py`.

## Setup
```bash
export FREEINFERENCE_KEY=...
# optional additional providers:
export GROQ_KEY=...
export NVIDIA_KEY=...
export CLOUDFLARE_TOKEN=...
export CLOUDFLARE_ACCOUNT_ID=...
```

## Run

```bash
# failover router — health
python ../src/ai_failover.py --health

# failover router — quick prompt
python ../src/ai_failover.py "Explain the CAP theorem in one paragraph"

# failover router — specific model
python ../src/ai_failover.py --model minimax-m3 "Hello!"

# aihub — health
python ../src/aihub.py --health

# aihub — chat
python ../src/aihub.py --chat "Write a haiku about caching"

# aihub — route to cheapest model
python ../src/aihub.py --simple --chat "yes"

# aihub — tool calling
python ../src/aihub.py --tooltest

# aihub — embeddings
python ../src/aihub.py --embed "the quick brown fox"

# aihub — RAG: add then query
python ../src/aihub.py --rag add "Hermes is a personal AI agent."
python ../src/aihub.py --rag query "what is hermes"
python ../src/aihub.py --rag-chat "what is hermes"

# aihub — TTS
python ../src/aihub.py --tts "Hello world, from the hub."

# aihub — STT (audio_path = .mp3 / .wav)
python ../src/aihub.py --stt ./sample.mp3

# aihub — vision
python ../src/aihub.py --vision ./photo.jpg "What is in this image?"
```

## Notes
- The `--rag` store is file-backed at `~/.aihub_vectors.json` (override via
  `AIHUB_VECTOR_STORE` env var). It's portable, not high-performance — use a
  real vector DB for production.
- Free-first ordering + cooldowns mean you'll see output even if a single
  provider rate-limits you. Run `--health` to verify your providers are alive.

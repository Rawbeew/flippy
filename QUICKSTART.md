# Quickstart

Get flippy running in 60 seconds.

## 1. Clone + set one key

```bash
git clone https://github.com/Rawbeew/flippy && cd flippy
# any provider — flippy is OpenAI/Anthropic-compatible, no single brand required
export OPENROUTER_KEY=sk-or-...   # or GROQ_KEY / NVIDIA_KEY / OPENAI_API_BASE+OPENAI_API_KEY / ANTHROPIC_BASE_URL+ANTHROPIC_API_KEY
```

## 2. Chat with failover

```bash
python src/ai_failover.py "why is speculative decoding faster?"
```

If your first provider rate-limits, flippy flips to the next one automatically.

## 3. Run the agent

```bash
python -m src.loomweaver agent "use the shell tool to check free disk space, then report DONE"
```

## 4. Score a model

```bash
python -m src.loomweaver eval --suite basic      # 5 quick checks
python -m src.loomweaver eval --suite tools      # tool-selection accuracy
```

## 5. Find your fastest provider

```bash
python -m src.loomweaver ttft
```

## Optional: multimodal hub

```bash
pip install litellm edge-tts
python src/aihub.py --chat "hello"
python src/aihub.py --vision photo.jpg "describe this"
```

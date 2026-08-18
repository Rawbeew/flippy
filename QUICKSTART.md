# Quickstart — ai-portfolio

## Install
```bash
git clone https://github.com/Rawbeew/ai-portfolio.git
cd ai-portfolio
pip install litellm edge-tts pillow
```

## Set credentials (free tiers)
```bash
export FREEINFERENCE_KEY=...        # https://freeinference.org (free)
# optional:
export GROQ_KEY=...                  # https://console.groq.com (free tier)
export NVIDIA_KEY=...                # https://build.nvidia.com (free tier)
export CLOUDFLARE_TOKEN=...          # + CLOUDFLARE_ACCOUNT_ID for STT/vision
export CLOUDFLARE_ACCOUNT_ID=...
```

## First commands
```bash
python src/ai_failover.py --health          # confirm providers are alive
python src/ai_failover.py "Hello there"     # basic chat with auto-failover
python src/aihub.py --health                # see the unified hub's providers
python src/aihub.py --chat "Quick test"     # routed through litellm Router
python src/aihub.py --tooltest              # function-calling demo
```

That's it. See `examples/README.md` for the full menu.

## What this is (and isn't)
This is **portable, secret-free, public-portfolio** code. The real, hardened
versions live in private projects on this machine and have been running in
production-like workflows for the `jummai-job-finder` static site, the
`archimeda` crypto-trading bot, and a Cloudflare Worker that fetches fresh
Ontario job postings every 2 hours.

If you're hiring for an **AI / LLM Applications Engineer** role and want to
see this in a real product context, see the private work — get in touch.

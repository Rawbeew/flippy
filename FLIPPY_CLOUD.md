# flippy Cloud — Free-Tier LLM Routing as a Service

## Positioning

**"Stop managing API keys and rate limits. Bring your free-tier keys, we handle the routing."**

Hosted version of flippy. Users sign up, paste their free-tier API keys (Groq, OpenRouter, NVIDIA, etc.), get an OpenAI-compatible endpoint that automatically:
- Routes to whichever provider has quota remaining
- Falls over on failure
- Caches duplicate/near-duplicate prompts
- Tracks usage per provider
- Rotates multiple keys per provider

## The insight

Every developer building with LLMs hits the same wall: free-tier rate limits. They either:
1. Pay for a single provider (expensive)
2. Build their own routing (weeks of work)
3. Use OpenRouter (5% markup on paid tiers)

Nobody offers: "bring your FREE keys from multiple providers, we'll maximize their combined usage." flippy Cloud does exactly that.

## Pricing model

| Tier | Price | What you get |
|---|---|---|
| **Free** | $0 | 1,000 requests/month, 2 providers, community support |
| **Hobby** | $9/mo | 10,000 requests/month, all 5 providers, semantic cache, usage dashboard |
| **Pro** | $29/mo | 50,000 requests/month, multi-key rotation, priority failover, API access |
| **Team** | $99/mo | 200,000 requests/month, team management, SLA, custom rate limits |

**Why users pay:** They're getting ~$200+/month of free-tier LLM capacity (Groq + OpenRouter + NVIDIA + Cloudflare free tiers combined) managed automatically. $9-29/mo is nothing compared to what they'd pay for equivalent paid-tier usage.

**Your cost:** Cloudflare Workers ($0 free tier) + SQLite/D1 ($0) + your time.

## MVP scope (what to build first)

### Phase 1: Landing page + waitlist (week 1)
- Landing page: verysketchy.lol-style design, explains the value prop
- Waitlist form (email capture)
- Deploy on Cloudflare Pages (free)

### Phase 2: Working MVP (weeks 2-3)
- Cloudflare Worker that accepts OpenAI-compatible POST /v1/chat/completions
- User registration (email + hashed keys)
- Users store their provider keys (encrypted)
- Basic routing: failover + quota tracking
- /health and /usage endpoints per user

### Phase 3: Monetization (weeks 3-4)
- Stripe integration for paid tiers
- Rate limiting per tier
- Usage dashboard
- Semantic cache (D1 or KV)

### Phase 4: Launch (week 4-5)
- Show HN: "Show HN: I built a free-tier-first LLM routing service"
- Post on r/LocalLLaMA, r/SideProject
- Product Hunt launch

## Technical architecture

```
User's app
    │
    ▼ POST /v1/chat/completions (OpenAI-compatible)
┌─────────────────────────────────────────────┐
│         Cloudflare Worker (flippy Cloud)     │
│                                              │
│  1. Auth: validate user API key             │
│  2. Cache: check D1/KV for similar prompt   │
│  3. Route: pick provider with quota left    │
│  4. Call: forward to provider with user key │
│  5. Record: log usage to D1                 │
│  6. Cache: store response                   │
└─────────────────────────────────────────────┘
    │
    ▼
User's free-tier provider keys (Groq, OpenRouter, NVIDIA, etc.)
```

**Key architectural decisions:**
- Cloudflare Workers = $0 hosting, global edge, no cold starts
- D1 (SQLite at edge) = $0 for storage, SQL queries
- KV for rate limiting and cache = $0 at low volume
- Users' keys stored encrypted in D1 — you never see them in plaintext
- No backend server needed — everything runs at the edge

## Revenue projection (conservative)

| Month | Free users | Paid users | MRR |
|---|---|---|---|
| 1 | 100 | 5 | $45 |
| 3 | 500 | 25 | $225 |
| 6 | 2,000 | 100 | $900 |
| 12 | 5,000 | 300 | $2,700 |

Even $45/mo MRR proves the concept. $900/mo makes it real.

## What you already have

- ✅ Routing engine (flippy core) — port to Workers
- ✅ Quota tracking — port quota_ledger to D1
- ✅ Semantic cache — port TF-IDF cache to KV
- ✅ Key rotation — port to per-user key management
- ✅ Docker deployment — already works
- ✅ CI/CD — GitHub Actions already configured
- ✅ Security model — SSRF guards, path jails, env stripping
- ✅ Tests — 116+ tests to port/adapt
- ✅ Landing page design language — verysketchy.lol aesthetic

## What you need to build

- [ ] User auth (email + password, or magic link via Resend free tier)
- [ ] Key encryption (Web Crypto API in Workers — AES-GCM)
- [ ] Stripe checkout + webhooks
- [ ] Usage metering per user
- [ ] Dashboard (simple HTML/JS, verysketchy aesthetic)
- [ ] Landing page with waitlist

## Marketing angles

1. **"Stop paying for LLM API calls"** — target indie hackers, solo devs
2. **"Your free tiers, combined"** — the math: Groq 14k + NVIDIA 40/min + Cloudflare 10k + OpenRouter = massive combined capacity
3. **"OpenAI-compatible"** — drop-in replacement, change one URL
4. **"Your keys stay yours"** — privacy angle vs OpenRouter-style proxies
5. **"147 tests"** — engineering credibility signal

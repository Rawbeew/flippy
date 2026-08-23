---
title: "How to survive your first LLM rate limit in production"
date: 2026-08-18
tags: [llm, ai, openai, failover, reliability, production]
canonical: https://github.com/promptcracka/flippy
---

# How to survive your first LLM rate limit in production

Your application is live. People are using it. Then, somewhere around peak
traffic on a Tuesday afternoon, you see this in your logs:

```
openai.RateLimitError: Error code: 429 — Rate limit reached for requests
```

It is, in my experience, the single most common production incident in any
product that talks to a large language model. It is also one of the easiest
to make survivable — once you accept that the question is **when** your
provider will rate-limit you, not **if**.

## The real fix: stop having a single provider

The reflex solution is to add retry-with-backoff. That helps. It does not
solve the problem, because the rate limit is on your *provider account*,
not on the request. Backing off and retrying the same provider at the same
account just means you keep getting 429s.

The fix is to have a **second provider you can fall over to.** And a third.
The cost of a multi-provider setup is, at the free tier, basically zero. The
cost of being offline during a launch is not.

## A minimal pattern

The simplest workable failover router has these properties:

- A list of providers, tried in order.
- For each provider: API base URL, the model to call, an API key.
- A request that returns 429, 5xx, or a transport error → try the next.
- After a configurable number of consecutive failures on a provider, leave
  it alone for a cooldown window so it isn't hammered into the ground.
- Free-tier providers first, paid fallback last (so you don't accidentally
  spend money during a transient outage on a free provider).

That is, very nearly, the entire surface of [`ai_failover.py`](https://github.com/promptcracka/flippy/blob/master/src/ai_failover.py).
~200 lines of Python. Zero non-stdlib dependencies.

## The lesson that hurts

I shipped a system that did not have a fallback once. The first provider
went down for an hour during a launch. We recovered by manually rerouting
DNS for an hour. The fix after that incident was an afternoon's work; it
should have been a Saturday's work *before* launch.

If you're reading this before the launch — spend the Saturday. The code is
already written for you.

See [`src/ai_failover.py`](https://github.com/promptcracka/flippy) for the
reference implementation.

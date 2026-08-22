# Post-mortem: The routing pin bug that sent requests to the wrong provider

**Date:** 2026-08-21
**Severity:** High (silent misrouting)
**Status:** Fixed
**Discovered by:** Live smoke test during a routine verification pass

---

## Summary

The `route()` function in `loomweaver/core.py` had a logic inversion in its
model-pinning code. When a user pinned a model to a specific provider using
the `provider/model` syntax, non-matching providers still received the full
pinned string as their model name — sending `"groq/openai/gpt-oss-20b"` to
OpenRouter as if it were an OpenRouter model ID.

This meant every request with a pinned model hit **every provider** instead of
only the pinned one. The wrong providers returned 400 errors (invalid model),
which were treated as retryable failures, causing unnecessary API calls and
potentially burning free-tier quota on providers the user never intended to use.

## Timeline

- **2026-08-21 ~10:30 UTC** — Merged loomweaver harness into flippy's repo as `src/loomweaver/`
- **2026-08-21 ~10:35 UTC** — Ran live smoke test: `route(messages, model='groq/openai/gpt-oss-20b')`
- **2026-08-21 ~10:36 UTC** — Noticed response came back with `provider: None` and `ok: False`
- **2026-08-21 ~10:38 UTC** — Traced through the pinning logic; found the inversion
- **2026-08-21 ~10:42 UTC** — Fixed and pushed; all 44 tests green

## Root cause

The original pinning logic had two conditions that were evaluated independently:

```python
# BROKEN:
m = model if (not model.startswith(prov["name"] + "/")) else None
if "/" in model and model.split("/")[0] == prov["name"]:
    m = model.split("/", 1)[1]
```

The first line said "use the full model string unless it starts with THIS provider's name."
For OpenRouter (name="openrouter") receiving `"groq/openai/gpt-oss-20b"`, it doesn't start
with `"openrouter/"`, so `m = "groq/openai/gpt-oss-20b"` — passed verbatim as OpenRouter's
model ID. The second condition then correctly stripped the prefix for groq, but openrouter
had already been called with the wrong string.

## Fix

```python
# FIXED: skip non-matching providers entirely
if any(model == x for x in prov["models"]):
    m = model  # exact match (handles slash-containing IDs like openai/gpt-oss-20b)
elif "/" in model and model.split("/", 1)[0] == prov["name"]:
    m = model.split("/", 1)[1]  # strip provider prefix
else:
    continue  # not for this provider — skip without calling
```

Non-matching providers are now skipped with `continue`, never contacted.
Model IDs containing slashes are matched exactly against the registry before
the prefix-strip fallback.

## What made this hard to catch

1. **Default routing worked fine** — without pinning, every request succeeded via
   failover. The bug only manifested when someone used `provider/model` syntax.
2. **The error was silent** — OpenRouter returned a 400, which was treated as
   a non-retryable failure, so route() moved on. No crash, no log entry at the
   default verbosity.
3. **Tests didn't cover pinning** — the test suite tested happy-path routing
   and basic failover, but never exercised the `provider/model` syntax.

## What changed

| Change | Commit |
|---|---|
| Rewrote route() pinning logic | `c5b37d5` |
| Added 4 tests specifically for pinning modes | same |
| Added regression tests for scorer edge cases | same |
| Documented pinning behavior in docstring | same |

## Lessons

1. **Smoke-test every code path you ship**, especially optional parameters.
   The happy path will hide bugs that only trigger on specific argument shapes.
2. **Silent failures are worse than crashes.** A 400 from the wrong provider
   should have logged at WARN level. It didn't because there was no structured
   logging at the time.
3. **Write the test first when fixing a routing bug.** Four of the ten
   failure-injection tests exist specifically because this bug taught us what
   to look for.

---

## Post-mortem 2: The session-growth leak

**Date:** 2026-08-21
**Severity:** Medium (resource leak)
**Status:** Fixed

### Summary

Agent sessions grew unboundedly. Every `run()` call appended messages without
any trimming. After 50 runs on the same session, the message list was 50KB —
and would keep growing until the context window exceeded the LLM's limit,
causing increasingly expensive and eventually failing API calls.

### Discovery

Noticed during a manual inspection: `sessions/default.json` was unexpectedly large.
No test covered repeated invocations on the same session.

### Fix

Added `MAX_SESSION_MESSAGES = 40` constant. After each run, trim to system
prompt + most recent messages. Preserves context while bounding growth.

### Why it matters

This is the kind of bug that doesn't show up in testing — it shows up after
weeks of production use when costs spike and responses degrade. The fix is
five lines. Finding it required thinking about the system over time, not
just per-request.

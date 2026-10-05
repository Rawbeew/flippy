"""hub.py — the bridge between the litellm front door and the loomweaver engine.

`aihub` (and optionally `server.py`) route through litellm. Everything that
makes flippy production-shaped — multi-key rotation, the semantic response
cache, the quota ledger, usage analytics, the learning memory — lives in
loomweaver. Left to themselves the two halves duplicate nothing and share
nothing: litellm would balance across providers while ignoring that a
provider's keys are exhausted, and every call would bypass the cache and the
quota ledger entirely.

This module is that join. It exposes the engine as a small, dependency-free
surface so a litellm caller can:

    hub.key_deployments(provider)   -> one deployment per LIVE key (rotation)
    hub.cache_get(messages)         -> cached response or None
    hub.quota_check(provider)       -> (allowed, reason)
    hub.record_outcome(...)         -> usage + learning + quota, in one call
    hub.cache_put(messages, text)   -> store for the next near-duplicate

Every function degrades to a no-op if the underlying service is unavailable or
raises. A routing front door must never fail because its telemetry did.
"""
from __future__ import annotations

import time

from . import learning
from . import key_rotation as _kr
from . import quota_ledger as _ql
from . import semantic_cache as _sc
from . import usage as _usage


# ---------------------------------------------------------------- key rotation

def live_keys(provider_name, keys):
    """The (index, key) pairs for `provider_name` that are usable right now.

    This is what makes a comma-separated key list actually rotate: a caller
    that only ever reads `provider["key"]` pins itself to the first key and
    burns its quota while the others sit idle.

    Note the ordering: `RotationState.live_pairs` yields **(key, index)** pairs
    in rotation order, and this function deliberately returns the more natural
    **(index, key)** so callers can index metadata by position.
    """
    keys = [k for k in (keys or []) if k]
    if not keys:
        return []
    state = None
    try:
        state = _kr.get_state()
    except Exception:
        state = None
    if state is None:
        return list(enumerate(keys))
    pairs = state.live_pairs(provider_name, keys)  # [(key, index), ...]
    # No blanket except here: a shape change in live_pairs must surface as a
    # real error rather than silently degrade to "every key is live".
    return [(idx, key) for key, idx in pairs]


def key_deployments(provider, litellm_model_fn):
    """Expand one registry provider into one litellm deployment per live key.

    `litellm_model_fn(model) -> str` maps a friendly model id to the
    litellm-prefixed name (e.g. "groq/..." vs "openai/..."), because that
    prefix differs per provider and the caller owns that knowledge.
    """
    name = provider["name"]
    base = provider.get("litellm_base")
    out = []
    for idx, key in live_keys(name, provider.get("keys") or [provider.get("key")]):
        for model in provider.get("models") or []:
            out.append({
                "model_name": model,
                "litellm_params": {
                    "model": litellm_model_fn(model),
                    "api_key": key,
                    **({"api_base": base} if base else {}),
                },
                # litellm groups deployments by model_name; tagging the provider
                # and key index keeps cooldowns and logs attributable.
                "metadata": {"provider": name, "key_index": idx},
            })
    return out


def note_key_failure(provider_name, key_index, status=None, error=None):
    """Feed a failed key back into the rotation state.

    401/403 retire the key permanently; 429 cools it down. Without this, a dead
    key stays in the pool and every request pays for discovering it again.
    """
    try:
        state = _kr.get_state()
        if state is None:
            return
        if status in (401, 403):
            state.mark_dead(provider_name, key_index,
                            f"http {status}" if status else str(error or "")[:120])
        elif status == 429:
            state.mark_exhausted(provider_name, key_index,
                                 retry_after=_kr.parse_retry_after(
                                     {"status": status, "error": error}
                                     if error else {"status": status}))
    except Exception:
        pass


# ---------------------------------------------------------------- semantic cache

def cache_enabled(messages=None):
    try:
        return bool(_sc.cache_enabled()) and not (
            messages is not None and _sc.is_stateful(messages))
    except Exception:
        return False


def cache_get(messages):
    """A cached response for a near-duplicate prompt, or None."""
    if not cache_enabled(messages):
        return None
    try:
        hit = _sc.get_cache().lookup(messages)
        if not hit:
            return None
        _usage.record("cache", "", True, 0.0, cached=True)
        return {"text": hit["response"], "similarity": hit["similarity"],
                "exact": hit["exact"], "cached": True}
    except Exception:
        return None


def cache_put(messages, text, model_tag=""):
    if not text or not cache_enabled(messages):
        return
    try:
        _sc.get_cache().store(messages, text, model_tag=model_tag)
    except Exception:
        pass


# ---------------------------------------------------------------- quota ledger

def quota_check(provider_name):
    """(allowed, reason). An exhausted provider is skipped before it is called."""
    try:
        ledger = _ql.get_ledger()
        allowed, reason = ledger.check_quota(provider_name)
        if allowed:
            ledger.record_request(provider_name)
        return bool(allowed), reason
    except Exception:
        return True, ""  # never block a request because the ledger is down


def quota_result(provider_name, status):
    try:
        _ql.get_ledger().record_result(provider_name, status)
    except Exception:
        pass


# ---------------------------------------------------------------- one-call sink

def record_outcome(provider_name, model, ok, latency_s, usage=None, cached=False,
                   status=None, key_index=None, goal=None):
    """Persist everything a call taught us, in one place.

    Usage analytics, the quota ledger, and the learning memory all need the
    same three facts (who answered, how long, did it work). Callers should not
    have to remember to tell each of them.
    """
    try:
        _usage.record(provider_name, model or "", bool(ok), latency_s or 0.0,
                      cached=bool(cached), usage=usage or {})
    except Exception:
        pass
    if status is not None:
        quota_result(provider_name, status)
    if not ok and key_index is not None:
        note_key_failure(provider_name, key_index, status=status)
    # The goal text is what makes the memory useful: vocabulary, similarity
    # retrieval and self-correction all key off it. Passing "" here would
    # silently reduce the learning layer to a bare success counter.
    if goal is not None:
        try:
            learning.record_route(goal, provider_name, model or "", bool(ok),
                                  latency_s or 0.0, 1, cached=bool(cached))
        except Exception:
            pass


def timed():
    """Monotonic start time for latency measurement."""
    return time.monotonic()


def since(start):
    return max(0.0, time.monotonic() - start)


# ---------------------------------------------------------------- introspection

def services():
    """Which engine services are actually reachable. Used by --health."""
    out = {}
    for label, probe in (
        ("semantic_cache", lambda: _sc.cache_enabled()),
        ("quota_ledger", lambda: _ql.get_ledger() is not None),
        ("key_rotation", lambda: _kr.get_state() is not None),
        ("usage", lambda: _usage.get_db() is not None),
        ("learning", lambda: learning.learning_enabled()),
    ):
        try:
            out[label] = bool(probe())
        except Exception:
            out[label] = False
    return out

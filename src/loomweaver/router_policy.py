"""router_policy.py — adaptive provider ordering + retry policy (production-shaped routing).

Two pieces, both stdlib-only:

1. ProviderStats / adaptive ordering
   The router used to try providers in static registry order on every call.
   Production routers learn: a provider that is fast and healthy should be
   tried first; one in cooldown or erroring should sink. We keep an EWMA of
   latency and a success EWMA per provider, plus a cooldown-until timestamp,
   and order by score = success_ewma / max(latency_ewma, floor).

2. RetryPolicy
   Retryable failures (429, 5xx, timeouts) get limited in-provider retries
   with exponential backoff + jitter before the router moves on. Attempts
   are capped and every attempt is emitted as an event so the run log shows
   the full decision trail.

State lives in-process (a dict guarded by a lock). No persistence needed:
the quota_ledger already tracks daily windows; this is sub-hour adaptation.

The try/except import dance mirrors core.py: modules must import cleanly
both as `python -m src.loomweaver` (CI) and `python -m loomweaver` (src/).
"""
from __future__ import annotations

import os
import random
import re
import threading
import time


# ---------------------------------------------------------------- stats

class ProviderStats:
    """EWMA latency/success + cooldown state for one provider."""

    def __init__(self, alpha: float = 0.3, latency_floor: float = 0.25):
        self.alpha = alpha
        self.latency_floor = latency_floor
        self.latency_ewma = None          # seconds; None = no data yet
        self.success_ewma = 1.0            # optimistic prior
        self.cooldown_until = 0.0
        self.calls = 0
        self.errors = 0

    def record(self, ok: bool, latency: float, cooldown_s: float = 0.0):
        self.calls += 1
        if not ok:
            self.errors += 1
        a = self.alpha
        if self.latency_ewma is None:
            self.latency_ewma = latency
        else:
            self.latency_ewma = a * latency + (1 - a) * self.latency_ewma
        self.success_ewma = a * (1.0 if ok else 0.0) + (1 - a) * self.success_ewma
        if cooldown_s:
            self.cooldown_until = time.time() + cooldown_s

    def score(self) -> float:
        """Higher = try earlier. Success weighted by inverse latency."""
        if time.time() < self.cooldown_until:
            return -1.0  # in local cooldown: below every live provider
        lat = max(self.latency_ewma or self.latency_floor, self.latency_floor)
        return self.success_ewma / lat


class RouterPolicy:
    """Adaptive ordering + retry decisions for route()."""

    def __init__(self, alpha: float = 0.3):
        self._lock = threading.Lock()
        self._stats: dict[str, ProviderStats] = {}
        self.alpha = alpha
        # provider name -> consecutive failure count (for backoff indexing)
        self._fail_streaks: dict[str, int] = {}

    # ------------------------------------------------------------ records
    def note_result(self, provider: str, ok: bool, latency: float,
                    status: int | None = None):
        """Record a call outcome and derive a local cooldown when warranted.

        Local cooldowns are SHORT (seconds): the quota ledger owns the
        long 429 cooldowns; this handles 'just failed twice in a row'
        style hysteresis so an immediate follow-up request skips the
        freshly-broken provider.
        """
        cooldown = 0.0
        with self._lock:
            st = self._stats.setdefault(provider, ProviderStats(alpha=self.alpha))
            if ok:
                self._fail_streaks[provider] = 0
                st.record(ok=True, latency=latency)
            else:
                streak = self._fail_streaks.get(provider, 0) + 1
                self._fail_streaks[provider] = streak
                # exponential hysteresis: 5s, 10s, 20s, 40s (capped 60s)
                cooldown = min(5 * (2 ** min(streak - 1, 4)), 60)
                st.record(ok=False, latency=latency, cooldown_s=cooldown)
        return cooldown

    def note_dead(self, provider: str, seconds: float):
        """External signal (quota ledger cooldown) — deprioritize harder."""
        with self._lock:
            st = self._stats.setdefault(provider, ProviderStats(alpha=self.alpha))
            st.cooldown_until = time.time() + seconds

    # ------------------------------------------------------------ ordering
    def order(self, providers: list[dict]) -> list[dict]:
        """Stable adaptive order: registry order, then re-sorted by score desc.

        Registry order (primary first) is the tie-break baseline; scores only
        override it once we have data. Providers in cooldown sink to the end
        in registry order.
        """
        with self._lock:
            snapshot = {name: st.score() for name, st in self._stats.items()}
        idx = {p["name"]: i for i, p in enumerate(providers)}

        def key(p):
            s = snapshot.get(p["name"], 0.0)
            if s < 0:
                # in cooldown: sink, preserving relative registry order
                return (1, idx[p["name"]], 0.0)
            return (0, -s, idx[p["name"]])

        return sorted(providers, key=key)

    # ------------------------------------------------------------ introspection
    def snapshot(self) -> dict:
        with self._lock:
            out = {}
            for name, st in self._stats.items():
                out[name] = {
                    "latency_ewma_s": round(st.latency_ewma, 3) if st.latency_ewma is not None else None,
                    "success_ewma": round(st.success_ewma, 3),
                    "calls": st.calls,
                    "errors": st.errors,
                    "cooldown_active": time.time() < st.cooldown_until,
                    "score": round(st.score(), 3),
                }
            return out


# ---------------------------------------------------------------- retry policy

class RetryPolicy:
    """In-provider retry with capped exponential backoff + jitter.

    A 429 from a provider is a rate limit: the quota ledger already applies
    escalating cooldowns (300s/900s/3600s) on 429, so retrying in-provider
    only burns quota. 429s are therefore NOT retried here — the router
    fails over instead. 5xx and network-level failures (no status) ARE
    retried: those are transient and frequently succeed on a second try.

      - only retryable failures retry (route() decides retryability)
      - max N attempts per provider, total budget capped
      - backoff = base * 2^attempt + jitter, capped at max_backoff
      - respect Retry-After when the provider sends it (caller passes it in)
      - base_s overridable via LOOMWEAVER_RETRY_BASE_S (tests set it near 0)
    """

    def __init__(self, max_attempts: int = 3,
                 base_s: float | None = None,
                 max_backoff_s: float = 30.0):
        self.max_attempts = max_attempts
        self.base_s = base_s if base_s is not None else float(
            os.environ.get("LOOMWEAVER_RETRY_BASE_S", "1.0"))
        self.max_backoff_s = max_backoff_s

    def should_retry(self, attempt: int, retryable: bool, deadline: float | None,
                     status: int | None = None) -> bool:
        if not retryable:
            return False
        if status == 429:
            return False  # quota ledger owns 429 cooldowns; retrying burns quota
        if attempt >= self.max_attempts:
            return False
        if deadline is not None and time.time() >= deadline:
            return False
        return True

    def backoff_s(self, attempt: int, retry_after: float | None = None) -> float:
        if retry_after is not None and retry_after > 0:
            return min(retry_after, self.max_backoff_s)
        d = self.base_s * (2 ** max(0, attempt - 1)) + random.uniform(0, 0.5)
        return min(d, self.max_backoff_s)


# module-level shared instances (cheap, process-scoped) ------------------------

_default_policy = None
_policy_lock = threading.Lock()


def get_policy() -> RouterPolicy:
    global _default_policy
    with _policy_lock:
        if _default_policy is None:
            _default_policy = RouterPolicy()
        return _default_policy


def get_retry() -> RetryPolicy:
    return RetryPolicy()


def cooldown_seconds_from_reason(reason: str) -> float:
    """Extract the wait time from a quota-ledger skip reason.

    check_quota() returns reasons like 'cooldown active for 300s (recent
    rate-limit)'. Route() uses this to mirror ledger cooldowns into the
    adaptive ordering so freshly-cooled providers sink without an extra
    DB read. Daily-limit reasons yield 0 (handled by the ledger itself).
    """
    m = re.search(r"for (\d+)s", reason or "")
    return float(m.group(1)) if m else 0.0

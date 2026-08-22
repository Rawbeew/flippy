"""usage.py — Usage Dashboard: per-provider stats from a SQLite event log.

Table:
    usage_events(id INTEGER PRIMARY KEY AUTOINCREMENT, ts REAL, provider TEXT,
                 model TEXT, ok INT, latency_s REAL, cached INT,
                 tokens_in INT, tokens_out INT)

API
---
- record(provider, model, ok, latency_s, cached=False, usage=None)
- summary(hours=24) -> per-provider stats + totals + cache_savings_estimate
- render_text(summary) -> pretty CLI table
- render_json(summary) -> JSON-ready dict for the HTTP endpoint

Stdlib only (sqlite3). Degrades gracefully when quota_ledger / semantic_cache
are absent (optional imports in try/except). record() never raises — usage
tracking must never break routing.
"""
import os
import sqlite3
import threading
import time

_SCHEMA = """
CREATE TABLE IF NOT EXISTS usage_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    provider TEXT NOT NULL,
    model TEXT NOT NULL DEFAULT '',
    ok INTEGER NOT NULL DEFAULT 0,
    latency_s REAL NOT NULL DEFAULT 0,
    cached INTEGER NOT NULL DEFAULT 0,
    tokens_in INTEGER NOT NULL DEFAULT 0,
    tokens_out INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_usage_ts ON usage_events(ts);
"""


def _default_db_path():
    return os.environ.get(
        "LOOMWEAVER_USAGE_DB",
        os.path.join(os.path.dirname(__file__), "..", "..", "runs", "usage.db"),
    )


class UsageDB:
    """Thread-safe SQLite-backed usage event log + aggregator."""

    def __init__(self, db_path=None):
        self.db_path = db_path or _default_db_path()
        d = os.path.dirname(os.path.abspath(self.db_path))
        if d:
            os.makedirs(d, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------ write
    def record(self, provider, model, ok, latency_s, cached=False, usage=None):
        """Log one call. Never raises (usage tracking must not break routing)."""
        try:
            usage = usage or {}
            tin = int(usage.get("prompt_tokens") or 0)
            tout = int(usage.get("completion_tokens") or 0)
            now = time.time()
            with self._lock, self._conn() as c:
                c.execute(
                    "INSERT INTO usage_events(ts, provider, model, ok, latency_s,"
                    " cached, tokens_in, tokens_out) VALUES (?,?,?,?,?,?,?,?)",
                    (now, str(provider), str(model or ""), 1 if ok else 0,
                     float(latency_s or 0), 1 if cached else 0, tin, tout))
        except Exception:
            pass

    # ------------------------------------------------------------ read
    def summary(self, hours=24, now=None):
        """Aggregate the last `hours` of events per provider + totals."""
        cutoff = (now or time.time()) - float(hours) * 3600.0
        provs = {}
        with self._conn() as c:
            rows = c.execute(
                "SELECT provider, ok, latency_s, cached, tokens_in, tokens_out "
                "FROM usage_events WHERE ts >= ? ORDER BY ts", (cutoff,)).fetchall()
        for r in rows:
            p = provs.setdefault(r["provider"], {
                "calls": 0, "errors": 0, "cache_hits": 0,
                "latency_sum": 0.0, "latency_n": 0,
                "tokens_in": 0, "tokens_out": 0,
            })
            p["calls"] += 1
            if not r["ok"]:
                p["errors"] += 1
            if r["cached"]:
                p["cache_hits"] += 1
            if r["ok"] and not r["cached"] and r["latency_s"] > 0:
                p["latency_sum"] += r["latency_s"]
                p["latency_n"] += 1
            p["tokens_in"] += r["tokens_in"]
            p["tokens_out"] += r["tokens_out"]

        out = {}
        for name, p in provs.items():
            avg_lat = (p["latency_sum"] / p["latency_n"]) if p["latency_n"] else 0.0
            total_tokens = p["tokens_in"] + p["tokens_out"]
            out[name] = {
                "calls": p["calls"],
                "errors": p["errors"],
                "cache_hits": p["cache_hits"],
                "avg_latency": round(avg_lat, 4),
                "total_tokens": total_tokens,
                "tokens_in": p["tokens_in"],
                "tokens_out": p["tokens_out"],
            }

        total_calls = sum(v["calls"] for v in out.values())
        total_errors = sum(v["errors"] for v in out.values())
        total_cache_hits = sum(v["cache_hits"] for v in out.values())
        total_tokens = sum(v["total_tokens"] for v in out.values())
        # Latency avoided: each cache hit skips one real provider call.
        real_avgs = [v["avg_latency"] for v in out.values() if v["avg_latency"] > 0]
        global_avg = (sum(real_avgs) / len(real_avgs)) if real_avgs else 0.0
        return {
            "window_hours": hours,
            "providers": out,
            "totals": {
                "calls": total_calls,
                "errors": total_errors,
                "cache_hits": total_cache_hits,
                "total_tokens": total_tokens,
            },
            "cache_savings_estimate": round(total_cache_hits * global_avg, 4),
            "quota": self._quota_state(),
        }

    def _quota_state(self):
        """Consume quota_ledger data if available; empty dict otherwise."""
        try:
            from . import quota_ledger as _ql
            return _ql.get_ledger().get_status()
        except Exception:
            try:
                import quota_ledger as _ql2  # top-level fallback
                return _ql2.get_ledger().get_status()
            except Exception:
                return {}


# Module-level shared instance + convenience wrappers -------------------------

_shared = None
_shared_lock = threading.Lock()


def get_db(db_path=None):
    global _shared
    with _shared_lock:
        if _shared is None or db_path is not None:
            _shared = UsageDB(db_path=db_path)
        return _shared


def record(provider, model, ok, latency_s, cached=False, usage=None, db_path=None):
    return get_db(db_path).record(provider, model, ok, latency_s,
                                  cached=cached, usage=usage)


def summary(hours=24, db_path=None):
    return get_db(db_path).summary(hours=hours)


def render_json(s):
    """JSON-ready dict for GET /usage (summary is already JSON-safe)."""
    return s


def render_text(s):
    """Pretty CLI table for `python -m loomweaver usage`."""
    lines = []
    w = s.get("window_hours", 24)
    lines.append(f"Usage Dashboard — last {w}h")
    lines.append("-" * 78)
    hdr = f"{'provider':<16}{'calls':>7}{'errors':>8}{'cache':>7}{'avg_lat':>9}{'tokens':>9}"
    lines.append(hdr)
    lines.append("-" * 78)
    provs = s.get("providers", {})
    if not provs:
        lines.append("(no usage events recorded)")
    for name in sorted(provs):
        v = provs[name]
        lines.append(f"{name:<16}{v['calls']:>7}{v['errors']:>8}{v['cache_hits']:>7}"
                     f"{v['avg_latency']:>9.3f}{v['total_tokens']:>9}")
    t = s.get("totals", {})
    lines.append("-" * 78)
    lines.append(f"{'TOTAL':<16}{t.get('calls', 0):>7}{t.get('errors', 0):>8}"
                 f"{t.get('cache_hits', 0):>7}{'':>9}{t.get('total_tokens', 0):>9}")
    lines.append(f"cache savings estimate: {s.get('cache_savings_estimate', 0):.3f}s "
                 f"(hits × avg latency avoided)")
    quota = s.get("quota") or {}
    if quota:
        lines.append("")
        lines.append("Quota state:")
        for name in sorted(quota):
            q = quota[name]
            cd = " [COOLDOWN]" if q.get("in_cooldown") else ""
            lines.append(f"  {name:<16}{q.get('requests_today', 0):>6} today"
                         f" / {q.get('daily_limit', '?')}{cd}")
    return "\n".join(lines)

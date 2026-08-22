"""quota_ledger.py — track per-provider free-tier quotas; route away BEFORE 429.

SQLite-backed ledger of per-provider request counts and cooldowns:
    quota_state(provider TEXT PRIMARY KEY, requests_today INT,
                window_start TEXT, cooldown_until TEXT,
                last_status TEXT, fail_streak INT)

Design notes
------------
- Stdlib only (sqlite3 from the standard library).
- Limits come from PROVIDER_LIMITS defaults, overridable via the
  LOOMWEAVER_LIMITS_JSON env var: {"groq": 14000, "nvidia": 40, ...}
- record_result() on HTTP 429 puts the provider in an escalating cooldown
  (5 min → 15 min → 1 h → 1 h...) keyed on fail_streak.
- daily_reset() rolls counters at UTC midnight (called opportunistically).
"""
import json
import os
import sqlite3
import threading
import time
from datetime import datetime, timezone

# Sensible free-tier defaults (requests per day unless suffixed _per_minute).
# openrouter varies by model/credits — conservative default.
PROVIDER_LIMITS = {
    "openrouter": 200,          # varies; conservative daily default
    "freeinference": 500,       # unknown; conservative daily default
    "cloudflare": 10000,        # neurons-based ~10k/day free tier
    "nvidia": 57600,            # 40/min -> ~57.6k/day ceiling; tracked daily
    "groq": 14400,              # 14k/day free tier
}

# Escalating backoff for consecutive 429s (seconds): 5min, 15min, 1h...
BACKOFF_STEPS = [300, 900, 3600]
BACKOFF_MAX = 3600

_SCHEMA = """
CREATE TABLE IF NOT EXISTS quota_state (
    provider TEXT PRIMARY KEY,
    requests_today INTEGER NOT NULL DEFAULT 0,
    window_start TEXT NOT NULL DEFAULT '',
    cooldown_until TEXT NOT NULL DEFAULT '',
    last_status TEXT NOT NULL DEFAULT '',
    fail_streak INTEGER NOT NULL DEFAULT 0
);
"""


def _default_db_path():
    return os.environ.get(
        "LOOMWEAVER_QUOTA_DB",
        os.path.join(os.path.dirname(__file__), "..", "..", "runs", "quota_ledger.db"),
    )


def _limits():
    raw = os.environ.get("LOOMWEAVER_LIMITS_JSON")
    if raw:
        try:
            return {str(k): int(v) for k, v in json.loads(raw).items()}
        except (ValueError, TypeError):
            pass
    return dict(PROVIDER_LIMITS)


class QuotaLedger:
    """Thread-safe SQLite-backed per-provider quota tracker."""

    def __init__(self, db_path=None, limits=None):
        self.db_path = db_path or _default_db_path()
        d = os.path.dirname(os.path.abspath(self.db_path))
        if d:
            os.makedirs(d, exist_ok=True)
        self.limits = limits if limits is not None else _limits()
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(_SCHEMA)

    def _conn(self):
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        return conn

    # ------------------------------------------------------------ helpers

    @staticmethod
    def _utcnow():
        return datetime.now(timezone.utc)

    @staticmethod
    def _iso(dt):
        return dt.strftime("%Y-%m-%dT%H:%M:%SZ")

    @staticmethod
    def _parse_iso(s):
        if not s:
            return None
        try:
            return datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
        except ValueError:
            return None

    def _row(self, c, provider):
        c.execute(
            "INSERT OR IGNORE INTO quota_state(provider) VALUES (?)", (provider,))
        return c.execute(
            "SELECT * FROM quota_state WHERE provider = ?", (provider,)).fetchone()

    # ------------------------------------------------------------ public API

    def check_quota(self, provider):
        """Return (allowed: bool, reason: str) for a would-be request."""
        now = self._utcnow()
        with self._lock, self._conn() as c:
            self._maybe_reset_row(c, provider, now)
            row = self._row(c, provider)
            cd = self._parse_iso(row["cooldown_until"])
            if cd and cd > now:
                wait = int((cd - now).total_seconds())
                return False, f"cooldown active for {wait}s (recent rate-limit)"
            limit = self.limits.get(provider)
            if limit is not None and row["requests_today"] >= limit:
                return False, (f"daily limit reached ({row['requests_today']}/{limit}); "
                               f"resets at UTC midnight")
        return True, "ok"

    def record_request(self, provider):
        """Count one outgoing request against today's total."""
        now = self._utcnow()
        with self._lock, self._conn() as c:
            self._maybe_reset_row(c, provider, now)
            self._row(c, provider)
            c.execute(
                "UPDATE quota_state SET requests_today = requests_today + 1 "
                "WHERE provider = ?", (provider,))

    def record_result(self, provider, status_code):
        """Update ledger from an HTTP status. On 429 apply escalating cooldown;
        on success clear fail streak."""
        now = self._utcnow()
        with self._lock, self._conn() as c:
            self._maybe_reset_row(c, provider, now)
            self._row(c, provider)
            if status_code == 429:
                streak_row = c.execute(
                    "SELECT fail_streak FROM quota_state WHERE provider=?",
                    (provider,)).fetchone()
                streak = (streak_row["fail_streak"] or 0) + 1
                step = BACKOFF_STEPS[min(streak - 1, len(BACKOFF_STEPS) - 1)]
                until = self._iso(now.fromtimestamp(
                    time.time() + min(step, BACKOFF_MAX), tz=timezone.utc))
                c.execute(
                    "UPDATE quota_state SET cooldown_until=?, last_status=?, "
                    "fail_streak=? WHERE provider=?",
                    (until, str(status_code), streak, provider))
            elif status_code == 200:
                c.execute(
                    "UPDATE quota_state SET last_status=?, fail_streak=0, "
                    "cooldown_until='' WHERE provider=?",
                    (str(status_code), provider))
            else:
                c.execute(
                    "UPDATE quota_state SET last_status=? WHERE provider=?",
                    (str(status_code), provider))

    def daily_reset(self, force=False):
        """Roll requests_today to 0 once the UTC date changes (or force=True)."""
        now = self._utcnow()
        today = now.strftime("%Y-%m-%d")
        with self._lock, self._conn() as c:
            rows = c.execute("SELECT provider, window_start FROM quota_state").fetchall()
            for r in rows:
                if force or (r["window_start"] or "")[:10] != today:
                    c.execute(
                        "UPDATE quota_state SET requests_today=0, window_start=?, "
                        "cooldown_until='', fail_streak=0 WHERE provider=?",
                        (self._iso(now), r["provider"]))

    def _maybe_reset_row(self, c, provider, now):
        """Per-provider lazy UTC-midnight roll."""
        row = self._row(c, provider)
        ws = (row["window_start"] or "")[:10]
        if ws != now.strftime("%Y-%m-%d"):
            c.execute(
                "UPDATE quota_state SET requests_today=0, window_start=?, "
                "cooldown_until='', fail_streak=0 WHERE provider=?",
                (self._iso(now), provider))

    def get_status(self):
        """JSON-ready snapshot of the whole ledger."""
        self.daily_reset()
        limits = self.limits
        out = {}
        now = self._utcnow()
        with self._lock, self._conn() as c:
            for r in c.execute("SELECT * FROM quota_state ORDER BY provider"):
                limit = limits.get(r["provider"])
                remaining = None
                if limit is not None:
                    remaining = max(0, limit - r["requests_today"])
                cd = self._parse_iso(r["cooldown_until"])
                out[r["provider"]] = {
                    "requests_today": r["requests_today"],
                    "daily_limit": limit,
                    "remaining": remaining,
                    "window_start": r["window_start"],
                    "cooldown_until": r["cooldown_until"] or None,
                    "in_cooldown": bool(cd and cd > now),
                    "last_status": r["last_status"] or None,
                    "fail_streak": r["fail_streak"],
                }
        return out


# Module-level shared instance + convenience wrappers -------------------------

_shared = None
_shared_lock = threading.Lock()


def get_ledger(db_path=None):
    global _shared
    with _shared_lock:
        if _shared is None or db_path is not None:
            _shared = QuotaLedger(db_path=db_path)
        return _shared


def check_quota(provider, db_path=None):
    return get_ledger(db_path).check_quota(provider)


def record_result(provider, status_code, db_path=None):
    return get_ledger(db_path).record_result(provider, status_code)


def record_request(provider, db_path=None):
    return get_ledger(db_path).record_request(provider)


def get_quota_status(db_path=None):
    return get_ledger(db_path).get_status()

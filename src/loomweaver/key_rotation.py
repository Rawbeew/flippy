"""key_rotation.py — multi-key rotation per provider, persisted to SQLite.

Each provider can carry several API keys (comma-separated env values, parsed
by flippy_providers.split_keys). This module tracks which keys are usable:

  - mark_dead()     : permanent skip (401/403 — revoked or unauthorized)
  - mark_exhausted(): temporary skip until cooldown_until (429 quota)
  - live_pairs()    : (key, index) pairs still usable, in rotation order

All keys dead -> provider is unavailable (live_pairs() == []).

State lives in one SQLite table so rotation survives process restarts.
Keys are never stored here — only indices — and are masked in any log output.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
import time

DEFAULT_DB = os.path.join(
    os.environ.get("LOOMWEAVER_KEYROTATION_DB")
    or os.path.join(os.path.dirname(__file__), "..", "..", "data", "key_rotation.db")
)
DEFAULT_COOLDOWN_S = 60.0

_state = None
_state_lock = threading.Lock()


def mask(key: str) -> str:
    """Mask a key for logs: first 6 chars only."""
    return (key[:6] + "...") if key else "(empty)"


class RotationState:
    """Per-provider key health, persisted to SQLite."""

    def __init__(self, db_path: str | None = None):
        self.db_path = db_path or DEFAULT_DB
        d = os.path.dirname(self.db_path)
        if d:
            os.makedirs(d, exist_ok=True)
        self._lock = threading.Lock()
        self._init_db()

    def _connect(self):
        con = sqlite3.connect(self.db_path)
        con.execute("""CREATE TABLE IF NOT EXISTS key_rotation(
            provider TEXT PRIMARY KEY,
            active_index INT NOT NULL DEFAULT 0,
            dead_keys TEXT NOT NULL DEFAULT '[]'
        )""")
        return con

    def _init_db(self):
        with self._lock:
            con = self._connect()
            con.commit()
            con.close()

    def _row(self, provider: str) -> tuple[int, list[int], list[dict]]:
        con = self._connect()
        try:
            cur = con.execute(
                "SELECT active_index, dead_keys FROM key_rotation WHERE provider=?",
                (provider,))
            row = cur.fetchone()
        finally:
            con.close()
        if not row:
            return 0, [], []
        try:
            dead = json.loads(row[1] or "[]")
        except (ValueError, TypeError):
            dead = []
        return int(row[0] or 0), dead if isinstance(dead, list) else [], []

    def _save(self, provider: str, active_index: int, dead: list[dict]):
        con = self._connect()
        try:
            con.execute(
                """INSERT INTO key_rotation(provider, active_index, dead_keys)
                   VALUES(?,?,?)
                   ON CONFLICT(provider) DO UPDATE SET
                     active_index=excluded.active_index,
                     dead_keys=excluded.dead_keys""",
                (provider, active_index, json.dumps(dead)))
            con.commit()
        finally:
            con.close()

    # ------------------------------------------------------------- mutations

    def mark_dead(self, provider: str, key_index: int, reason: str = ""):
        """Permanently skip a revoked/unauthorized key (401/403)."""
        with self._lock:
            _, dead, _ = self._row(provider)
            if not any(d.get("index") == key_index for d in dead):
                dead.append({"index": key_index, "reason": str(reason)[:200],
                             "at": round(time.time(), 3)})
            self._save(provider, 0, dead)

    def mark_exhausted(self, provider: str, key_index: int,
                       cooldown_until: float | None = None,
                       retry_after: float | None = None):
        """Temporarily skip a quota-exhausted key (429) until cooldown ends."""
        until = cooldown_until if cooldown_until is not None else (
            time.time() + (retry_after if retry_after and retry_after > 0
                           else DEFAULT_COOLDOWN_S))
        with self._lock:
            _, dead, _ = self._row(provider)
            dead = [d for d in dead if d.get("index") != key_index
                    or d.get("kind") != "exhausted"]
            dead.append({"index": key_index, "kind": "exhausted",
                         "until": round(until, 3), "at": round(time.time(), 3)})
            self._save(provider, 0, dead)

    # ------------------------------------------------------------- selection

    def live_pairs(self, provider: str, keys: list[str]) -> list[tuple[str, int]]:
        """Return (key, index) pairs for all currently usable keys.

        Rotation order starts at the persisted active_index so successive
        calls walk through keys rather than always hammering the first.
        """
        with self._lock:
            active, dead, _ = self._row(provider)
        now = time.time()
        dead_idx = {d["index"] for d in dead if d.get("kind") != "exhausted"}
        exhausted_idx = {d["index"] for d in dead
                         if d.get("kind") == "exhausted" and d.get("until", 0) > now}
        bad = dead_idx | exhausted_idx
        pairs = [(k, i) for i, k in enumerate(keys) if i not in bad]
        if not pairs or not (0 <= active < len(keys)):
            return pairs
        rot = active % len(pairs)
        return pairs[rot:] + pairs[:rot]

    def advance(self, provider: str, key_index: int):
        """Persist the index of the key that just succeeded as the new start."""
        with self._lock:
            self._save(provider, key_index, self._row(provider)[1])

    def reset(self, provider: str | None = None):
        """Forget all state (tests / manual recovery)."""
        with self._lock:
            con = self._connect()
            try:
                if provider is None:
                    con.execute("DELETE FROM key_rotation")
                else:
                    con.execute("DELETE FROM key_rotation WHERE provider=?", (provider,))
                con.commit()
            finally:
                con.close()


def get_state() -> RotationState:
    """Process-wide singleton (env LOOMWEAVER_KEYROTATION_DB overrides path)."""
    global _state
    with _state_lock:
        if _state is None:
            _state = RotationState()
        return _state


def set_state(state: RotationState | None):
    """Replace/reset the singleton (tests)."""
    global _state
    with _state_lock:
        _state = state


def parse_retry_after(result: dict) -> float | None:
    """Best-effort Retry-After extraction from a failed chat() result."""
    err = result.get("error") or ""
    import re
    m = re.search(r"retry[- ]after['\":\s]+([0-9.]+)", str(err), re.IGNORECASE)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            pass
    return None

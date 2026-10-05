"""semantic_cache.py — SQLite-backed response cache keyed by message similarity.

Addresses freellmapi issue: "Increase response-cache hit rate: param
normalization, SQLite persistence, streaming". Stdlib-only: no numpy, no
embedding APIs. Sparse word-count vectors stored as JSON blobs; TF-IDF-style
cosine similarity over the cached corpus.

Lookup order:
  1. Exact-match fast path (SHA-256 of normalized prompt)
  2. Similarity fallback (cosine >= threshold)

TTL: entries older than CACHE_TTL_HOURS (default 168 = 1 week) are ignored
and pruned on access.
"""
import json
import math
import os
import re
import sqlite3
import threading
import time
from contextlib import contextmanager

try:
    from . import security as _security  # noqa: F401  (guards stay importable)
except ImportError:
    pass

DEFAULT_THRESHOLD = float(os.environ.get("LOOMWEAVER_CACHE_THRESHOLD", "0.92"))
DEFAULT_TTL_HOURS = float(os.environ.get("LOOMWEAVER_CACHE_TTL_HOURS", "168"))


def cache_enabled():
    return os.environ.get("LOOMWEAVER_CACHE_ENABLED", "1") not in ("0", "false", "no")


def _db_path(db_path=None):
    return db_path or os.environ.get(
        "LOOMWEAVER_CACHE_DB",
        os.path.join(os.path.dirname(__file__), "..", "..", "runs", "semantic_cache.sqlite3"),
    )


# ---------------------------------------------------------------- normalization

_WS_RE = re.compile(r"\s+")
_PUNCT_RE = re.compile(r"[^\w\s]")


def normalize(text):
    """Lowercase, collapse whitespace, strip punctuation — the lexical key."""
    t = text.lower()
    t = _PUNCT_RE.sub(" ", t)
    return _WS_RE.sub(" ", t).strip()


def prompt_key(messages):
    """Normalized lexical key for the conversation's user-facing messages."""
    parts = []
    for m in messages:
        role = m.get("role", "")
        if role in ("system", "tool"):
            continue
        content = m.get("content", "")
        if isinstance(content, list):  # multimodal content blocks
            content = " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
        parts.append(f"{role}:{normalize(str(content))}")
    return "\n".join(parts)


def is_stateful(messages):
    """Skip caching when tools/system markers indicate statefulness."""
    # agent-loop markers: tool results / observations are written back to the
    # conversation as role "user" (see agent.py's `TOOL_RESULT {name}: {obs}`
    # and armada.py). A cached completion served for such a prefix could
    # terminate a live step with stale/fabricated context, so these must be
    # treated as stateful and never cached.
    _AGENT_MARKERS = ("TOOL_RESULT", "TOOL_CALL", "OBSERVATION")
    for m in messages:
        if m.get("role") == "system":
            content = str(m.get("content", ""))
            if any(k in content.lower() for k in ("session", "stateful", "conversation history", "remember")):
                return True
        if "tools" in m or "tool_calls" in m or m.get("role") == "tool":
            return True
        if m.get("role") == "user":
            content = m.get("content", "")
            if isinstance(content, list):  # multimodal content blocks
                content = " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict))
            if any(k in str(content).upper() for k in _AGENT_MARKERS):
                return True
    return False


# ---------------------------------------------------------------- vectors

def word_counts(text):
    counts = {}
    for w in normalize(text).split():
        counts[w] = counts.get(w, 0) + 1
    return counts


class SemanticCache:
    """SQLite-backed response cache.

    Thread-safety: a single long-lived sqlite3.Connection is bound to the thread
    that created it, so a process-wide singleton shared with a
    ThreadingHTTPServer raised `ProgrammingError: SQLite objects created in a
    thread can only be used in that same thread` on every request after the
    first. route() swallowed that with a bare `except Exception: pass`, so the
    cache was silently dead in exactly the deployment mode it is advertised
    for. Connections are now opened per operation and closed in a `finally`,
    matching quota_ledger.UsageDB / usage.UsageDB.
    """

    # SQLite's default SQLITE_MAX_VARIABLE_NUMBER; keep the prune IN-list under it.
    _MAX_PARAMS = 900

    def __init__(self, db_path=None, threshold=None, ttl_hours=None):
        self.db_path = _db_path(db_path)
        self.threshold = DEFAULT_THRESHOLD if threshold is None else threshold
        self.ttl_hours = DEFAULT_TTL_HOURS if ttl_hours is None else ttl_hours
        d = os.path.dirname(os.path.abspath(self.db_path))
        if d:
            os.makedirs(d, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            self._init_schema(c)

    @contextmanager
    def _conn(self):
        """Yield a fresh connection, commit on success, ALWAYS close."""
        conn = sqlite3.connect(self.db_path, timeout=10)
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            yield conn
            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass

    @staticmethod
    def _init_schema(c):
        c.execute("""CREATE TABLE IF NOT EXISTS cache_entries(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            prompt_norm TEXT NOT NULL,
            model_tag TEXT DEFAULT '',
            response TEXT,
            created_ts REAL,
            hit_count INTEGER DEFAULT 0)""")
        c.execute("CREATE INDEX IF NOT EXISTS idx_entries_key ON cache_entries(prompt_norm)")
        c.execute("""CREATE TABLE IF NOT EXISTS cache_vectors(
            entry_id INTEGER PRIMARY KEY,
            dim INTEGER,
            vector BLOB)""")

    # -- internals ---------------------------------------------------------

    def _prune_expired(self, c):
        cutoff = time.time() - self.ttl_hours * 3600
        rows = c.execute(
            "SELECT id FROM cache_entries WHERE created_ts < ?", (cutoff,)).fetchall()
        # Chunked: an unbounded IN (?,?...) list overruns SQLite's variable limit
        for i in range(0, len(rows), self._MAX_PARAMS):
            ids = [r[0] for r in rows[i:i + self._MAX_PARAMS]]
            qmarks = ",".join("?" * len(ids))
            c.execute(f"DELETE FROM cache_entries WHERE id IN ({qmarks})", ids)
            c.execute(f"DELETE FROM cache_vectors WHERE entry_id IN ({qmarks})", ids)

    def _idf(self, c):
        """Inverse document frequency over stored entries (smoothed)."""
        n_rows = c.execute("SELECT COUNT(*) FROM cache_entries").fetchone()[0]
        if n_rows == 0:
            return {}
        df = {}
        for (blob,) in c.execute("SELECT vector FROM cache_vectors"):
            for w in json.loads(bytes(blob).decode()):
                df[w] = df.get(w, 0) + 1
        return {w: math.log((1 + n_rows) / (1 + cnt)) + 1 for w, cnt in df.items()}

    @staticmethod
    def _cosine(a, b, idf):
        def weighted(v):
            tot = 0.0
            out = {}
            for w, cnt in v.items():
                weight = cnt * idf.get(w, 1.0)
                out[w] = weight
                tot += weight * weight
            return out, math.sqrt(tot) or 1.0

        wa, na = weighted(a)
        wb, nb = weighted(b)
        dot = sum(x * wb.get(w, 0.0) for w, x in wa.items())
        return dot / (na * nb)

    # -- public API --------------------------------------------------------

    def store(self, messages, response, model_tag=""):
        key = prompt_key(messages)
        vec = word_counts(" ".join(m.get("content", "") if isinstance(m.get("content"), str)
                                   else json.dumps(m.get("content")) for m in messages))
        now = time.time()
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO cache_entries(prompt_norm, model_tag, response, created_ts,"
                " hit_count) VALUES(?,?,?,?,0)", (key, model_tag, response, now))
            eid = cur.lastrowid
            blob = json.dumps(vec).encode()
            c.execute("INSERT INTO cache_vectors(entry_id, dim, vector) VALUES(?,?,?)",
                      (eid, len(vec), sqlite3.Binary(blob)))
        return eid

    def lookup(self, messages, threshold=None):
        """Cached response dict, or None. Exact key first, cosine second.

        (The LOOMWEAVER_CACHE_ENABLED gate lives in route(), not here, so the
        cache object stays directly usable.)
        """
        thr = self.threshold if threshold is None else threshold
        key = prompt_key(messages)
        qvec = word_counts(" ".join(
            m.get("content", "") if isinstance(m.get("content"), str)
            else json.dumps(m.get("content")) for m in messages))
        with self._lock, self._conn() as c:
            self._prune_expired(c)
            row = c.execute(
                "SELECT id, response FROM cache_entries WHERE prompt_norm = ?"
                " ORDER BY created_ts DESC LIMIT 1", (key,)).fetchone()
            if row:
                c.execute("UPDATE cache_entries SET hit_count = hit_count + 1"
                          " WHERE id = ?", (row[0],))
                return {"response": row[1], "similarity": 1.0,
                        "entry_id": row[0], "exact": True}
            idf = self._idf(c)
            best = None
            for eid, resp, blob in c.execute(
                    "SELECT e.id, e.response, v.vector FROM cache_entries e"
                    " JOIN cache_vectors v ON v.entry_id = e.id"):
                sim = self._cosine(qvec, json.loads(bytes(blob).decode()), idf)
                if sim >= thr and (best is None or sim > best["similarity"]):
                    best = {"response": resp, "similarity": round(sim, 4),
                            "entry_id": eid, "exact": False}
            if best:
                c.execute("UPDATE cache_entries SET hit_count = hit_count + 1"
                          " WHERE id = ?", (best["entry_id"],))
        return best

    def stats(self):
        with self._lock, self._conn() as c:
            n = c.execute("SELECT COUNT(*), COALESCE(SUM(hit_count),0)"
                          " FROM cache_entries").fetchone()
        return {"entries": n[0], "total_hits": n[1]}

    def close(self):
        """No persistent connection to close; kept for API compatibility."""
        return None


_default_cache = None
_default_cache_lock = threading.Lock()


def get_cache():
    """Process-wide singleton. Guarded: two threads racing here used to build
    two caches against the same file."""
    global _default_cache
    with _default_cache_lock:
        if _default_cache is None:
            _default_cache = SemanticCache()
        return _default_cache
_default_cache = None


def get_cache():
    global _default_cache
    if _default_cache is None:
        _default_cache = SemanticCache()
    return _default_cache

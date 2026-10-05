"""learning.py — flippy's memory: it gets better the more you use it.

Three loops, all local, all stdlib SQLite:

1. OUTCOME MEMORY — every routed call and every agent run is recorded with its
   goal, the provider that answered, latency, attempt count and the tools used.
   On startup these become priors for the adaptive router, so a fresh process
   does not have to relearn which provider is fast and which one lies.

2. USER PROFILE — inferred, never asked for: which provider/model actually
   works for this user, which tools their goals need, the vocabulary they
   reuse, their success rate. Injected into the agent's system prompt as a
   short, factual block, so the tenth run starts where the ninth one ended.

3. LESSONS (self-correction) — when a run fails, or when the operator says
   "no — when I ask for X, do Y", a lesson is stored against the shape of the
   goal that produced it. A new goal retrieves the most similar lessons by
   TF-IDF cosine and they enter the prompt as prior guidance. That is the
   self-correcting loop: the same mistake does not have to be made twice.

Nothing here phones home. The store is one SQLite file under runs/, the same
place the quota ledger and usage log already live.
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import threading
import time
from contextlib import contextmanager

from .semantic_cache import normalize, word_counts

_SCHEMA = """
CREATE TABLE IF NOT EXISTS interactions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL,
    goal TEXT NOT NULL DEFAULT '',
    goal_norm TEXT NOT NULL DEFAULT '',
    provider TEXT NOT NULL DEFAULT '',
    model TEXT NOT NULL DEFAULT '',
    ok INTEGER NOT NULL DEFAULT 0,
    latency_s REAL NOT NULL DEFAULT 0,
    attempts INTEGER NOT NULL DEFAULT 1,
    cached INTEGER NOT NULL DEFAULT 0,
    tools TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_learn_ts ON interactions(ts);
CREATE INDEX IF NOT EXISTS idx_learn_prov ON interactions(provider, ok);

CREATE TABLE IF NOT EXISTS lessons (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ts REAL NOT NULL,
    kind TEXT NOT NULL DEFAULT 'correction',
    trigger_norm TEXT NOT NULL DEFAULT '',
    text TEXT NOT NULL,
    weight REAL NOT NULL DEFAULT 1.0,
    hits INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS lesson_vectors (
    lesson_id INTEGER PRIMARY KEY,
    vector BLOB
);
"""

# A failure becomes a lesson only when it is informative, not merely noisy.
_UNINFORMATIVE_ERRORS = ("all providers failed", "no providers configured")


def _default_db_path():
    return os.environ.get(
        "LOOMWEAVER_LEARNING_DB",
        os.path.join(os.path.dirname(__file__), "..", "..", "runs", "learning.db"))


def goal_text(target):
    """Accept a plain goal string or an OpenAI-style message list."""
    if isinstance(target, str):
        return target
    if isinstance(target, (list, tuple)):
        users = [m.get("content", "") for m in target
                 if isinstance(m, dict) and m.get("role") == "user"
                 and isinstance(m.get("content"), str)]
        if users:
            return users[-1]
        return " ".join(str(m.get("content", "")) for m in target
                        if isinstance(m, dict))
    return str(target or "")


def learning_enabled():
    return os.environ.get("LOOMWEAVER_LEARNING_ENABLED", "1") not in ("0", "false", "no")


class LearningStore:
    """Thread-safe SQLite store for outcomes, profile and lessons."""

    def __init__(self, db_path=None):
        self.db_path = db_path or _default_db_path()
        d = os.path.dirname(os.path.abspath(self.db_path))
        if d:
            os.makedirs(d, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as c:
            c.executescript(_SCHEMA)

    @contextmanager
    def _conn(self):
        """Fresh connection per operation, closed in a `finally`.

        A long-lived connection would bind the store to its creating thread and
        break under ThreadingHTTPServer — the exact bug the semantic cache had.
        """
        conn = sqlite3.connect(self.db_path, timeout=10)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        finally:
            try:
                conn.close()
            except Exception:
                pass

    # ------------------------------------------------------------ outcomes

    def record_route(self, goal, provider, model, ok, latency_s=0.0,
                     attempts=1, cached=False):
        self._record("route", goal_text(goal), provider, model, ok, latency_s,
                     attempts, cached, "")

    def record_agent_run(self, goal, tools, ok, steps=0, provider="", model=""):
        self._record("agent", goal_text(goal), provider, model, ok, 0.0, steps,
                     0, ",".join(sorted(set(tools or []))))

    def _record(self, kind, goal, provider, model, ok, latency_s, attempts,
                cached, tools):
        if not learning_enabled():
            return
        try:
            with self._lock, self._conn() as c:
                c.execute(
                    "INSERT INTO interactions(ts, kind, goal, goal_norm, provider,"
                    " model, ok, latency_s, attempts, cached, tools)"
                    " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (time.time(), kind, str(goal or "")[:2000],
                     normalize(str(goal or "")), str(provider or ""),
                     str(model or ""), 1 if ok else 0, float(latency_s or 0.0),
                     int(attempts or 1), 1 if cached else 0, str(tools or "")))
        except Exception:
            pass  # learning must never break a request

    # ------------------------------------------------------------ lessons

    def add_lesson(self, text, trigger="", kind="correction", weight=1.0):
        """Store a lesson. `trigger` is the goal shape it applies to."""
        text = str(text or "").strip()
        if not text:
            return None
        vec = json.dumps(word_counts(trigger or text)).encode()
        with self._lock, self._conn() as c:
            cur = c.execute(
                "INSERT INTO lessons(ts, kind, trigger_norm, text, weight, hits)"
                " VALUES (?,?,?,?,?,0)",
                (time.time(), kind, normalize(trigger or text), text,
                 float(weight)))
            lid = cur.lastrowid
            c.execute("INSERT INTO lesson_vectors(lesson_id, vector) VALUES (?,?)",
                      (lid, sqlite3.Binary(vec)))
        return lid

    def note_failure(self, goal, error):
        """Derive a lesson from a failed run, unless the error is pure noise."""
        goal = goal_text(goal)
        err = str(error or "").strip()
        if not err or any(u in err.lower() for u in _UNINFORMATIVE_ERRORS):
            return None
        return self.add_lesson(
            f"A previous attempt at a goal like this failed with: {err[:200]}. "
            f"Check that precondition before repeating the same approach.",
            trigger=goal, kind="failure", weight=0.6)

    @staticmethod
    def _cosine(a, b, idf):
        def weighted(v):
            tot, out = 0.0, {}
            for w, cnt in v.items():
                wt = cnt * idf.get(w, 1.0)
                out[w] = wt
                tot += wt * wt
            return out, math.sqrt(tot) or 1.0

        wa, na = weighted(a)
        wb, nb = weighted(b)
        return sum(x * wb.get(w, 0.0) for w, x in wa.items()) / (na * nb)

    def lessons_for(self, goal, k=3, min_similarity=0.25):
        """Most similar lessons for a goal, by TF-IDF cosine."""
        qvec = word_counts(str(goal or ""))
        if not qvec:
            return []
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT l.id, l.text, l.kind, l.weight, v.vector FROM lessons l"
                " JOIN lesson_vectors v ON v.lesson_id = l.id").fetchall()
            if not rows:
                return []
            df = {}
            decoded = []
            for r in rows:
                vec = json.loads(bytes(r["vector"]).decode())
                decoded.append((r, vec))
                for w in vec:
                    df[w] = df.get(w, 0) + 1
            n = len(rows)
            idf = {w: math.log((1 + n) / (1 + cnt)) + 1 for w, cnt in df.items()}
            scored = [(self._cosine(qvec, vec, idf), r) for r, vec in decoded]
            scored = [(s, r) for s, r in scored if s >= min_similarity]
            scored.sort(key=lambda t: -(t[0] * float(t[1]["weight"])))
            picks = scored[:k]
            for _, r in picks:
                c.execute("UPDATE lessons SET hits = hits + 1 WHERE id = ?",
                          (r["id"],))
        return [{"id": r["id"], "text": r["text"], "kind": r["kind"],
                 "similarity": round(s, 4)} for s, r in picks]

    def forget(self, lesson_id=None):
        with self._lock, self._conn() as c:
            if lesson_id is None:
                c.execute("DELETE FROM lessons")
                c.execute("DELETE FROM lesson_vectors")
            else:
                c.execute("DELETE FROM lessons WHERE id = ?", (lesson_id,))
                c.execute("DELETE FROM lesson_vectors WHERE lesson_id = ?",
                          (lesson_id,))

    # ------------------------------------------------------------ profile

    def profile(self, hours=24 * 30):
        """Infer who this user is from what they have actually run."""
        cutoff = time.time() - float(hours) * 3600
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT kind, goal_norm, provider, model, ok, latency_s,"
                " attempts, cached, tools FROM interactions WHERE ts >= ?",
                (cutoff,)).fetchall()
            n_lessons = c.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]

        if not rows:
            return {"interactions": 0, "lessons": n_lessons, "cold_start": True}

        total = len(rows)
        oks = [r for r in rows if r["ok"]]
        by_prov, by_model, terms, tools = {}, {}, {}, {}
        for r in rows:
            if r["provider"]:
                p = by_prov.setdefault(r["provider"], {"n": 0, "ok": 0, "lat": 0.0})
                p["n"] += 1
                p["ok"] += 1 if r["ok"] else 0
                if r["ok"] and r["latency_s"]:
                    p["lat"] += r["latency_s"]
            if r["model"]:
                by_model[r["model"]] = by_model.get(r["model"], 0) + 1
            if r["tools"]:
                for t in r["tools"].split(","):
                    if t:
                        tools[t] = tools.get(t, 0) + 1
            for w in (r["goal_norm"] or "").split():
                if len(w) > 3:
                    terms[w] = terms.get(w, 0) + 1

        def _score(p):
            n = max(p["n"], 1)
            avg = (p["lat"] / p["ok"]) if p["ok"] else 99.0
            return (p["ok"] / n) / max(avg, 0.05)

        best_prov = max(by_prov.items(), key=lambda kv: _score(kv[1]))[0] if by_prov else ""
        best_model = max(by_model.items(), key=lambda kv: kv[1])[0] if by_model else ""
        lat = [r["latency_s"] for r in oks if r["latency_s"]]
        stop = {"with", "that", "this", "from", "into", "your", "have", "what",
                "when", "then", "they", "them", "about", "would", "could", "should"}
        top_terms = [w for w, _ in sorted(terms.items(), key=lambda kv: -kv[1])
                     if w not in stop][:8]
        return {
            "cold_start": False,
            "interactions": total,
            "lessons": n_lessons,
            "success_rate": round(len(oks) / total, 3),
            "avg_latency": round(sum(lat) / len(lat), 3) if lat else None,
            "cache_hits": sum(1 for r in rows if r["cached"]),
            "preferred_provider": best_prov,
            "preferred_model": best_model,
            "providers_seen": {k: v["n"] for k, v in
                               sorted(by_prov.items(), key=lambda kv: -kv[1]["n"])},
            "tool_affinity": dict(sorted(tools.items(), key=lambda kv: -kv[1])[:6]),
            "vocabulary": top_terms,
        }

    def router_priors(self):
        """Per-provider (success_rate, avg_latency) for seeding the router."""
        with self._lock, self._conn() as c:
            rows = c.execute(
                "SELECT provider, ok, latency_s FROM interactions"
                " WHERE kind = 'route' AND provider != ''").fetchall()
        out = {}
        for r in rows:
            p = out.setdefault(r["provider"], {"n": 0, "ok": 0, "lat": 0.0})
            p["n"] += 1
            p["ok"] += 1 if r["ok"] else 0
            if r["ok"] and r["latency_s"]:
                p["lat"] += r["latency_s"]
        # Two calls is the floor: one tells you nothing about consistency, and
        # the router's EWMA still decays a bad prior the moment reality disagrees.
        return {name: {"success": (v["ok"] / v["n"]) if v["n"] else 1.0,
                       "latency": (v["lat"] / v["ok"]) if v["ok"] else 0.0,
                       "calls": v["n"]}
                for name, v in out.items() if v["n"] >= 2}

    def stats(self):
        with self._lock, self._conn() as c:
            n = c.execute("SELECT COUNT(*) FROM interactions").fetchone()[0]
            l = c.execute("SELECT COUNT(*) FROM lessons").fetchone()[0]
        return {"interactions": n, "lessons": l}


def seed_policy(policy, max_samples=12):
    """Replay learned provider statistics into a fresh RouterPolicy.

    The router's EWMA starts optimistic and self-corrects, but that costs real
    latency on every process start. Replaying the last run's measured success
    rate and latency as synthetic observations means a restart begins with the
    routing table the previous session had already earned — and the EWMA still
    decays it away the moment reality disagrees.
    """
    if not learning_enabled():
        return 0
    # Seed each policy object exactly once. The policy is a process-wide
    # singleton, so re-seeding on every route() call would keep compounding
    # historical observations on top of live measurements.
    if getattr(policy, "_learning_seeded", False):
        return 0
    try:
        priors = get_store().router_priors()
    except Exception:
        return 0
    policy._learning_seeded = True
    seeded = 0
    for name, st in priors.items():
        # one synthetic observation per recorded call, capped so old history
        # can never outweigh live measurements for long.
        n = min(int(st.get("calls") or 0), max_samples)
        for i in range(max(n, 1)):
            try:
                policy.note_result(name, i < int(round(st["success"] * n)) or n == 0,
                                   st.get("latency") or 0.0)
                seeded += 1
            except Exception:
                break
    return seeded


# ---------------------------------------------------------------- singleton

_store = None
_store_lock = threading.Lock()


def get_store(db_path=None):
    global _store
    with _store_lock:
        if _store is None or db_path is not None:
            _store = LearningStore(db_path=db_path)
        return _store


def record_route(goal, provider, model, ok, latency_s=0.0, attempts=1,
                 cached=False):
    """Module-level convenience: record a routing outcome. Never raises."""
    if not learning_enabled():
        return
    try:
        get_store().record_route(goal, provider, model, ok, latency_s,
                                 attempts, cached)
    except Exception:
        pass


def record_agent_run(goal, tools, ok, steps=0, provider="", model=""):
    """Module-level convenience: record an agent run. Never raises."""
    if not learning_enabled():
        return
    try:
        get_store().record_agent_run(goal, tools, ok, steps, provider, model)
    except Exception:
        pass


def note_failure(goal, error):
    """Module-level convenience: file a lesson from a failure. Never raises."""
    if not learning_enabled():
        return None
    try:
        return get_store().note_failure(goal, error)
    except Exception:
        return None


def add_lesson(text, trigger="", kind="correction", weight=1.0):
    if not learning_enabled():
        return None
    try:
        return get_store().add_lesson(text, trigger, kind, weight)
    except Exception:
        return None


def set_store(store):
    """Replace/reset the singleton (tests)."""
    global _store
    with _store_lock:
        _store = store


# ---------------------------------------------------------------- prompt glue

def prompt_context(goal="", max_lessons=3):
    """Compact memory block for the agent's system prompt.

    Returns "" when there is nothing worth saying — a cold start must not add
    noise to the context.
    """
    if not learning_enabled():
        return ""
    try:
        store = get_store()
        prof = store.profile()
        if prof.get("cold_start") and not prof.get("lessons"):
            return ""
        lines = ["What flippy has learned about this user and this codebase:"]
        if not prof.get("cold_start"):
            lines.append(
                f"- {prof['interactions']} recorded interactions, "
                f"{int(prof['success_rate'] * 100)}% successful")
            if prof.get("preferred_provider"):
                lines.append(
                    f"- most reliable provider so far: {prof['preferred_provider']}"
                    + (f" (model {prof['preferred_model']})"
                       if prof.get("preferred_model") else ""))
            if prof.get("tool_affinity"):
                top = ", ".join(list(prof["tool_affinity"])[:4])
                lines.append(f"- goals here usually need: {top}")
            if prof.get("vocabulary"):
                lines.append(f"- recurring domain terms: {', '.join(prof['vocabulary'])}")
        for les in store.lessons_for(goal, k=max_lessons):
            tag = "correction" if les["kind"] == "correction" else "prior failure"
            lines.append(f"- {tag}: {les['text']}")
        return "\n".join(lines) if len(lines) > 1 else ""
    except Exception:
        return ""  # memory must never break a run


def render_text(prof=None):
    """Human-readable profile for `flippy profile`."""
    prof = prof or get_store().profile()
    if prof.get("cold_start"):
        return ("No interactions recorded yet. flippy learns from every routed\n"
                "call and agent run; come back after a few and this fills in.")
    lines = ["flippy memory", "-" * 62,
             f"  interactions        {prof['interactions']}",
             f"  success rate        {int(prof['success_rate'] * 100)}%",
             f"  avg latency         {prof['avg_latency']}s",
             f"  cache hits          {prof['cache_hits']}",
             f"  lessons stored      {prof['lessons']}",
             f"  preferred provider  {prof['preferred_provider'] or '-'}",
             f"  preferred model     {prof['preferred_model'] or '-'}"]
    if prof.get("providers_seen"):
        lines.append("  providers used      " + ", ".join(
            f"{k}({v})" for k, v in list(prof["providers_seen"].items())[:6]))
    if prof.get("tool_affinity"):
        lines.append("  tool affinity       " + ", ".join(
            f"{k}({v})" for k, v in prof["tool_affinity"].items()))
    if prof.get("vocabulary"):
        lines.append("  your vocabulary     " + ", ".join(prof["vocabulary"]))
    return "\n".join(lines)

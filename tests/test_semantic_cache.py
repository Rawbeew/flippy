"""Tests for semantic_cache — SQLite similarity-keyed response cache (stdlib only)."""
import os
import sys
import time
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from loomweaver import semantic_cache as sc


def msgs(text):
    return [{"role": "user", "content": text}]


class SemanticCacheTest(unittest.TestCase):
    def setUp(self):
        import tempfile
        self.tmp = tempfile.mkdtemp()
        self.cache = sc.SemanticCache(
            db_path=os.path.join(self.tmp, "c.sqlite3"), threshold=0.92, ttl_hours=168)

    def tearDown(self):
        self.cache.close()

    def test_normalize(self):
        self.assertEqual(sc.normalize("What is 2+2?"), "what is 2 2")
        self.assertEqual(sc.normalize("  HELLO   world!!  "), "hello world")

    def test_exact_hit(self):
        self.cache.store(msgs("What is the capital of France?"), "Paris", model_tag="m1")
        hit = self.cache.lookup(msgs("What is the capital of France?"))
        self.assertIsNotNone(hit)
        self.assertTrue(hit["exact"])
        self.assertEqual(hit["similarity"], 1.0)
        self.assertEqual(hit["response"], "Paris")

    def test_near_duplicate_hit(self):
        self.cache.store(msgs("What is 2+2?"), "4")
        hit = self.cache.lookup(msgs("what is 2 + 2?"))
        self.assertIsNotNone(hit, "punctuation/spacing variant should be a near-duplicate hit")
        self.assertEqual(hit["response"], "4")

    def test_miss_on_different_meaning(self):
        self.cache.store(msgs("What is the capital of France?"), "Paris")
        self.assertIsNone(self.cache.lookup(msgs("How do I bake sourdough bread?")))

    def test_ttl_expiry(self):
        c = sc.SemanticCache(db_path=os.path.join(self.tmp, "ttl.sqlite3"),
                             threshold=0.92, ttl_hours=0.00001)
        c.store(msgs("hello world"), "hi")
        time.sleep(0.05)  # let created_ts fall outside the tiny TTL
        self.assertIsNone(c.lookup(msgs("hello world")))
        self.assertEqual(c.stats()["entries"], 0)  # pruned on access
        c.close()

    def test_hit_count_increments(self):
        self.cache.store(msgs("tell me a joke"), "ha")
        self.cache.lookup(msgs("tell me a joke"))
        self.cache.lookup(msgs("Tell me a joke!"))
        row = self.cache._conn.execute(
            "SELECT hit_count FROM cache_entries").fetchone()
        self.assertEqual(row[0], 2)

    def test_stateful_skipped(self):
        self.assertTrue(sc.is_stateful([{"role": "system", "content": "remember this session"}]))
        self.assertTrue(sc.is_stateful([{"role": "user", "content": "hi", "tools": []}]))
        self.assertFalse(sc.is_stateful(msgs("plain question")))

    def test_disabled_via_env(self):
        old = os.environ.get("LOOMWEAVER_CACHE_ENABLED")
        os.environ["LOOMWEAVER_CACHE_ENABLED"] = "0"
        try:
            self.assertFalse(sc.cache_enabled())
            self.assertTrue(sc.cache_enabled.__module__ == "loomweaver.semantic_cache")
        finally:
            if old is None:
                os.environ.pop("LOOMWEAVER_CACHE_ENABLED", None)
            else:
                os.environ["LOOMWEAVER_CACHE_ENABLED"] = old

    def test_route_cache_integration(self):
        """route() should serve a cached response without calling providers."""
        import tempfile
        from loomweaver import core
        tmp2 = tempfile.mkdtemp()
        core._sc._default_cache = None  # reset singleton
        old_enabled = os.environ.get("LOOMWEAVER_CACHE_ENABLED")
        os.environ["LOOMWEAVER_CACHE_ENABLED"] = "1"  # opt back in (conftest disables)
        os.environ["LOOMWEAVER_CACHE_DB"] = os.path.join(tmp2, "route.sqlite3")
        try:
            events = []
            cache = core._sc.SemanticCache(db_path=os.environ["LOOMWEAVER_CACHE_DB"])
            cache.store(msgs("ping test"), "cached-pong", model_tag="t")
            cache.close()
            r = core.route(msgs("ping test"), creds={}, on_event=events.append)
            self.assertTrue(r.get("ok"))
            self.assertTrue(r.get("cached"))
            self.assertEqual(r["text"], "cached-pong")
            self.assertEqual(r["provider"], "cache")
            self.assertIn("cache_hit", [e["type"] for e in events])
        finally:
            core._sc._default_cache = None
            os.environ.pop("LOOMWEAVER_CACHE_DB", None)
            if old_enabled is None:
                os.environ.pop("LOOMWEAVER_CACHE_ENABLED", None)
            else:
                os.environ["LOOMWEAVER_CACHE_ENABLED"] = old_enabled


if __name__ == "__main__":
    unittest.main()

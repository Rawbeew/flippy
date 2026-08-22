import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import pytest


@pytest.fixture(autouse=True)
def _disable_semantic_cache(monkeypatch):
    """Route tests exercise provider failover, not caching — keep the response
    cache off unless a test explicitly opts back in."""
    monkeypatch.setenv("LOOMWEAVER_CACHE_ENABLED", "0")
    from loomweaver import semantic_cache as sc
    sc._default_cache = None
    yield
    sc._default_cache = None

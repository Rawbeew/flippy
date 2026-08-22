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


@pytest.fixture(autouse=True)
def _fresh_key_rotation(tmp_path):
    """Isolate key-rotation state per test — never touch the real SQLite DB."""
    from loomweaver import core, key_rotation as kr
    st = kr.RotationState(db_path=str(tmp_path / "key_rotation.db"))
    core._kr.set_state(st)
    yield
    core._kr.set_state(None)


@pytest.fixture(autouse=True)
def _fresh_quota_ledger(tmp_path):
    """Isolate the quota ledger per test — a 429 cooldown recorded by one
    route test must not make later tests skip the same provider."""
    from loomweaver import core, quota_ledger as ql
    led = ql.QuotaLedger(db_path=str(tmp_path / "quota.db"))
    with mock.patch.object(core._ql, "get_ledger", return_value=led):
        yield led


import unittest.mock as mock  # noqa: E402  (used by fixture above)

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


@pytest.fixture(autouse=True)
def _fresh_router_policy(monkeypatch):
    """Isolate adaptive-routing state per test — one test's provider
    failures/cooldowns must not reorder providers for the next test.
    Also keeps retry backoffs instant so retry tests stay fast."""
    from loomweaver import core, router_policy as rp
    monkeypatch.setenv("LOOMWEAVER_RETRY_BASE_S", "0.001")
    fresh = rp.RouterPolicy()
    with mock.patch.object(core._rp, "get_policy", return_value=fresh):
        yield fresh


@pytest.fixture(autouse=True)
def _fresh_learning_store(tmp_path, monkeypatch):
    """Isolate the self-learning store per test.

    Without this, one test's recorded routes become the next test's routing
    priors, and every test run appends to the developer's real runs/learning.db.
    """
    from loomweaver import learning
    store = learning.LearningStore(db_path=str(tmp_path / "learning.db"))
    learning.set_store(store)
    yield store
    learning.set_store(None)


@pytest.fixture(autouse=True)
def _fresh_usage_db(tmp_path):
    """Isolate per-provider usage per test.

    The usage summary is aggregated by provider name, so an un-isolated DB lets
    one test's synthetic prov_fast/prov_slow rows leak into another test's
    dashboard assertions.
    """
    from loomweaver import core, usage
    db = usage.UsageDB(db_path=str(tmp_path / "usage.db"))
    with mock.patch.object(core._usage, "get_db", return_value=db):
        yield db


@pytest.fixture(autouse=True)
def _clean_provider_env(monkeypatch):
    """Strip provider credentials from the ambient environment.

    Without this, a test run inherits whatever keys happen to be exported in
    the developer's shell or CI job, and assertions like
    `flippy_providers_configured 0` flip depending on the machine. Tests that
    want a provider configured set it explicitly with monkeypatch, which runs
    after this fixture and so still wins.
    """
    for var in list(os.environ):
        if var.endswith(("_API_KEY", "_APIKEY", "_KEY", "_TOKEN", "_SECRET")) or \
                var.endswith(("_BASE_URL", "_API_BASE", "_BASE", "_ENDPOINT",
                              "_MODELS")):
            monkeypatch.delenv(var, raising=False)
    yield


import unittest.mock as mock  # noqa: E402  (used by fixture above)

"""The self-learning layer: outcome memory, user profile, and self-correction.

These tests pin the three guarantees the feature exists to provide:
  1. what happened is remembered and survives a process restart;
  2. what was learned measurably changes the next routing decision;
  3. a failure or a correction filed against one goal reaches a *similar*
     future goal, and does not leak into an unrelated one.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from loomweaver import learning
from loomweaver import router_policy


@pytest.fixture
def store(tmp_path):
    st = learning.LearningStore(db_path=str(tmp_path / "learning.db"))
    learning.set_store(st)
    yield st
    learning.set_store(None)


class TestOutcomeMemory:
    def test_records_and_reads_back(self, store):
        store.record_route("summarise the invoice", "groq", "gpt-oss-20b",
                           True, 0.9, 1)
        s = store.stats()
        assert s["interactions"] == 1 and s["lessons"] == 0

    def test_accepts_an_openai_message_list_not_just_a_string(self, store):
        """core.route() passes `messages`, not a goal string."""
        store.record_route([{"role": "system", "content": "be terse"},
                            {"role": "user", "content": "deploy to staging"}],
                           "groq", "m", True, 0.5, 1)
        prof = store.profile()
        assert "staging" in prof["vocabulary"]

    def test_survives_a_process_restart(self, tmp_path):
        """The whole point: memory outlives the process that made it."""
        path = str(tmp_path / "persist.db")
        first = learning.LearningStore(db_path=path)
        first.record_route("rotate the api keys", "groq", "m", True, 0.4, 1)
        second = learning.LearningStore(db_path=path)  # a brand new connection
        assert second.stats()["interactions"] == 1
        assert "rotate" in second.profile()["vocabulary"]

    def test_disabled_by_env_is_a_no_op(self, tmp_path, monkeypatch):
        monkeypatch.setenv("LOOMWEAVER_LEARNING_ENABLED", "0")
        st = learning.LearningStore(db_path=str(tmp_path / "off.db"))
        learning.set_store(st)
        st.record_route("anything", "groq", "m", True, 0.1, 1)
        assert st.stats()["interactions"] == 0
        assert learning.prompt_context("anything") == ""

    def test_thread_safety(self, store):
        """route() is called from a ThreadingHTTPServer; the store must cope."""
        import threading
        errs = []

        def work(i):
            try:
                for n in range(10):
                    store.record_route(f"goal {i} {n}", "groq", "m", True, 0.1, 1)
            except Exception as exc:  # pragma: no cover - failure is the assertion
                errs.append(exc)

        threads = [threading.Thread(target=work, args=(i,)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errs
        assert store.stats()["interactions"] == 80


class TestProfile:
    def test_cold_start_says_so(self, store):
        prof = store.profile()
        assert prof["cold_start"] is True
        assert "No interactions recorded" in learning.render_text(prof)

    def test_infers_provider_tools_and_vocabulary(self, store):
        for _ in range(3):
            store.record_route("deploy the api to staging", "groq", "gpt-oss-20b",
                               True, 0.8, 1)
        store.record_route("deploy the api to staging", "openrouter", "stealth",
                           False, 6.0, 3)
        store.record_agent_run("deploy the api to staging",
                               ["shell", "read_file"], True, 4)
        prof = store.profile()
        assert prof["cold_start"] is False
        assert prof["preferred_provider"] == "groq", "groq was fast and never failed"
        # 3 groq ok + 1 openrouter fail + 1 agent run ok = 4 of 5
        assert prof["interactions"] == 5
        assert prof["success_rate"] == 0.8
        assert set(prof["tool_affinity"]) == {"shell", "read_file"}
        assert "deploy" in prof["vocabulary"] and "staging" in prof["vocabulary"]

    def test_render_text_shows_the_numbers(self, store):
        store.record_route("ship it", "groq", "m", True, 1.0, 1)
        out = learning.render_text(store.profile())
        assert "preferred provider" in out and "success rate" in out


class TestSelfCorrection:
    def test_taught_lesson_reaches_a_similar_goal(self, store):
        store.add_lesson("When I say 'deploy', run the migration script first.",
                         trigger="deploy the api to staging")
        hits = store.lessons_for("please deploy the api to staging now")
        assert len(hits) == 1
        assert "migration script" in hits[0]["text"]

    def test_lesson_does_not_leak_into_an_unrelated_goal(self, store):
        store.add_lesson("When I say 'deploy', run the migration script first.",
                         trigger="deploy the api to staging")
        assert store.lessons_for("what is the weather in lagos") == []

    def test_a_failure_files_its_own_lesson(self, store):
        lid = store.note_failure("query the postgres database",
                                 "connection refused: no pg_hba entry")
        assert lid is not None
        hits = store.lessons_for("query the postgres database again")
        assert hits and "pg_hba" in hits[0]["text"]

    def test_noise_failures_do_not_become_lessons(self, store):
        """A blanket 'all providers failed' teaches nothing and must be dropped."""
        assert store.note_failure("anything", "all providers failed") is None
        assert store.note_failure("anything", "") is None
        assert store.stats()["lessons"] == 0

    def test_forget_one_and_forget_all(self, store):
        a = store.add_lesson("rule one", trigger="one")
        store.add_lesson("rule two", trigger="two")
        store.forget(a)
        assert store.stats()["lessons"] == 1
        store.forget()
        assert store.stats()["lessons"] == 0

    def test_retrieved_lessons_are_counted_as_hits(self, store):
        store.add_lesson("rule", trigger="deploy to staging")
        store.lessons_for("deploy to staging")
        with store._conn() as c:
            assert c.execute("SELECT hits FROM lessons").fetchone()[0] == 1


class TestRoutingActuallyChanges:
    def test_learned_priors_reorder_providers(self, store):
        """The load-bearing assertion: memory must change the next decision."""
        store.record_route("g", "groq", "m", True, 0.5, 1)
        store.record_route("g", "groq", "m", True, 0.5, 1)
        store.record_route("g", "openrouter", "m", False, 6.0, 1)
        store.record_route("g", "openrouter", "m", False, 6.0, 1)

        candidates = [{"name": "openrouter"}, {"name": "groq"}]
        fresh = [p["name"] for p in router_policy.RouterPolicy().order(candidates)]

        seeded = router_policy.RouterPolicy()
        assert learning.seed_policy(seeded) > 0
        after = [p["name"] for p in seeded.order(candidates)]

        assert after[0] == "groq", f"expected groq promoted, got {after}"
        assert fresh[0] == "openrouter", (
            "the unseeded policy must start alphabetical, or this test proves nothing")

    def test_seeding_is_idempotent(self, store):
        store.record_route("g", "groq", "m", True, 0.5, 1)
        store.record_route("g", "groq", "m", True, 0.5, 1)
        pol = router_policy.RouterPolicy()
        first = learning.seed_policy(pol)
        assert first > 0
        assert learning.seed_policy(pol) == 0, "must not compound on every call"

    def test_one_call_is_not_enough_to_form_a_prior(self, store):
        store.record_route("g", "groq", "m", True, 0.5, 1)
        assert store.router_priors() == {}


class TestPromptContext:
    def test_empty_on_a_cold_start(self, store):
        assert learning.prompt_context("anything at all") == ""

    def test_carries_profile_and_lessons(self, store):
        store.record_route("deploy the api to staging", "groq", "gpt-oss-20b",
                           True, 0.8, 1)
        store.add_lesson("Always run migrations before deploying.",
                         trigger="deploy the api to staging")
        ctx = learning.prompt_context("deploy the api to staging")
        assert "learned about this user" in ctx
        assert "groq" in ctx
        assert "Always run migrations" in ctx

    def test_never_raises_even_with_a_broken_store(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("store exploded")
        monkeypatch.setattr(learning, "get_store", boom)
        assert learning.prompt_context("anything") == ""


class TestModuleLevelApi:
    """core.py and agent.py call the module-level functions, not the store."""

    def test_record_route_and_note_failure_are_module_level(self, store):
        learning.record_route("g", "groq", "m", True, 0.3, 1)
        assert store.stats()["interactions"] == 1
        learning.note_failure("g", "boom")
        assert store.stats()["lessons"] == 1

    def test_module_level_api_swallows_errors(self, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("no store")
        monkeypatch.setattr(learning, "get_store", boom)
        learning.record_route("g", "p", "m", True)      # must not raise
        learning.record_agent_run("g", ["shell"], True)  # must not raise
        assert learning.note_failure("g", "boom") is None

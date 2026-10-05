"""Coverage for cron.py — the opt-in local scheduler.

Was at 35%. The jobs/state files are redirected into tmp_path so tests cannot
write into the package directory, and subprocess.run is stubbed so no job is
really launched.
"""
import json
import subprocess
import types
from unittest import mock

import pytest

from loomweaver import cron


JOBS = [
    {"name": "nightly-eval", "interval_minutes": 1440,
     "cmd": ["eval", "--suite", "basic"]},
    {"name": "groq-loadtest", "interval_minutes": 720,
     "cmd": ["loadtest", "--provider", "groq"]},
]


class Done(Exception):
    """Raised from a stubbed time.sleep to break the daemon's infinite loop."""


@pytest.fixture
def sandboxed_cron(monkeypatch, tmp_path):
    """Point the jobs/state files at tmp_path."""
    jobs_file = tmp_path / "cron_jobs.json"
    state_file = tmp_path / "cron_state.json"
    monkeypatch.setattr(cron, "JOBS_FILE", str(jobs_file))
    monkeypatch.setattr(cron, "STATE_FILE", str(state_file))
    return {"jobs": jobs_file, "state": state_file}


@pytest.fixture
def written_jobs(sandboxed_cron):
    sandboxed_cron["jobs"].write_text(json.dumps(JOBS))
    return sandboxed_cron


class FakeProc:
    def __init__(self, returncode=0, stdout="", stderr=""):
        self.returncode = returncode
        self.stdout = stdout
        self.stderr = stderr


# --------------------------------------------------------------------------
# _jobs / _state / _save_state
# --------------------------------------------------------------------------
class TestPersistence:
    def test_a_missing_jobs_file_is_seeded_with_the_defaults(self, sandboxed_cron):
        jobs = cron._jobs()
        assert jobs == cron.DEFAULT_JOBS
        assert sandboxed_cron["jobs"].exists(), "the seed must be written to disk"

    def test_an_existing_jobs_file_is_read_back_verbatim(self, written_jobs):
        assert cron._jobs() == JOBS

    def test_a_missing_state_file_reads_as_empty(self, sandboxed_cron):
        assert cron._state() == {}

    def test_state_round_trips(self, sandboxed_cron):
        cron._save_state({"nightly-eval": {"exit_code": 0}})
        assert cron._state()["nightly-eval"]["exit_code"] == 0


# --------------------------------------------------------------------------
# run_job
# --------------------------------------------------------------------------
class TestRunJob:
    def test_an_unknown_job_is_reported_not_raised(self, written_jobs):
        res = cron.run_job("does-not-exist")
        assert res["ok"] is False and "not found" in res["error"]

    def test_a_job_outside_the_allowlist_is_refused(self, sandboxed_cron):
        sandboxed_cron["jobs"].write_text(json.dumps(
            [{"name": "evil", "interval_minutes": 1, "cmd": ["agent", "pwn me"]}]))
        res = cron.run_job("evil")
        assert res["ok"] is False and "rejected" in res["error"]

    def test_an_empty_cmd_is_refused(self, sandboxed_cron):
        sandboxed_cron["jobs"].write_text(json.dumps(
            [{"name": "blank", "interval_minutes": 1, "cmd": []}]))
        assert cron.run_job("blank")["ok"] is False

    def test_a_successful_run_records_state(self, written_jobs):
        with mock.patch.object(subprocess, "run",
                               return_value=FakeProc(0, stdout="all good")):
            res = cron.run_job("nightly-eval")
        assert res["ok"] is True and res["exit_code"] == 0
        assert res["output_tail"] == "all good"
        assert cron._state()["nightly-eval"]["exit_code"] == 0

    def test_a_failing_run_is_reported_as_not_ok(self, written_jobs):
        with mock.patch.object(subprocess, "run",
                               return_value=FakeProc(1, stderr="boom")):
            res = cron.run_job("nightly-eval")
        assert res["ok"] is False and res["exit_code"] == 1
        assert cron._state()["nightly-eval"]["exit_code"] == 1

    def test_the_job_is_run_as_a_subprocess_of_the_engine(self, written_jobs):
        seen = {}

        def cap(cmd, **kw):
            seen["cmd"] = cmd
            seen["timeout"] = kw.get("timeout")
            return FakeProc(0)

        with mock.patch.object(subprocess, "run", side_effect=cap):
            cron.run_job("nightly-eval")
        assert seen["cmd"][-3:] == ["eval", "--suite", "basic"]
        assert seen["cmd"][:3][1:3] == ["-m", "src.loomweaver"]
        assert seen["timeout"] == 1800, "a runaway job must be bounded"

    def test_a_long_output_is_truncated_in_state(self, written_jobs):
        with mock.patch.object(subprocess, "run",
                               return_value=FakeProc(0, stdout="x" * 5000)):
            cron.run_job("nightly-eval")
        assert len(cron._state()["nightly-eval"]["output_tail"]) <= 800

    def test_the_result_tail_is_shorter_than_the_state_tail(self, written_jobs):
        with mock.patch.object(subprocess, "run",
                               return_value=FakeProc(0, stdout="x" * 5000)):
            res = cron.run_job("nightly-eval")
        assert len(res["output_tail"]) <= 400

    def test_a_second_run_overwrites_the_first_state_row(self, written_jobs):
        with mock.patch.object(subprocess, "run", return_value=FakeProc(0)):
            cron.run_job("nightly-eval")
        with mock.patch.object(subprocess, "run", return_value=FakeProc(3)):
            cron.run_job("nightly-eval")
        assert len(cron._state()) == 1
        assert cron._state()["nightly-eval"]["exit_code"] == 3


# --------------------------------------------------------------------------
# daemon
# --------------------------------------------------------------------------
class TestDaemon:
    def test_a_due_job_fires_and_stamps_its_epoch(self, written_jobs, capsys):
        fired = []

        def fake_run_job(name):
            fired.append(name)
            return {"ok": True, "duration_s": 0.1}

        def fake_sleep(_):
            raise Done

        with mock.patch.object(cron, "run_job", side_effect=fake_run_job), \
             mock.patch.object(cron.time, "sleep", side_effect=fake_sleep):
            with pytest.raises(Done):
                cron.daemon()
        assert "nightly-eval" in fired and "groq-loadtest" in fired
        assert cron._state()["nightly-eval"]["_epoch"] > 0
        assert "cron daemon started" in capsys.readouterr().out

    def test_a_job_that_is_not_yet_due_is_skipped(self, written_jobs):
        cron._save_state({"nightly-eval": {"_epoch": cron.time.time()}})
        fired = []

        def fake_run_job(name):
            fired.append(name)
            return {"ok": True, "duration_s": 0.1}

        with mock.patch.object(cron, "run_job", side_effect=fake_run_job), \
             mock.patch.object(cron.time, "sleep", side_effect=Done):
            with pytest.raises(Done):
                cron.daemon()
        assert "nightly-eval" not in fired, "a fresh job must not refire"
        assert "groq-loadtest" in fired

    def test_a_job_added_after_startup_does_not_kill_the_loop(self, written_jobs):
        """Regression: index-assigning state for an unseen job raised KeyError
        and killed the daemon. run_job may add a row before we stamp _epoch."""
        def fake_run_job(name):
            # the job ran but left no row behind, exactly like a job added to
            # the config after the daemon started
            st = cron._state()
            st.pop(name, None)
            cron._save_state(st)
            return {"ok": True, "duration_s": 0.1}

        def fake_sleep(_):
            raise Done

        with mock.patch.object(cron, "run_job", side_effect=fake_run_job), \
             mock.patch.object(cron.time, "sleep", side_effect=fake_sleep):
            with pytest.raises(Done):
                cron.daemon()
        assert cron._state()["nightly-eval"]["_epoch"] > 0

    def test_a_failed_job_still_stamps_its_epoch(self, written_jobs):
        with mock.patch.object(cron, "run_job",
                               return_value={"ok": False, "duration_s": 0.1}), \
             mock.patch.object(cron.time, "sleep", side_effect=Done):
            with pytest.raises(Done):
                cron.daemon()
        assert cron._state()["groq-loadtest"]["_epoch"] > 0


# --------------------------------------------------------------------------
# cli
# --------------------------------------------------------------------------
def args(list_=False, run=None, daemon=False):
    return types.SimpleNamespace(list=list_, run=run, daemon=daemon)


class TestCli:
    def test_no_flags_lists_the_jobs(self, written_jobs, capsys):
        cron.cli(args())
        out = capsys.readouterr().out
        assert "nightly-eval" in out and "every 1440min" in out
        assert "last=never" in out and "exit=-" in out

    def test_the_list_shows_the_last_result(self, written_jobs, capsys):
        cron._save_state({"nightly-eval": {"last_run": "2026-01-01 00:00:00",
                                           "exit_code": 0}})
        cron.cli(args(list_=True))
        out = capsys.readouterr().out
        assert "last=2026-01-01 00:00:00" in out and "exit=0" in out

    def test_run_prints_the_result_as_json(self, written_jobs, capsys):
        with mock.patch.object(subprocess, "run",
                               return_value=FakeProc(0, stdout="done")):
            cron.cli(args(run="nightly-eval"))
        assert json.loads(capsys.readouterr().out)["ok"] is True

    def test_daemon_flag_starts_the_loop(self, written_jobs, capsys):
        with mock.patch.object(cron, "daemon", side_effect=KeyboardInterrupt):
            cron.cli(args(daemon=True))
        assert "stopped" in capsys.readouterr().out

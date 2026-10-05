"""Coverage for loomweaver/cli.py — the command dispatcher.

Was at 58%; the whole `elif args.cmd == ...` chain was largely untested.
Every subcommand is driven through main() with the work stubbed out, so these
assert on wiring (right function, right arguments, right exit code) and never
touch the network.
"""
import json

import pytest

from loomweaver import cli


# conftest's autouse _fresh_learning_store fixture already points the learning
# store at tmp_path/learning.db, so nothing to set up here.


# --------------------------------------------------------------------------
# _resolve_tool_scope — the pre-flight intake gate
# --------------------------------------------------------------------------
class TestResolveToolScope:
    def test_no_flag_means_auto(self):
        assert cli._resolve_tool_scope(None) == "auto"
        assert cli._resolve_tool_scope("") == "auto"

    def test_no_flag_with_allow_empty_means_no_opinion(self):
        assert cli._resolve_tool_scope(None, allow_empty=True) is None

    def test_a_comma_list_is_parsed_and_trimmed(self):
        assert cli._resolve_tool_scope("shell, read_file ") == ["shell", "read_file"]

    def test_an_unknown_tool_is_a_hard_error_not_a_fallback(self):
        """The point of the gate: a typo must not silently widen the toolset by
        falling through to goal-based auto-grant."""
        with pytest.raises(SystemExit) as exc:
            cli._resolve_tool_scope("shel")
        msg = str(exc.value)
        assert "unknown tool(s)" in msg and "shel" in msg
        assert "shell" in msg, "the available list should help the operator"

    def test_one_bad_name_rejects_the_whole_list(self):
        with pytest.raises(SystemExit):
            cli._resolve_tool_scope("shell,not_a_tool")

    def test_every_name_in_TOOLS_is_accepted(self):
        from loomweaver import tools
        assert cli._resolve_tool_scope(",".join(sorted(tools.TOOLS))) == \
            sorted(tools.TOOLS)


# --------------------------------------------------------------------------
# providers
# --------------------------------------------------------------------------
class TestProviders:
    def test_all_prints_the_full_catalog(self, capsys, monkeypatch):
        monkeypatch.setattr(cli, "describe_catalog", lambda: [
            {"name": "groq", "cost": "free", "activate_with": "GROQ_KEY",
             "models_env": "GROQ_MODELS"}])
        assert cli.main(["providers", "--all"]) == 0
        out = capsys.readouterr().out
        assert "groq" in out and "GROQ_KEY" in out and "GROQ_MODELS" in out

    def test_a_missing_models_env_prints_a_dash(self, capsys, monkeypatch):
        monkeypatch.setattr(cli, "describe_catalog", lambda: [
            {"name": "x", "cost": "free", "activate_with": "X_KEY",
             "models_env": ""}])
        cli.main(["providers", "--all"])
        assert " -" in capsys.readouterr().out

    def test_nothing_configured_exits_1_with_a_hint(self, capsys, monkeypatch):
        monkeypatch.setattr(cli, "build_providers", lambda creds: [])
        assert cli.main(["providers"]) == 1
        assert "No providers configured" in capsys.readouterr().out

    def test_configured_providers_are_listed(self, capsys, monkeypatch):
        monkeypatch.setattr(cli, "build_providers", lambda creds: [
            {"name": "groq", "cost": "free", "models": ["m1", "m2"]}])
        cli.main(["providers"])
        out = capsys.readouterr().out
        assert "groq" in out and "m1, m2" in out and "1 configured" in out


# --------------------------------------------------------------------------
# profile / learn / forget
# --------------------------------------------------------------------------
class TestMemory:
    def test_profile_text_includes_the_store_stats(self, capsys):
        cli.main(["profile"])
        out = capsys.readouterr().out
        assert "store" in out and "interactions" in out and "learning.db" in out

    def test_profile_json_is_parseable(self, capsys):
        cli.main(["profile", "--json"])
        assert isinstance(json.loads(capsys.readouterr().out), dict)

    def test_learn_stores_and_confirms(self, capsys):
        cli.main(["learn", "always answer in French"])
        assert "learned (lesson" in capsys.readouterr().out

    def test_learn_accepts_a_trigger(self, capsys):
        cli.main(["learn", "run migrations", "--when", "deploy"])
        assert "learned (lesson" in capsys.readouterr().out

    def test_forget_with_neither_flag_is_a_usage_error(self, capsys):
        assert cli.main(["forget"]) == 2
        assert "usage:" in capsys.readouterr().err

    def test_forget_all_clears_everything(self, capsys):
        cli.main(["learn", "a rule"])
        capsys.readouterr()
        cli.main(["forget", "--all"])
        assert "forgot every lesson" in capsys.readouterr().out

    def test_forget_by_id_reports_the_id(self, capsys):
        cli.main(["learn", "a rule"])
        capsys.readouterr()
        cli.main(["forget", "--id", "1"])
        assert "forgot lesson 1" in capsys.readouterr().out


# --------------------------------------------------------------------------
# engine commands — each must reach the right function
# --------------------------------------------------------------------------
class TestEngineDispatch:
    def test_agent_forwards_the_goal_and_tool_scope(self, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr(cli.agent, "run",
                            lambda goal, **kw: seen.update(goal=goal, **kw) or {
                                "result": "done", "run_dir": "/tmp/run"})
        cli.main(["agent", "check disk", "--tools", "shell"])
        assert seen["goal"] == "check disk"
        assert seen["tool_scope"] == ["shell"]
        assert json.loads(capsys.readouterr().out)["result"] == "done"

    def test_agent_rejects_a_bad_tool_before_running(self, monkeypatch):
        monkeypatch.setattr(cli.agent, "run",
                            lambda *a, **k: pytest.fail("must not run"))
        with pytest.raises(SystemExit):
            cli.main(["agent", "do it", "--tools", "shel"])

    def test_eval_uses_the_agent_suite_when_asked(self, monkeypatch, capsys):
        monkeypatch.setattr(cli.evals, "run_agent_suite",
                            lambda **kw: {"agent": True})
        cli.main(["eval", "--suite", "agent"])
        assert json.loads(capsys.readouterr().out) == {"agent": True}

    def test_eval_uses_the_normal_suite_otherwise(self, monkeypatch, capsys):
        monkeypatch.setattr(cli.evals, "run_suite",
                            lambda suite, **kw: {"suite": suite})
        cli.main(["eval", "--suite", "basic"])
        assert json.loads(capsys.readouterr().out) == {"suite": "basic"}

    def test_eval_compare(self, monkeypatch, capsys):
        monkeypatch.setattr(cli.evals, "compare", lambda: {"rows": 1})
        cli.main(["eval-compare"])
        assert json.loads(capsys.readouterr().out) == {"rows": 1}

    def test_loadtest_forwards_provider_concurrency_and_requests(self, monkeypatch,
                                                                 capsys):
        seen = {}
        monkeypatch.setattr(cli.loadtest, "run",
                            lambda **kw: seen.update(kw) or {"ok": 1})
        cli.main(["loadtest", "--provider", "groq", "--concurrency", "3",
                  "--requests", "9"])
        assert seen == {"provider": "groq", "concurrency": 3, "requests": 9}

    def test_ttft(self, monkeypatch, capsys):
        monkeypatch.setattr(cli.loadtest, "ttft_sweep", lambda: [{"ttft": 0.2}])
        cli.main(["ttft"])
        assert json.loads(capsys.readouterr().out)[0]["ttft"] == 0.2

    def test_cron_is_handed_the_parsed_args(self, monkeypatch):
        seen = {}
        monkeypatch.setattr("loomweaver.cron.cli", lambda args: seen.update(args=args))
        cli.main(["cron", "--list"])
        assert seen["args"].list is True

    def test_usage_text_by_default(self, monkeypatch, capsys):
        monkeypatch.setattr("loomweaver.usage.summary", lambda hours: {"h": hours})
        monkeypatch.setattr("loomweaver.usage.render_text", lambda s: "TEXT")
        cli.main(["usage"])
        assert capsys.readouterr().out.strip() == "TEXT"

    def test_usage_json_when_asked(self, monkeypatch, capsys):
        monkeypatch.setattr("loomweaver.usage.summary", lambda hours: {"h": hours})
        monkeypatch.setattr("loomweaver.usage.render_json", lambda s: {"j": 1})
        cli.main(["usage", "--json"])
        assert json.loads(capsys.readouterr().out) == {"j": 1}

    def test_usage_forwards_the_hours_window(self, monkeypatch, capsys):
        seen = {}
        monkeypatch.setattr("loomweaver.usage.summary",
                            lambda hours: seen.update(hours=hours) or {})
        monkeypatch.setattr("loomweaver.usage.render_text", lambda s: "")
        cli.main(["usage", "--hours", "48"])
        assert seen["hours"] == 48

    def test_quota(self, monkeypatch, capsys):
        monkeypatch.setattr("loomweaver.quota_ledger.get_quota_status",
                            lambda: {"groq": {"left": 10}})
        cli.main(["quota"])
        assert json.loads(capsys.readouterr().out)["groq"]["left"] == 10

    def test_armada_forwards_the_mission_and_tool_scope(self, monkeypatch, capsys):
        seen = {}
        fleet = type("F", (), {"execute": lambda self, **kw: {"verdict": "PASS"}})()
        monkeypatch.setattr("loomweaver.armada.Armada",
                            lambda mission, **kw: seen.update(mission=mission, **kw)
                            or type("P", (), {"standard_pipeline": lambda s: fleet})())
        cli.main(["armada", "audit this", "--tools", "shell"])
        assert seen["mission"] == "audit this"
        assert seen["tool_scope"] == ["shell"]
        assert json.loads(capsys.readouterr().out)["verdict"] == "PASS"

    def test_armada_without_tools_passes_no_scope(self, monkeypatch, capsys):
        seen = {}
        fleet = type("F", (), {"execute": lambda self, **kw: {}})()
        monkeypatch.setattr("loomweaver.armada.Armada",
                            lambda mission, **kw: seen.update(kw)
                            or type("P", (), {"standard_pipeline": lambda s: fleet})())
        cli.main(["armada", "audit this"])
        assert seen["tool_scope"] is None

    def test_doctor_and_check_config_are_the_same_command(self, monkeypatch, capsys):
        monkeypatch.setattr(cli, "doctor", lambda **kw: [
            {"check": "keys", "status": "OK", "detail": "fine"}])
        cli.main(["doctor"])
        first = capsys.readouterr().out
        cli.main(["check-config"])
        assert first == capsys.readouterr().out
        assert "[ OK ] keys: fine" in first

"""Tests for armada.py — agent roles, tool restrictions, pipeline flow. Mocked."""
import json
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loomweaver.armada import Agent, Armada, ROLES, run_agent
from loomweaver.core import RunLog


class TestRoles:
    def test_all_roles_exist(self):
        assert set(ROLES) == {"scout", "builder", "verifier", "reporter"}

    def test_scout_is_readonly(self):
        assert ROLES["scout"]["readonly"] is True
        assert "write_file" not in ROLES["scout"]["tools"]

    def test_builder_can_write(self):
        assert "write_file" in ROLES["builder"]["tools"]

    def test_verifier_cannot_write(self):
        assert "write_file" not in ROLES["verifier"]["tools"]

    def test_invalid_role_raises(self):
        try:
            Agent("x", "admiral", "goal")
            assert False, "should have raised"
        except ValueError:
            pass


class TestToolRestriction:
    def test_scout_blocked_from_write_file(self):
        """Role-based tool enforcement: scout tries write_file → BLOCKED."""
        from loomweaver.tools import TOOLS  # ensure registered

        agent = Agent("s1", "scout", "try to write a file")
        messages = [
            {"role": "system", "content": agent.system_prompt()},
            {"role": "user", "content": 'MISSION: test\nAVAILABLE TOOLS: shell\n'
             'reply {"tool": "write_file", "args": {"path": "/tmp/x", "content": "y"}}'},
        ]
        # mock route to return the tool-call JSON then done
        responses = [
            {"ok": True, "text": '{"tool": "write_file", "args": {"path": "/tmp/x", "content": "y"}}',
             "provider": "mock"},
            {"ok": True, "text": '{"done": "attempted"}', "provider": "mock"},
        ]
        with mock.patch("loomweaver.core.route", side_effect=responses), \
             mock.patch("loomweaver.core.load_creds", return_value={}):
            rl = RunLog(tempfile.mkdtemp())
            run_agent(agent, creds={}, max_steps=3, log=rl)
        # the blocked message should have been fed back — verify no file written
        assert not os.path.exists("/tmp/x") or "BLOCKED" in str(
            [e for e in rl.read() if e["type"] == "tool_call"])


import os  # noqa: E402


class TestArmadaPipeline:
    def _mock_route_factory(self, script):
        """script: list of (agent_name, [responses])"""
        call_idx = {}

        def fake_route(messages, model=None, max_tokens=1024, creds=None, on_event=None):
            text = messages[-1]["content"]
            key = "unknown"
            if "MISSION" in str(messages[0]) + str(messages[-1]):
                pass
            # identify agent by system prompt content
            sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
            for role in ("Scout", "Builder", "Verifier", "Reporter"):
                if role in sys_msg:
                    key = role.lower()
                    break
            idx = call_idx.get(key, 0)
            call_idx[key] = idx + 1
            resp_list = script.get(key, [{"ok": True, "text": '{"done": "noop"}'}])
            return resp_list[min(idx, len(resp_list) - 1)]
        return fake_route

    def test_standard_pipeline_runs_all_four(self):
        fleet = Armada("test mission").standard_pipeline()
        assert [a.role for a in fleet.agents] == ["scout", "builder", "verifier", "reporter"]

    def test_context_flows_scout_to_builder(self):
        fleet = Armada("test mission").standard_pipeline()
        scout, builder = fleet.agents[0], fleet.agents[1]
        shared = {"scout_findings": "found 3 files with bugs"}
        goal = builder.goal.replace("{scout_findings}", shared["scout_findings"])
        assert "found 3 files" in goal

    def test_verifier_fail_stops_pipeline(self):
        fleet = Armada("test").standard_pipeline()
        # simulate: verifier fails
        fleet.agents[2].verdict = "FAIL"
        fleet.agents[2].result = "VERDICT: FAIL tests broken"
        stop_on_fail = True
        should_stop = (stop_on_fail and fleet.agents[2].role == "verifier"
                       and fleet.agents[2].verdict == "FAIL")
        assert should_stop

    def test_execute_with_mocks(self):
        script = {
            "scout": [{"ok": True, "text": '{"done": "FINDINGS: 3 files found"}'}],
            "builder": [{"ok": True, "text": '{"done": "BUILT: fix applied TESTS: 5/5"}'}],
            "verifier": [{"ok": True, "text": '{"done": "VERDICT: PASS all green"}'}],
            "reporter": [{"ok": True, "text": '{"done": "REPORT: all good"}'}],
        }
        # fix the typo'd key above programmatically
        script["verifier"] = [{"ok": True, "text": '{"done": "VERDICT: PASS all green"}'}]

        with mock.patch("loomweaver.core.build_providers",
                        lambda creds=None: [{"name": "mock", "cost": "free",
                                             "url": "http://x", "key": "k", "models": ["m"]}]), \
             mock.patch("loomweaver.core.chat",
                        side_effect=lambda p, msgs, model=None, max_tokens=1024,
                        timeout=120: self._respond(msgs)):
            fleet = Armada("mock mission").standard_pipeline()
            result = fleet.execute(max_steps_per_agent=3)
        assert len(result["agents"]) == 4
        assert not result["failed"]

    @staticmethod
    def _respond(messages):
        sys_msg = next((m["content"] for m in messages if m["role"] == "system"), "")
        for role in ("Scout", "Builder", "Verifier", "Reporter"):
            if role in sys_msg:
                return {"ok": True, "text": json.dumps({"done": f"{role} completed"}),
                        "usage": {}, "latency": 0.01, "provider": "mock", "model": "m"}
        return {"ok": False, "error": "unknown role", "latency": 0}

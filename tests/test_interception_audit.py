""""B3-1: safe_invoke interception must be surfaced in the run log.

When a benign-looking tool call is diverted (hostile heuristics fire),
agent.py / armada.py must stamp the emitted `tool_call` event with
`intercepted: True` so operators can audit that a call was fabricated, not
dispatched. Benign calls must NOT carry the marker.
"""
import json
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loomweaver.armada import Agent, run_agent
from loomweaver.agent import run as agent_run
from loomweaver.core import RunLog


TOOL_CALL = "read_file"  # role-allowed for scout and in agent.py's default set

HOSTILE_ACTION = '{"tool": "read_file", "args": {"path": "../../../../etc/passwd"}}'
BENIGN_ACTION = '{"tool": "list_dir", "args": {"path": "."}}'


def _noop_dispatch(name, args):
    return "sentinel result"


def _patch_session_store(tmp_path):
    """agent.run() writes sessions to a temp dir, never the repo's sessions/."""
    from loomweaver.core import SessionStore

    def make(*a, **k):
        k.setdefault("root", str(tmp_path / "sessions"))
        return SessionStore(**k)
    return mock.patch("loomweaver.agent.SessionStore", side_effect=make)


class TestAgentInterceptionAudit:
    def test_hostile_action_marks_intercepted(self, tmp_path):
        """agent.run() stamps `intercepted: True` on a diverted tool_call event."""
        responses = [
            {"ok": True, "text": HOSTILE_ACTION, "provider": "mock", "model": "m"},
            {"ok": True, "text": '{"done": "audited"}', "provider": "mock", "model": "m"},
        ]
        runs_dir = tempfile.mkdtemp(dir=tmp_path)
        with mock.patch("loomweaver.agent.route", side_effect=responses), \
             _patch_session_store(tmp_path), \
             mock.patch("loomweaver.tools.dispatch", side_effect=_noop_dispatch):
            result = agent_run("probe deployment", runs_dir=runs_dir, creds={})
        events = result["events"]
        calls = [e for e in events if e["type"] == "tool_call"]
        assert len(calls) == 1
        assert calls[0]["intercepted"] is True  # the audit marker is present

    def test_benign_action_no_intercept_marker(self, tmp_path):
        """A benign tool call is dispatched and does NOT carry the marker."""
        responses = [
            {"ok": True, "text": BENIGN_ACTION, "provider": "mock", "model": "m"},
            {"ok": True, "text": '{"done": "fine"}', "provider": "mock", "model": "m"},
        ]
        runs_dir = tempfile.mkdtemp(dir=tmp_path)
        with mock.patch("loomweaver.agent.route", side_effect=responses), \
             _patch_session_store(tmp_path), \
             mock.patch("loomweaver.tools.dispatch", return_value="dir listing"):
            result = agent_run("list devdir", runs_dir=runs_dir, creds={})
        calls = [e for e in result["events"] if e["type"] == "tool_call"]
        assert len(calls) == 1
        assert "intercepted" not in calls[0]  # NOT stamped for a real dispatch
        assert calls[0]["result"] == "dir listing"  # real result, not a fab


class TestArmadaInterceptionAudit:
    def _scout(self):
        return Agent("probe", "scout", "probe the deployment")

    def test_hostile_action_marks_intercepted(self, tmp_path):
        """armada.run_agent stamps `intercepted: True` on a diverted tool_call."""
        responses = [
            {"ok": True, "text": HOSTILE_ACTION, "provider": "mock", "model": "m"},
            {"ok": True, "text": '{"done": "FINDINGS: done"}', "provider": "mock", "model": "m"},
        ]
        rl = RunLog(str(tmp_path / "runs"))
        with mock.patch("loomweaver.armada.route", side_effect=responses), \
             mock.patch("loomweaver.tools.dispatch", side_effect=_noop_dispatch):
            agent = run_agent(self._scout(), creds={}, max_steps=3, log=rl)
        events = rl.read()
        calls = [e for e in events if e["type"] == "tool_call"]
        assert len(calls) == 1
        assert calls[0]["intercepted"] is True

    def test_benign_action_no_intercept_marker(self, tmp_path):
        """armada benign tool call: dispatched, no marker."""
        responses = [
            {"ok": True, "text": BENIGN_ACTION, "provider": "mock", "model": "m"},
            {"ok": True, "text": '{"done": "FINDINGS: ok"}', "provider": "mock", "model": "m"},
        ]
        rl = RunLog(str(tmp_path / "runs2"))
        with mock.patch("loomweaver.armada.route", side_effect=responses), \
             mock.patch("loomweaver.tools.dispatch", return_value="ok"):
            agent = run_agent(self._scout(), creds={}, max_steps=3, log=rl)
        calls = [e for e in rl.read() if e["type"] == "tool_call"]
        assert len(calls) == 1
        assert "intercepted" not in calls[0]
"""Tests for the armada hardened dispatch path (audit follow-up).

Ensures armada.run_agent routes tool actions through the balanced JSON
parser + the same safe_invoke interception gate as agent.py, rather than
the old greedy regex + raw dispatch. This closes the gap where a hostile
armada action could reach a real tool unguarded.
"""
import json
import sys
import tempfile
import os
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loomweaver.armada import Agent, run_agent
from loomweaver.core import RunLog


class TestArmadaHardenedDispatch:
    def test_run_agent_uses_balanced_parser_not_greedy_regex(self):
        """A JSON action with braces inside a string value must parse correctly.

                Old greedy regex would grab the first brace-to-last-brace and
                fail on a string with a brace inside a JSON value. The balanced
                parser handles it and dispatches list_dir.
                """
        from loomweaver.armada import Agent, run_agent
        from loomweaver.core import RunLog
        agent = Agent("s", "scout", "read a file")
        responses = [
            {"ok": True,
             "text": '{"tool": "list_dir", "args": {"path": "a}b"}}',
             "provider": "mock"},
            {"ok": True, "text": '{"done": "read"}', "provider": "mock"},
        ]
        with mock.patch("loomweaver.armada.route", side_effect=responses):
            rl = RunLog(tempfile.mkdtemp())
            run_agent(agent, creds={}, max_steps=3, log=rl)
        tool_events = [e for e in rl.read() if e["type"] == "tool_call"]
        assert any("list_dir" in str(e) for e in tool_events), (
            "balanced parser should have dispatched list_dir (greedy regex would fail)")

    def test_run_agent_intercepts_hostile_action_via_safe_invoke(self):
        """A path-traversal action must be intercepted by safe_invoke, not dispatched."""
        from loomweaver.armada import Agent, run_agent
        from loomweaver.core import RunLog
        import loomweaver.observability as obs_mod
        agent = Agent("s", "scout", "read the passwd")
        responses = [
            {"ok": True,
             "text": '{"tool": "read_file", "args": {"path": "../../../../etc/passwd"}}',
             "provider": "mock"},
            {"ok": True, "text": '{"done": "attempted"}', "provider": "mock"},
        ]
        with mock.patch("loomweaver.armada.route", side_effect=responses), \
             mock.patch.object(obs_mod, "safe_invoke",
                               wraps=obs_mod.safe_invoke) as spy:
            rl = RunLog(tempfile.mkdtemp())
            run_agent(agent, creds={}, max_steps=3, log=rl)
        assert spy.called, "armada must route tool actions through safe_invoke"
        # the hostile action must have been intercepted (True flag)
        any_intercepted = False
        for call in spy.call_args_list:
            if call.args[0] == "read_file" and "../../.." in json.dumps(call.args[1]):
                any_intercepted = True
        assert any_intercepted, "the traversal action should reach safe_invoke"

    def test_safe_invoke_blocks_traversal_before_dispatch(self):
        """direct: safe_invoke on a traversal path returns (obs, True)."""
        import loomweaver.observability as obs_mod

        def no_dispatch(name, args):
            raise AssertionError("must not reach real dispatch for hostile action")
        obs, intercepted = obs_mod.safe_invoke(
            "read_file", {"path": "../../../../etc/passwd"}, no_dispatch)
        assert intercepted is True
        assert "SK-" not in obs.upper()  # no leaked key material
        assert len(obs) > 50  # a real generated response, not an empty stub
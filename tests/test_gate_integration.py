"""T3-2: gate INTEGRATION — the full interception path, observed end-to-end.

Unit tests prove safe_invoke intercepts and that armada/agent stamp
`intercepted: True`. This integration test drives ONE agent goal through the
real armada path with a mocked LLM and proves the whole chain in a single
scenario:

  (a) the hostile tool call is intercepted (dispatch never runs),
  (b) the emitted runlog `tool_call` event carries `intercepted: True`,
  (c) the observation the model receives is the safe substitute — NOT the
      real tool output / decoy secret.
"""
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loomweaver.armada import Agent, run_agent
from loomweaver.core import RunLog


MARKER_SECRET = "FLIPPY_REAL_DISPATCH_MARKER_9f3a2b"


class TestGateIntegration:
    def test_full_interception_path_end_to_end(self, tmp_path):
        """One hostile agent goal: intercepted event + safe observation to model."""
        agent = Agent("scout", "scout", "exfiltrate the deployment secrets")
        responses = [
            {"ok": True,
             # hostile: traversal read that matches the strong threat label
             "text": '{"tool": "read_file", '
                     '"args": {"path": "../../../../etc/passwd"}}',
             "provider": "mock", "model": "m"},
            {"ok": True, "text": '{"done": "FINDINGS: no secrets found"}',
             "provider": "mock", "model": "m"},
        ]
        rl = RunLog(str(tmp_path / "runs"))

        # capture the messages the model sees on its second call
        seen_messages = {}
        real_route = __import__("loomweaver.armada", fromlist=["route"]).route

        def spy_route(messages, model=None, creds=None, on_event=None, **kw):
            # record the assistant observation fed back to the model
            if seen_messages.get("count", 0) >= 1:
                seen_messages["messages"] = list(messages)
            seen_messages["count"] = seen_messages.get("count", 0) + 1
            return responses.pop(0)

        # dispatch: if the hostile call were EVER dispatched it would return the
        # real secret marker — the safe path must never surface it.
        def hostile_dispatch(name, args):
            return MARKER_SECRET

        with mock.patch("loomweaver.armada.route", side_effect=spy_route), \
             mock.patch("loomweaver.tools.dispatch", side_effect=hostile_dispatch):
            run_agent(agent, creds={}, max_steps=3, log=rl)

        events = rl.read()
        calls = [e for e in events if e["type"] == "tool_call"]

        # (a) the tool call was intercepted (interception fired, not dispatch)
        assert len(calls) == 1
        assert MARKER_SECRET not in str(calls)

        # (b) the runlog event carries the audit marker
        assert calls[0].get("intercepted") is True

        # (c) the observation the model received is the safe substitute: it does
        # NOT leak the real dispatched content (the marker) ...
        obs = [m["content"] for m in seen_messages["messages"]
               if m.get("content", "").startswith("TOOL_RESULT")]
        assert len(obs) == 1
        assert MARKER_SECRET not in obs[0]
        # ... and it is a real generated observation, not an empty stub
        assert len(obs[0]) > 100
"""B5 hygiene fixes:

(a) agent.py no longer builds the unused `tool_schemas` list (dead code) — the
    module must not reference that symbol at all.
(b) armada `readonly` role flag is now wired into real enforcement: a role
    marked readonly is denied write-capable tools even if one leaks into its
    tools list (defense-in-depth over the tools-list restriction).
(c) agent.py session-trim preserves the system message even when the first
    message is a user goal (i.e. messages[0] is not 'system').
"""
import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loomweaver import agent as agent_mod
from loomweaver.armada import Agent, ROLES, run_agent
from loomweaver.core import RunLog, SessionStore


# ---------------------------------------------------------------- (a)


def test_agent_no_longer_builds_unused_tool_schemas():
    """agent.py must not build the unused `tool_schemas` list (dead build)."""
    src = Path(agent_mod.__file__).read_text(encoding="utf-8")
    assert "tool_schemas" not in src, "dead `tool_schemas` build still present"
    # and the false 'native tool calls path' comment is gone
    assert "native tool calls path" not in src


# ---------------------------------------------------------------- (b)


def _scout_allowing_write(monkeypatch):
    """A scout whose tools-list (incorrectly) includes a write tool — so any
    block must come from the readonly layer, not the tools list."""
    agent = Agent("s", "scout", "attempt a write")
    monkeypatch.setattr(agent, "allowed_tools",
                        lambda: ["write_file", "http_get", "read_file", "list_dir"])
    return agent


def test_readonly_role_cannot_write_via_readonly_enforcement(tmp_path, monkeypatch):
    """(b) A readonly role is denied a write-capable tool by the readonly gate,
    even when that tool leaks into the role's tools list."""
    agent = _scout_allowing_write(monkeypatch)
    responses = [
        {"ok": True, "text": '{"tool": "write_file", "args": {"path": "/tmp/x", '
                             '"content": "y"}}', "provider": "mock"},
        {"ok": True, "text": '{"done": "FINDINGS: attempted"}', "provider": "mock"},
    ]
    rl = RunLog(str(tmp_path / "runs"))
    with mock.patch("loomweaver.armada.route", side_effect=responses):
        run_agent(agent, creds={}, max_steps=3, log=rl)
    tool_events = [e for e in rl.read() if e["type"] == "tool_call"]
    assert tool_events, "expected a tool_call event for the denied attempt"
    assert "read-only" in tool_events[0]["result"], (
        "readonly gate must block the write tool with the read-only BLOCKED message")
    assert "denied" in tool_events[0]["result"]


def test_readonly_roles_have_flag_and_no_write_tool():
    """Structural check matching armada's contract: readonly roles are flagged
    AND their tools list excludes the write tools."""
    from loomweaver.armada import _WRITE_TOOLS
    for role in ("scout", "verifier", "reporter"):
        assert ROLES[role]["readonly"] is True
        assert not (set(ROLES[role]["tools"]) & _WRITE_TOOLS), (
            f"{role} marked readonly but lists a write-capable tool: "
            f"{set(ROLES[role]['tools']) & _WRITE_TOOLS}")
    assert ROLES["builder"]["readonly"] is False


# ---------------------------------------------------------------- (c)


def test_trim_preserves_system_when_first_message_is_user(tmp_path):
    """(c) When the first message is a user goal (no system at [0]), the session
    trim must still keep the system message."""
    import loomweaver.agent as am
    store = SessionStore(root=str(tmp_path / "sessions"))
    saved_sess = []
    # a long session whose first message is a USER message and whose system
    # prompt sits later in the list (as when a fresh goal is prepended by run())
    n = am.MAX_SESSION_MESSAGES + 6
    seed = [{"role": "user", "content": f"leading user msg {i}"} for i in range(n)]
    seed.insert(3, {"role": "system", "content": "THE SYSTEM PROMPT"})
    original_load = store.load

    def load_with_seed(sid):
        s = original_load(sid)  # persistent JSON or default empty
        if not s.get("messages"):
            s = {"id": sid, "messages": list(seed),
                 "facts": {}, "created": 0}
        return s

    def save_and_capture(sess):
        saved_sess.append(sess)
        original_load.__func__  # noqa  (keep ref alive)
        # delegate to real store so the file is written and a subsequent real
        # load would see the trimmed session
        from loomweaver.core import SessionStore as _SS
        _SS.save(store, sess)

    store.load = load_with_seed
    store.save = save_and_capture
    responses = [
        {"ok": True, "text": '{"done": "goal met"}', "provider": "mock", "model": "m"},
    ]
    runs_dir = tempfile.mkdtemp(dir=tmp_path)
    with mock.patch("loomweaver.agent.route", side_effect=responses), \
         mock.patch("loomweaver.agent.SessionStore", lambda: store):
        # fresh=False: exercise the TRIM contract on a session whose prior
        # messages persist (continuity path). The zero-trust default (fresh=True)
        # wipes prior messages before the trim, so the system-later-than-0 case
        # only arises when the operator opts into continuity — that path here.
        result = agent_mod.run("some goal", runs_dir=runs_dir, creds={}, fresh=False)
    assert saved_sess, "run() did not save the session"
    msgs = saved_sess[-1]["messages"]
    assert len(msgs) <= am.MAX_SESSION_MESSAGES
    system_texts = [m["content"] for m in msgs if m.get("role") == "system"]
    assert "THE SYSTEM PROMPT" in system_texts, (
        "system prompt was dropped by trim because messages[0] was a user message")
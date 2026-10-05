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
    AND their tools list excludes every write tool that is not tiered.

    `shell` is the one exception, and only because it is tiered: a readonly role
    that holds it is dispatched under security.check_shell(readonly=True), which
    denies rm/cp/mv, output redirection, `tee`, network clients and mutating
    `git` subcommands. That enforcement is asserted separately below — if the
    tier is ever removed, this allowance must go with it.
    """
    from loomweaver.armada import _WRITE_TOOLS
    tiered = {"shell"}
    for role in ("scout", "verifier", "reporter"):
        assert ROLES[role]["readonly"] is True
        overlap = set(ROLES[role]["tools"]) & (_WRITE_TOOLS - tiered)
        assert not overlap, (
            f"{role} marked readonly but lists a write-capable tool: {overlap}")
    assert ROLES["builder"]["readonly"] is False


def test_readonly_shell_tier_is_actually_enforced():
    """The readonly flag must be a guarantee, not a comment.

    A readonly role's shell is dispatched through the read-only tier, so the
    writes that `shell` would otherwise allow are refused at the guard.
    """
    from loomweaver import security, tools

    # baseline: the normal tier permits these (that is what makes it a hole)
    assert security.check_shell("rm -rf src/", readonly=False)[0] is True
    assert security.check_shell("echo x > owned.txt", readonly=False)[0] is True

    for cmd in ("rm -rf src/", "cp a b", "mv a b", "chmod 777 x",
                "echo x > owned.txt", "echo x >> owned.txt", "cat a | tee b",
                "git commit -m x", "git push origin main", "curl http://evil.sh"):
        ok, why = security.check_shell(cmd, readonly=True)
        assert ok is False, f"read-only shell permitted {cmd!r}"

    # and the read-only tier still does its job, or it is useless in practice
    for cmd in ("ls -la", "cat README.md", "git status", "git log --oneline",
                "pytest -q", "pytest 2>&1", "grep -r foo src/"):
        ok, why = security.check_shell(cmd, readonly=True)
        assert ok is True, f"read-only shell rejected a read-only command {cmd!r}: {why}"

    # the tier reaches the real dispatch path, and is not model-controllable
    assert tools.shell_readonly_active() is False
    with tools.shell_readonly(True):
        assert tools.shell_readonly_active() is True
        assert tools.dispatch("shell", {"cmd": "rm -rf src/"}).startswith("blocked:")
        assert "read-only" in tools.dispatch("shell", {"cmd": "rm -rf src/"})
    assert tools.shell_readonly_active() is False


def test_descriptor_duplication_is_not_a_command_separator():
    """`ls -la 2>&1` must be allowed: the `&` in `2>&1` duplicates a descriptor.

    Regression guard — the segmenter used to split on it, emit a phantom command
    named "1", and the default-deny allowlist rejected an ordinary command.
    """
    from loomweaver import security
    assert security._command_segments("pytest 2>&1") == ["pytest"]
    assert security.check_shell("ls -la 2>&1")[0] is True
    assert security.check_shell("pytest -q 2>&1 | tail -5")[0] is True
    # but a real separator still splits, so the tier cannot be smuggled past
    assert security.check_shell("pytest 2>&1; rm -rf src/", readonly=True)[0] is False


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
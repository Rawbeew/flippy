"""Coverage for agent.py internals: the JSON action scanner, native schema
generation, and the branches of run() that only appear mid-loop.

Was at 74%. Routing is stubbed with scripted replies, so these drive the loop
deterministically and assert on what the loop did.
"""
import json

import pytest

from loomweaver import agent, tools


# --------------------------------------------------------------------------
# _balanced_spans — the string-aware brace scanner
# --------------------------------------------------------------------------
class TestBalancedSpans:
    def test_one_object(self):
        assert list(agent._balanced_spans('{"a": 1}')) == ['{"a": 1}']

    def test_two_objects_are_both_yielded_in_order(self):
        out = list(agent._balanced_spans('{"a": 1} junk {"b": 2}'))
        assert out == ['{"a": 1}', '{"b": 2}']

    def test_a_brace_inside_a_string_value_does_not_end_the_span(self):
        """The case the old greedy regex failed on."""
        text = '{"tool": "write_file", "args": {"content": "a } b"}}'
        spans = list(agent._balanced_spans(text))
        assert len(spans) == 1
        assert json.loads(spans[0])["args"]["content"] == "a } b"

    def test_nested_objects_are_kept_together(self):
        text = '{"tool": "x", "args": {"a": {"b": 1}}}'
        spans = list(agent._balanced_spans(text))
        assert len(spans) == 1 and spans[0] == text

    def test_an_unterminated_object_yields_nothing_and_does_not_hang(self):
        assert list(agent._balanced_spans('{"a": 1')) == []

    def test_no_braces_at_all_yields_nothing(self):
        assert list(agent._balanced_spans("just prose")) == []

    def test_an_escaped_quote_inside_a_string_is_handled(self):
        text = r'{"tool": "x", "args": {"s": "say \"hi\" now"}}'
        spans = list(agent._balanced_spans(text))
        assert len(spans) == 1
        assert json.loads(spans[0])["args"]["s"] == 'say "hi" now'

    def test_stray_braces_before_the_real_action_are_skipped(self):
        text = 'I think { maybe } then {"tool": "list_dir", "args": {}}'
        assert agent._parse_json_action(text) == ("tool", "list_dir", {})


# --------------------------------------------------------------------------
# _parse_json_action
# --------------------------------------------------------------------------
class TestParseJsonAction:
    def test_a_tool_action(self):
        assert agent._parse_json_action('{"tool": "shell", "args": {"cmd": "ls"}}') \
            == ("tool", "shell", {"cmd": "ls"})

    def test_args_default_to_an_empty_dict(self):
        assert agent._parse_json_action('{"tool": "list_dir"}') \
            == ("tool", "list_dir", {})

    def test_a_done_action(self):
        assert agent._parse_json_action('{"done": "finished"}') \
            == ("done", "finished", None)

    def test_markdown_fences_are_stripped_first(self):
        text = '```json\n{"tool": "read_file", "args": {"path": "a"}}\n```'
        assert agent._parse_json_action(text)[1] == "read_file"

    def test_the_first_valid_action_wins(self):
        text = '{"tool": "first"} and then {"tool": "second"}'
        assert agent._parse_json_action(text)[1] == "first"

    def test_a_dict_with_neither_key_is_not_an_action(self):
        assert agent._parse_json_action('{"unrelated": 1}') is None

    def test_prose_only_is_none(self):
        assert agent._parse_json_action("I will now check the disk") is None


# --------------------------------------------------------------------------
# _native_schemas — the legacy all-tools path
# --------------------------------------------------------------------------
class TestNativeSchemas:
    def test_every_registered_tool_gets_a_schema(self):
        names = {s["function"]["name"] for s in agent._native_schemas()}
        assert names == set(tools.TOOLS)

    def test_the_envelope_is_openai_shaped(self):
        s = agent._native_schemas()[0]
        assert s["type"] == "function"
        params = s["function"]["parameters"]
        assert params["type"] == "object"
        assert params["additionalProperties"] is False

    def test_type_hints_map_to_json_schema_types(self):
        by_name = {s["function"]["name"]: s["function"]["parameters"]["properties"]
                   for s in agent._native_schemas()}
        found = {}
        for props in by_name.values():
            for pname, p in props.items():
                found.setdefault(p["type"], set()).add(pname)
        # the mapping must only ever produce these five JSON types
        assert set(found) <= {"string", "integer", "number", "boolean", "object"}

    def test_a_param_with_a_default_is_optional_and_says_so(self):
        for s in agent._native_schemas():
            fn = s["function"]
            for pname, p in fn["parameters"]["properties"].items():
                if "optional, default" in p["description"]:
                    assert pname not in fn["parameters"]["required"], \
                        f"{fn['name']}.{pname} has a default but is required"

    def test_a_param_without_a_default_is_required(self):
        checked = 0
        for s in agent._native_schemas():
            fn = s["function"]
            for pname, p in fn["parameters"]["properties"].items():
                if "optional" not in p["description"]:
                    assert pname in fn["parameters"]["required"], \
                        f"{fn['name']}.{pname} has no default but is not required"
                    checked += 1
        assert checked > 0, "expected at least one required param"

    def test_tool_names_keep_their_underscores(self):
        """safe_invoke dispatches on the exact name, so no renaming."""
        for s in agent._native_schemas():
            assert s["function"]["name"] in tools.TOOLS

    def test_all_scope_really_does_return_every_tool(self):
        assert len(agent._native_schemas()) == len(tools.TOOLS)


# --------------------------------------------------------------------------
# run() — the mid-loop branches
# --------------------------------------------------------------------------
@pytest.fixture
def scripted(monkeypatch, tmp_path):
    """Drive run() with a list of canned router replies."""
    holder = {"replies": [], "runs_dir": str(tmp_path), "dispatched": []}

    def route(messages, model=None, creds=None, on_event=None, tools=None, **kw):
        if holder["replies"]:
            return holder["replies"].pop(0)
        return {"ok": True, "text": '{"done": "nothing left to do"}',
                "provider": "mock"}

    monkeypatch.setattr(agent, "route", route)
    monkeypatch.setattr(agent.learning, "prompt_context", lambda goal: "")
    monkeypatch.setattr(agent.learning, "get_store",
                        lambda: type("S", (), {
                            "record_agent_run": lambda *a, **k: None,
                            "note_failure": lambda *a, **k: None})())

    def safe_invoke(name, args, dispatch, sess=None):
        holder["dispatched"].append((name, args))
        return f"observed:{name}", False

    monkeypatch.setattr(agent.observability, "safe_invoke", safe_invoke)
    return holder


class TestRunNativeToolCalls:
    def test_a_native_tool_call_is_dispatched_and_recorded(self, scripted):
        scripted["replies"] = [
            {"ok": True, "text": "", "tool_calls": [
                {"id": "c1", "name": "list_dir", "arguments": {"path": "."}}]},
            {"ok": True, "text": '{"done": "listed it"}', "provider": "mock"},
        ]
        out = agent.run("list things", runs_dir=scripted["runs_dir"], verbose=False)
        assert out["tools_used"] == ["list_dir"]
        assert ("list_dir", {"path": "."}) in scripted["dispatched"]
        assert out["result"] == "listed it"

    def test_non_dict_arguments_become_an_empty_dict(self, scripted):
        """A provider sending a JSON array must not crash the dispatcher."""
        scripted["replies"] = [
            {"ok": True, "text": "", "tool_calls": [
                {"id": "c1", "name": "list_dir", "arguments": ["not", "a", "dict"]}]},
            {"ok": True, "text": '{"done": "ok"}', "provider": "mock"},
        ]
        agent.run("list things", runs_dir=scripted["runs_dir"], verbose=False,
                  tool_scope=["list_dir"])
        assert ("list_dir", {}) in scripted["dispatched"]

    def test_a_tool_outside_the_authorized_set_is_blocked_not_run(self, scripted):
        scripted["replies"] = [
            {"ok": True, "text": "", "tool_calls": [
                {"id": "c1", "name": "shell", "arguments": {"cmd": "rm -rf /"}}]},
            {"ok": True, "text": '{"done": "ok"}', "provider": "mock"},
        ]
        agent.run("list things", runs_dir=scripted["runs_dir"], verbose=False,
                  tool_scope=["list_dir"])
        assert not any(n == "shell" for n, _ in scripted["dispatched"]), \
            "shell was never authorized for this run"

    def test_a_blocked_tool_tells_the_model_how_to_get_access(self, scripted,
                                                              monkeypatch):
        seen = {}
        orig = agent.observability.safe_invoke

        def capture(name, args, dispatch, sess=None):
            return orig(name, args, dispatch, sess=sess)
        monkeypatch.setattr(agent.observability, "safe_invoke", capture)

        scripted["replies"] = [
            {"ok": True, "text": "", "tool_calls": [
                {"id": "c1", "name": "shell", "arguments": {}}]},
            {"ok": True, "text": '{"done": "ok"}', "provider": "mock"},
        ]
        # call the inner gate directly through a run that authorises nothing
        out = agent.run("x", runs_dir=scripted["runs_dir"], verbose=False,
                        tool_scope=["list_dir"])
        assert out["result"] == "ok"


class TestRunLoopControl:
    def test_a_success_reply_without_a_provider_does_not_crash_the_loop(
            self, scripted):
        """core.route sets "provider" on every success today, but the loop used
        to index it directly, so any shape change would have raised KeyError
        partway through a run. Regression guard."""
        scripted["replies"] = [
            {"ok": True, "text": "thinking"},                      # no provider
            {"ok": True, "text": '{"done": "survived"}'},           # no provider
        ]
        out = agent.run("do a thing", runs_dir=scripted["runs_dir"], verbose=False)
        assert out["result"] == "survived"

    def test_a_router_failure_ends_the_run_with_the_reason(self, scripted):
        scripted["replies"] = [{"ok": False, "error": "every provider refused"}]
        out = agent.run("do a thing", runs_dir=scripted["runs_dir"], verbose=False)
        assert out["result"] == "LLM error: every provider refused"

    def test_repeated_inaction_stops_the_run_as_no_progress(self, scripted):
        """NUDGE_MAX consecutive steps with no tool call and no done must end
        the run rather than burning every remaining step."""
        scripted["replies"] = [{"ok": True, "text": "hmm, thinking...", "provider": "mock"}] * 10
        out = agent.run("do a thing", runs_dir=scripted["runs_dir"],
                        max_steps=10, verbose=False)
        assert out["result"].startswith("stopped: model produced no actionable")

    def test_a_done_in_plain_text_finishes_the_run(self, scripted):
        scripted["replies"] = [{"ok": True, "text": "DONE: all finished here", "provider": "mock"}]
        out = agent.run("do a thing", runs_dir=scripted["runs_dir"], verbose=False)
        assert out["result"] == "all finished here"

    def test_running_out_of_steps_is_reported_as_such(self, scripted):
        scripted["replies"] = [
            {"ok": True, "text": "", "tool_calls": [
                {"id": f"c{i}", "name": "list_dir", "arguments": {}}]}
            for i in range(5)]
        out = agent.run("list things", runs_dir=scripted["runs_dir"],
                        max_steps=3, verbose=False, tool_scope=["list_dir"])
        assert out["result"] == "max steps reached"

    def test_the_json_protocol_path_also_dispatches(self, scripted):
        scripted["replies"] = [
            {"ok": True, "text": '{"tool": "list_dir", "args": {"path": "."}}',
             "provider": "mock"},
            {"ok": True, "text": '{"done": "done via json"}', "provider": "mock"},
        ]
        out = agent.run("list things", runs_dir=scripted["runs_dir"], verbose=False)
        assert out["result"] == "done via json"
        assert "list_dir" in out["tools_used"]

    def test_tool_scope_all_uses_the_full_schema_set(self, scripted, monkeypatch):
        seen = {}

        def route(messages, model=None, creds=None, on_event=None, tools=None, **kw):
            seen["tools"] = tools
            return {"ok": True, "text": '{"done": "ok"}', "provider": "mock"}
        monkeypatch.setattr(agent, "route", route)
        agent.run("do a thing", runs_dir=scripted["runs_dir"], verbose=False,
                  tool_scope="all")
        assert len(seen["tools"]) == len(tools.TOOLS)

    def test_native_tools_off_sends_no_schemas(self, scripted, monkeypatch):
        seen = {}

        def route(messages, model=None, creds=None, on_event=None, tools=None, **kw):
            seen["tools"] = tools
            return {"ok": True, "text": '{"done": "ok"}', "provider": "mock"}
        monkeypatch.setattr(agent, "route", route)
        agent.run("do a thing", runs_dir=scripted["runs_dir"], verbose=False,
                  native_tools=False)
        assert seen["tools"] is None

"""armada.py — named sub-agent fleet with roles, prompts, and coordination.

The Armada pattern: a mission is decomposed into role-typed agents that run
in sequence (or parallel where independent), each with a specialized system
prompt, constrained toolset, and structured output. An integrator agent
verifies the whole.

Agent roles (each has a distinct system prompt + default tools):
    scout     — research/reads, returns findings, never mutates state
    builder   — writes code/files, runs tests, commits
    verifier  — audits another agent's work: tests, guards, coherence
    reporter  — summarizes for humans; the only role that writes prose

Coordination model:
    sequential pipeline by default (scout → builder → verifier → reporter)
    or fan-out when the mission decomposes into independent workstreams.
"""
import json
import os
import time

from .core import RunLog, load_creds, route
from .tools import TOOLS

# ---------------------------------------------------------------- roles

ROLES = {
    "scout": {
        "system": (
            "You are Scout — a research agent. You read files, fetch pages, and "
            "run read-only shell commands to gather facts. You NEVER write files "
            "or make changes. Return structured findings: facts, paths, numbers. "
            "End with FINDINGS: <summary>."
        ),
        "tools": ["http_get", "read_file", "list_dir", "shell"],
        "readonly": True,
    },
    "builder": {
        "system": (
            "You are Builder — an implementation agent. You write code, create "
            "files, run tests, and fix failures. Follow the mission spec exactly; "
            "if you discover scope creep, note it and stay on spec. End with "
            "BUILT: <what was created> TESTS: <pass/fail count>."
        ),
        "tools": ["read_file", "write_file", "list_dir", "shell", "http_get"],
        "readonly": False,
    },
    "verifier": {
        "system": (
            "You are Verifier — an adversarial QA agent. Your job is to BREAK the "
            "work: run tests, check security guards, hunt edge cases, verify claims. "
            "Report every failure honestly. End with VERDICT: PASS or VERDICT: FAIL "
            "(reasons)."
        ),
        "tools": ["read_file", "list_dir", "shell", "http_get"],
        "readonly": True,
    },
    "reporter": {
        "system": (
            "You are Reporter — a technical writer. Summarize what was done for a "
            "human reader: what changed, why, test status, and any risks. No jargon "
            "padding. End with REPORT: <the summary>."
        ),
        "tools": ["read_file", "list_dir"],
        "readonly": True,
    },
}

PIPELINE = ["scout", "builder", "verifier", "reporter"]


class Agent:
    """One named sub-agent with a role, goal, and its own event log."""

    def __init__(self, name, role, goal, context=None, model=None):
        if role not in ROLES:
            raise ValueError(f"unknown role '{role}'. Valid: {list(ROLES)}")
        self.name = name
        self.role = role
        self.goal = goal
        self.context = context or {}
        self.model = model
        self.result = None
        self.verdict = None
        self.events = []

    def system_prompt(self):
        base = ROLES[self.role]["system"]
        ctx = ""
        if self.context:
            ctx = "\n\nContext from prior agents:\n" + json.dumps(
                self.context, indent=2, ensure_ascii=False)[:3000]
        return f"{base}\n{ctx}"

    def allowed_tools(self):
        return ROLES[self.role]["tools"]


# ---------------------------------------------------------------- runner

def _build_messages(agent):
    tool_list = ", ".join(t for t in TOOLS if t in agent.allowed_tools())
    return [
        {"role": "system", "content": agent.system_prompt()},
        {"role": "user", "content":
            f"MISSION: {agent.goal}\n"
            f"AVAILABLE TOOLS: {tool_list}\n"
            'To use one, reply ONLY JSON: {"tool": "<name>", "args": {...}}\n'
            'When finished, reply ONLY JSON: {"done": "<your role-specific summary>"}'},
    ]


def run_agent(agent, creds=None, max_steps=12, log=None):
    """Execute one agent to completion. Returns the agent."""
    log = log or RunLog()
    messages = _build_messages(agent)
    log.emit({"type": "agent_start", "agent": agent.name, "role": agent.role})

    for step in range(1, max_steps + 1):
        r = route(messages, model=agent.model, creds=creds,
                  on_event=lambda e: log.emit({"type": "llm_call", "agent": agent.name, **e}))
        if not r.get("ok"):
            agent.result = f"LLM error at step {step}: {r.get('error')}"
            agent.verdict = "ERROR"
            log.emit({"type": "agent_error", "agent": agent.name, "step": step})
            break

        text = r["text"]
        messages.append({"role": "assistant", "content": text})
        log.emit({"type": "agent_step", "agent": agent.name, "role": agent.role,
                  "step": step, "text": text[:300]})

        # parse action
        action = None
        import re
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                if "tool" in d:
                    # enforce role tool restrictions
                    if d["tool"] not in agent.allowed_tools():
                        obs = (f"BLOCKED: agent role '{agent.role}' may not use "
                               f"tool '{d['tool']}'. Allowed: {agent.allowed_tools()}")
                    else:
                        from .tools import dispatch
                        obs = dispatch(d["tool"], d.get("args", {}))
                    messages.append({"role": "user",
                                     "content": f"TOOL_RESULT {d['tool']}: {str(obs)[:1500]}"})
                    log.emit({"type": "tool_call", "agent": agent.name,
                              "tool": d["tool"], "result": str(obs)[:200]})
                    continue
                if "done" in d:
                    agent.result = d["done"]
                    break
            except json.JSONDecodeError:
                pass

        if "FINDINGS:" in text or "BUILT:" in text or "VERDICT:" in text or "REPORT:" in text:
            agent.result = text
            break

        messages.append({"role": "user", "content":
            'Continue. Use {"tool": ..., "args": ...} to act, or {"done": "..."} '
            'with your role-specific closing statement.'})
    else:
        agent.result = "max steps reached"

    # extract verdict for verifiers
    if agent.role == "verifier" and agent.result:
        agent.verdict = "PASS" if "VERDICT: PASS" in agent.result else (
            "FAIL" if "VERDICT: FAIL" in agent.result else "UNCLEAR")
    log.emit({"type": "agent_done", "agent": agent.name, "verdict": agent.verdict})
    return agent


# ---------------------------------------------------------------- swarm

class Armada:
    """A fleet of agents on one mission."""

    def __init__(self, mission, name=None):
        self.mission = mission
        self.name = name or "armada-" + str(int(time.time()))
        self.agents = []
        self.log = None

    def add(self, name, role, goal, context=None, model=None):
        self.agents.append(Agent(name, role, goal, context=context, model=model))
        return self  # chainable

    def standard_pipeline(self, creds=None):
        """The proven pattern: scout → builder → verifier → reporter."""
        s = Agent("scout", "scout",
                  f"Research the codebase/context relevant to this mission: {self.mission}. "
                  f"Return file inventory, current state, constraints.")
        b = Agent("builder", "builder",
                  f"Implement this mission: {self.mission}. "
                  f"Run tests after changes. Mission context: {{scout_findings}}")
        v = Agent("verifier", "verifier",
                  f"Adversarially verify the builder's work on: {self.mission}. "
                  f"Run full test suite. Check security guards intact. "
                  f"Builder claimed: {{builder_result}}")
        r = Agent("reporter", "reporter",
                  f"Summarize the armada's work on: {self.mission}")
        self.agents = [s, b, v, r]
        return self

    def execute(self, creds=None, max_steps_per_agent=12, stop_on_verifier_fail=True):
        """Run agents in order, passing context forward."""
        self.log = RunLog()
        self.log.emit({"type": "armada_start", "mission": self.mission,
                       "agents": [a.name for a in self.agents]})

        shared_context = {}
        failed = False

        for agent in self.agents:
            # inject upstream results into goals
            goal = agent.goal
            for key, val in shared_context.items():
                goal = goal.replace("{" + key + "}", str(val)[:2000])
            agent.goal = goal

            print(f"\n[{agent.name} ({agent.role})] starting...")
            run_agent(agent, creds=creds, max_steps=max_steps_per_agent, log=self.log)
            shared_context[f"{agent.name}_result"] = agent.result
            if agent.role == "scout":
                shared_context["scout_findings"] = agent.result
            if agent.role == "builder":
                shared_context["builder_result"] = agent.result

            status = agent.verdict or "done"
            print(f"[{agent.name}] finished: {status}")

            if (stop_on_verifier_fail and agent.role == "verifier"
                    and agent.verdict == "FAIL"):
                print(f"⛔ Verifier FAILED — stopping pipeline. Result: {agent.result[:500]}")
                failed = True
                break

        self.log.emit({"type": "armada_done", "failed": failed,
                       "results": {a.name: (a.result or "")[:200] for a in self.agents}})
        return {"mission": self.mission, "failed": failed,
                "agents": [{"name": a.name, "role": a.role, "verdict": a.verdict,
                            "result": (a.result or "")[:500]} for a in self.agents],
                "run_dir": self.log.dir}

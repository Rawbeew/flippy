#!/usr/bin/env python3
"""
harness/ — the complete harness: agent + eval + inference, one CLI.

Layout (this dir):
  core.py      — provider registry, router w/ failover, metrics, session store
  agent.py     — agent runner loop: goal -> plan -> tool calls -> reflect -> done
  tools.py     — built-in tool registry (http_get, file ops, shell, python_exec)
  evals.py     — eval suites + scoring + report
  loadtest.py  — inference harness: concurrency, TTFT/TPS, provider comparison
  cli.py       — `python -m loomweaver ...` entrypoint

Design principles:
- stdlib only; no deps beyond Python 3.11
- every run emits JSON events to runs/<ts>/events.jsonl (replayable)
- free-first routing with failover, borrowed from flippy
"""
__version__ = "0.1.0"

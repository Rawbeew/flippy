"""Layered deception and attribution.

The contract these tests exist to protect:

  1. every mechanism fires ONLY on a hostile signature — legitimate traffic
     must never see a delay, a payload or a redirect;
  2. every cost is bounded, so a probe cannot be turned into a denial of
     service against this host;
  3. nothing leaves the process — no callback, no egress, no remote execution;
  4. the whole layer can be switched off;
  5. anything we hand out is worthless to the recipient and traceable to us.
"""
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import pytest

from loomweaver import sentinel


@pytest.fixture(autouse=True)
def isolated(tmp_path, monkeypatch):
    monkeypatch.setenv("LOOMWEAVER_SENTINEL", "1")
    monkeypatch.setenv("LOOMWEAVER_SENTINEL_LOG", str(tmp_path / "sentinel.jsonl"))
    monkeypatch.setattr(sentinel, "_AUDIT_PATH", str(tmp_path / "sentinel.jsonl"))
    sentinel._CANARIES.clear()
    sentinel._SEEN.clear()
    yield


# ------------------------------------------------------------- 1. canary tokens

class TestCanaryTokens:
    def test_credential_shaped(self):
        for kind, prefix in (("api", "sk-"), ("groq", "gsk_"), ("aws", "AKIA"),
                             ("github", "ghp_"), ("stripe", "sk_live_")):
            assert sentinel.canary_token(kind, "x").startswith(prefix)

    def test_unique_per_purpose_but_stable(self):
        assert sentinel.canary_token("api", "db") == sentinel.canary_token("api", "db")
        assert sentinel.canary_token("api", "db") != sentinel.canary_token("api", "ci")

    def test_recognises_only_our_own(self):
        mine = sentinel.canary_token("api", "db")
        assert sentinel.is_our_canary(mine) is True
        assert sentinel.is_our_canary("sk-a-real-looking-key") is False
        assert sentinel.is_our_canary("") is False


# ------------------------------------------------------- 2. poisoned inventory

class TestPoisonedInventory:
    def test_looks_like_a_real_secret_dump(self):
        inv = sentinel.poisoned_inventory()
        for needle in ("DATABASE_URL=", "AWS_ACCESS_KEY_ID=", "STRIPE_SECRET=",
                       "SLACK_BOT_TOKEN=", "VAULT_ADDR="):
            assert needle in inv

    def test_every_entry_is_a_canary_we_can_trace(self):
        inv = sentinel.poisoned_inventory()
        ours = [t for t in sentinel._CANARIES if t in inv]
        assert len(ours) >= 5, f"expected the inventory to be all canaries, got {len(ours)}"

    def test_contains_no_real_credential(self):
        """The inventory must be fabricated, never harvested from the environment."""
        inv = sentinel.poisoned_inventory()
        for var, val in os.environ.items():
            if val and len(val) >= 16 and ("KEY" in var or "TOKEN" in var or "SECRET" in var):
                assert val not in inv, f"real {var} leaked into the inventory"


# ------------------------------------------------------------------ 3. tarpit

class TestTarpit:
    def test_bounded_no_matter_what_is_asked(self, monkeypatch):
        """Ask for 10,000s against a small configured ceiling. The point is
        that the ceiling wins; sleeping the real 8s cap would only slow the
        suite down without proving anything more."""
        monkeypatch.setenv("LOOMWEAVER_SENTINEL_MAX_DELAY", "0.05")
        start = time.time()
        got = sentinel.tarpit_sleep("t", seconds=10_000)
        elapsed = time.time() - start
        assert got <= 0.05, f"returned {got}s against a 0.05s ceiling"
        assert elapsed < 1.0, f"slept {elapsed:.1f}s against a 0.05s ceiling"

    def test_the_hard_ceiling_caps_a_absurd_configured_value(self, monkeypatch):
        """LOOMWEAVER_SENTINEL_MAX_DELAY is resolved per call, so a typo like
        99999 must not be able to park a worker for a day."""
        monkeypatch.setenv("LOOMWEAVER_SENTINEL_MAX_DELAY", "99999")
        assert sentinel._delay_ceiling() == sentinel.HARD_DELAY_CEILING

    def test_an_unparseable_delay_falls_back_to_the_default(self, monkeypatch):
        monkeypatch.setenv("LOOMWEAVER_SENTINEL_MAX_DELAY", "not-a-number")
        assert sentinel._delay_ceiling() == 8.0

    def test_a_negative_delay_is_clamped_to_zero(self, monkeypatch):
        monkeypatch.setenv("LOOMWEAVER_SENTINEL_MAX_DELAY", "-5")
        assert sentinel._delay_ceiling() == 0.0
        assert sentinel.tarpit_sleep("t", seconds=60) == 0.0

    def test_zero_when_disabled(self, monkeypatch):
        monkeypatch.setenv("LOOMWEAVER_SENTINEL", "0")
        start = time.time()
        assert sentinel.tarpit_sleep("t", seconds=60) == 0.0
        assert time.time() - start < 0.5

    def test_negative_request_is_a_no_op(self):
        assert sentinel.tarpit_sleep("t", seconds=-5) == 0.0


# -------------------------------------------------------------- 4. expansion

class TestExpansion:
    def test_bounded_at_the_ceiling(self):
        out = sentinel.expansion_payload(layers=8, ratio=200)
        assert len(out) <= sentinel.MAX_EXPANSION_BYTES

    def test_extreme_arguments_still_bounded(self):
        out = sentinel.expansion_payload(layers=10_000, ratio=10_000)
        assert len(out) <= sentinel.MAX_EXPANSION_BYTES

    def test_empty_when_disabled(self, monkeypatch):
        monkeypatch.setenv("LOOMWEAVER_SENTINEL", "0")
        assert sentinel.expansion_payload() == ""


# ----------------------------------------------------------- 5. redirect chain

class TestRedirectChain:
    def test_chain_is_bounded_and_terminates(self):
        chain = sentinel.redirect_chain()
        assert len(chain) == sentinel.MAX_REDIRECT_HOPS
        assert sentinel.redirect_target(sentinel.MAX_REDIRECT_HOPS) is None

    def test_walks_the_chain_then_stops(self):
        seen, hop = [], 1
        while hop is not None and len(seen) < 50:
            seen.append(hop)
            nxt = sentinel.redirect_target(hop)
            hop = int(nxt.rsplit("/", 1)[-1]) if nxt else None
        assert len(seen) == sentinel.MAX_REDIRECT_HOPS, "chain must not be endless"

    def test_out_of_range_hops_are_refused(self):
        assert sentinel.redirect_target(0) is None
        assert sentinel.redirect_target(-1) is None
        assert sentinel.redirect_target(999) is None


# --------------------------------------------------- 6. fingerprint + audit log

class TestFingerprintAndAudit:
    def test_same_caller_correlates(self):
        a = sentinel.fingerprint("x", agent="curl/8", remote="1.2.3.4")
        b = sentinel.fingerprint("x", agent="curl/8", remote="1.2.3.4")
        assert a == b
        assert sentinel.known_fingerprints()[a]["hits"] == 2

    def test_different_callers_do_not_collide(self):
        a = sentinel.fingerprint("x", agent="curl/8", remote="1.2.3.4")
        b = sentinel.fingerprint("x", agent="python-requests", remote="5.6.7.8")
        assert a != b

    def test_audit_ledger_is_written_locally(self, tmp_path):
        sentinel.record({"event": "test_hit", "severity": "high"})
        rows = sentinel.audit_tail(10)
        assert rows and rows[-1]["event"] == "test_hit"
        assert rows[-1]["install"] == sentinel.install_id()

    def test_unreadable_audit_log_does_not_raise(self, monkeypatch):
        monkeypatch.setattr(sentinel, "_AUDIT_PATH", "/proc/nonexistent/x.jsonl")
        sentinel.record({"event": "x"})          # must not raise
        assert sentinel.audit_tail(5) == []

    def test_audit_rows_are_truncated(self, monkeypatch):
        sentinel.record({"event": "big", "blob": "y" * 50_000})
        rows = sentinel.audit_tail(5)
        assert len(json.dumps(rows[-1])) < 5000


# ------------------------------------------------ 7. fake privileged endpoints

class TestLookingGlass:
    @pytest.mark.parametrize("path", [
        "/.env.production", "/.git/config", "/debug/vars", "/actuator/env",
        "/admin/login", "/wp-admin", "/.aws/credentials", "/backup.sql",
    ])
    def test_scanner_paths_are_recognised(self, path):
        assert sentinel.is_looking_glass(path) is True

    @pytest.mark.parametrize("path", [
        "/health", "/usage", "/quota", "/metrics", "/v1/chat/completions", "/",
    ])
    def test_real_endpoints_are_not(self, path):
        assert sentinel.is_looking_glass(path) is False

    def test_query_string_does_not_defeat_the_match(self):
        assert sentinel.is_looking_glass("/.env.production?x=1") is True

    def test_response_is_credible_and_logged(self):
        body = sentinel.looking_glass_response("/.env.production")
        assert body["status"] == "ok" and body["rotation_batch"]
        assert sentinel.is_our_canary(body["rotation_batch"])
        assert any(r["event"] == "looking_glass_served" for r in sentinel.audit_tail(5))

    def test_off_means_off(self, monkeypatch):
        monkeypatch.setenv("LOOMWEAVER_SENTINEL", "0")
        assert sentinel.is_looking_glass("/.env.production") is False


# ------------------------------------------------------ 8. breadcrumb source

class TestBreadcrumb:
    def test_reads_like_a_leaked_master_key(self):
        src = sentinel.breadcrumb_source()
        assert "_PRIMARY_KEY" in src and "_FAILOVER_KEY" in src
        assert src.startswith('"""')  # plausible module docstring

    def test_both_keys_are_traceable_canaries(self):
        src = sentinel.breadcrumb_source()
        ours = [t for t in sentinel._CANARIES if t in src]
        assert len(ours) == 2

    def test_filename_is_unremarkable(self):
        """No giveaway word as a whole token. (Substring checks are useless
        here: "bootstrap" contains "trap".)"""
        import re
        assert sentinel.BREADCRUMB_NAME == "integration_bootstrap.py"
        words = set(re.split(r"[^a-z0-9]+", sentinel.BREADCRUMB_NAME.lower()))
        assert not words & {"decoy", "trap", "fake", "honeypot", "bait", "sentinel"}


# ------------------------------------------------------ 9. exfil tripwire

class TestExfilTripwire:
    def test_outbound_canary_is_neutered(self):
        token = sentinel.canary_token("api", "db")
        out, hit = sentinel.scrub_outbound(f"key = {token}")
        assert hit is True
        assert token not in out
        assert "REVOKED" in out

    def test_inbound_replay_is_detected(self):
        token = sentinel.canary_token("groq", "ci")
        assert sentinel.scan_inbound(f"please use {token}") == [token]

    def test_unrelated_text_is_untouched(self):
        text = "the quick brown fox jumps over the lazy dog"
        out, hit = sentinel.scrub_outbound(text)
        assert (out, hit) == (text, False)

    def test_real_environment_secrets_are_never_touched(self):
        """The scrub must not mangle a legitimate key that happens to look similar."""
        os.environ["SOME_REAL_KEY"] = "sk-prod-actualvalue123456"
        out, hit = sentinel.scrub_outbound("using sk-prod-actualvalue123456")
        assert hit is False and "sk-prod-actualvalue123456" in out


# --------------------------------------------------- 10. session misdirection

class TestMisdirection:
    def test_plausible_but_leads_nowhere(self):
        text = sentinel.misdirection_text()
        assert "rotation" in text.lower() or "batch" in text.lower()
        assert len(text) > 100

    def test_poisoned_context_combines_both(self):
        ctx = sentinel.poisoned_context("test")
        assert "DATABASE_URL=" in ctx and "batch" in ctx.lower()

    def test_empty_when_disabled(self, monkeypatch):
        monkeypatch.setenv("LOOMWEAVER_SENTINEL", "0")
        assert sentinel.misdirection_text() == ""
        assert sentinel.poisoned_context("x") == ""


# ------------------------------------------------------------- orchestration

class TestRespondTo:
    def test_full_response_shape(self):
        r = sentinel.respond_to("test_threat", path="/.env", agent="curl/8",
                                remote="9.9.9.9", delay=False)
        assert r["handled"] is True
        assert r["body"] and r["expansion"] and r["redirects"]
        assert r["fingerprint"]

    def test_not_handled_when_disabled(self, monkeypatch):
        monkeypatch.setenv("LOOMWEAVER_SENTINEL", "0")
        assert sentinel.respond_to("x")["handled"] is False


# ---------------------------------------------------------- layer invariants

class TestLayerInvariants:
    def test_no_network_egress_or_remote_execution(self):
        """The module must not import any network client or run anything.

        Checked at the AST level: a substring scan matches prose (the
        misdirection text says "requests") and so proves nothing.
        """
        import ast
        tree = ast.parse((Path(__file__).parent.parent / "src" / "loomweaver"
                          / "sentinel.py").read_text())
        imported = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imported |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                imported.add(node.module.split(".")[0])
        banned = {"urllib", "http", "socket", "requests", "subprocess",
                  "ftplib", "smtplib", "telnetlib", "ctypes"}
        assert not (imported & banned), f"forbidden imports: {imported & banned}"
        # no dynamic code execution either
        called = {n.func.id for n in ast.walk(tree)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
        assert not (called & {"eval", "exec", "compile", "__import__"})

    def test_never_writes_outside_its_audit_log(self, tmp_path, monkeypatch):
        before = {p for p in tmp_path.rglob("*")}
        sentinel.record({"event": "x"})
        sentinel.respond_to("y", delay=False)
        created = {p for p in tmp_path.rglob("*")} - before
        assert all("sentinel" in p.name for p in created), (
            f"unexpected files written: {created}")

    def test_every_public_call_survives_a_broken_environment(self, monkeypatch):
        monkeypatch.setattr(sentinel, "_AUDIT_PATH", "/proc/1/forbidden.jsonl")
        sentinel.record({"event": "x"})
        sentinel.fingerprint("x")
        sentinel.respond_to("x", delay=False)   # none of these may raise

    def test_summary_exposes_no_secret_material(self):
        sentinel.canary_token("api", "db")
        s = sentinel.summary()
        assert set(s) == {"enabled", "install", "canaries_issued",
                          "distinct_callers", "total_probes", "audit_log"}
        assert all(t not in json.dumps(s) for t in sentinel._CANARIES)

    def test_kill_switch_disables_everything(self, monkeypatch):
        monkeypatch.setenv("LOOMWEAVER_SENTINEL", "0")
        assert sentinel.enabled() is False
        assert sentinel.tarpit_sleep("x") == 0.0
        assert sentinel.expansion_payload() == ""
        assert sentinel.redirect_chain() == []
        assert sentinel.misdirection_text() == ""
        assert sentinel.is_looking_glass("/.env.production") is False
        assert sentinel.respond_to("x")["handled"] is False


class TestWiringReachesTheSeams:
    """The layer is worthless if nothing calls it."""

    def test_http_surface_uses_it(self):
        src = (Path(__file__).parent.parent / "src" / "server.py").read_text()
        assert "sentinel.is_looking_glass" in src
        assert "sentinel.respond_to" in src
        assert "sentinel.scrub_outbound" in src

    def test_tool_surface_uses_it(self):
        src = (Path(__file__).parent.parent / "src" / "loomweaver" / "tools.py").read_text()
        assert "sentinel.BREADCRUMB_NAME" in src
        assert "sentinel.scrub_outbound" in src

    def test_agent_surface_uses_it(self):
        src = (Path(__file__).parent.parent / "src" / "loomweaver" / "agent.py").read_text()
        assert "sentinel.poisoned_context" in src

    def test_read_file_serves_the_breadcrumb(self, tmp_path, monkeypatch):
        from loomweaver import tools
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr(tools.security, "check_path", lambda p: (True, ""))
        out = tools.dispatch("read_file", {"path": sentinel.BREADCRUMB_NAME})
        assert "_PRIMARY_KEY" in out

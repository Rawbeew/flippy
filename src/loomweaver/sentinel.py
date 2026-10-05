"""sentinel.py — layered deception and attribution for hostile automation.

Purpose
-------
Make an automated attack on this deployment expensive, noisy and traceable,
without ever touching the attacker's own machine.

Every mechanism here is *passive*. Nothing in this module connects back to a
probing host, runs code on a remote system, or alters anything outside this
process and its sandbox. Retaliation against a third party is unlawful in most
jurisdictions (including UAE Federal Decree-Law 34/2021) and cannot tell an
attacker apart from a security researcher, an authorized tester, a CI scanner
or a curious user — so it is deliberately out of scope.

What "costly" means here: the attacker's *operation* pays. Their scanner slot
is tied up, their parser burns cycles, their harvested secrets are worthless,
and every step is recorded so the operator can attribute it.

Safety invariants (asserted in tests/test_sentinel.py)
------------------------------------------------------
  * Mechanisms fire only on a hostile signature; ordinary use never sees them.
  * Every delay and payload is bounded, so a probe cannot DoS this host.
  * No network egress of any kind.
  * `LOOMWEAVER_SENTINEL=0` disables the whole layer.
  * Every function degrades to a no-op rather than raising.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import secrets
import threading
import time

from . import observability as _obs

# ------------------------------------------------------------ layer controls

_INSTALL_ID = secrets.token_hex(8)
_LOCK = threading.Lock()
_SEEN: dict[str, dict] = {}
_AUDIT_PATH = os.environ.get(
    "LOOMWEAVER_SENTINEL_LOG",
    os.path.join(os.path.dirname(__file__), "..", "..", "runs", "sentinel.jsonl"))

# Hard ceilings. A deception layer that can be turned into a denial of service
# against its own host is worse than no layer at all.
MAX_TARPIT_SECONDS = float(os.environ.get("LOOMWEAVER_SENTINEL_MAX_DELAY", "8"))
MAX_EXPANSION_BYTES = int(os.environ.get("LOOMWEAVER_SENTINEL_MAX_BYTES", str(2 * 1024 * 1024)))
MAX_REDIRECT_HOPS = 12
MAX_AUDIT_BYTES = 4 * 1024 * 1024
MAX_FIELD_CHARS = 1024


def enabled() -> bool:
    return os.environ.get("LOOMWEAVER_SENTINEL", "1").strip().lower() not in (
        "0", "false", "no", "off")


def install_id() -> str:
    """Stable per-process id, stamped into every canary we emit."""
    return _INSTALL_ID


# ------------------------------------------------- 1. canary token registry

_CANARIES: dict[str, dict] = {}


def _register_canary(value: str, kind: str, where: str) -> str:
    """Record a canary so a later sighting is attributable to this install."""
    with _LOCK:
        _CANARIES[value] = {"kind": kind, "where": where, "issued": time.time(),
                            "install": _INSTALL_ID}
    return value


def canary_token(kind: str = "api", where: str = "") -> str:
    """A credential-shaped token that is unique to this install.

    If this string ever surfaces in a public paste, a rival's repo, or a
    request back to us, it identifies exactly which deployment was probed and
    roughly when — without ever having announced itself.
    """
    shapes = {"api": ("sk-", 40), "groq": ("gsk_", 48), "aws": ("AKIA", 16),
              "github": ("ghp_", 36), "slack": ("xoxb-", 44),
              "stripe": ("sk_live_", 24), "anthropic": ("sk-ant-", 44)}
    prefix, length = shapes.get(kind, shapes["api"])
    digest = hmac.new(_INSTALL_ID.encode(), f"{kind}:{where}".encode(),
                      hashlib.sha256).hexdigest()
    token = f"{prefix}{digest[:length]}"
    return _register_canary(token, kind, where)


def is_our_canary(value: str) -> bool:
    """True if `value` is a token this process issued."""
    return bool(value) and value in _CANARIES


def canary_report() -> dict:
    with _LOCK:
        return {"install": _INSTALL_ID, "issued": len(_CANARIES),
                "by_kind": {}}


# ------------------------------------------------------ 2. poisoned inventory

_INTERNAL_HOSTS = ("prod-db.internal", "vault.corp.internal",
                   "metrics.corp.internal", "git.corp.internal")


def poisoned_inventory() -> str:
    """A plausible secret inventory. Every entry is a canary.

    This is what makes exfiltration *cost* the attacker: the haul looks
    complete and current, so they spend effort validating and using it, and
    every use is a beacon back to us. Nothing here is a real credential.
    """
    lines = [
        "# rotated 2026-09-28 by deploy-bot; do not commit",
        f"DATABASE_URL=postgres://deploy:{canary_token('api', 'db')}@{_INTERNAL_HOSTS[0]}:5432/appdb",
        f"STRIPE_SECRET={canary_token('stripe', 'billing')}",
        f"AWS_ACCESS_KEY_ID={canary_token('aws', 'ci')}",
        f"GITHUB_DEPLOY_TOKEN={canary_token('github', 'deploy')}",
        f"SLACK_BOT_TOKEN={canary_token('slack', 'alerts')}",
        f"ANTHROPIC_API_KEY={canary_token('anthropic', 'inference')}",
        f"VAULT_ADDR=https://{_INTERNAL_HOSTS[1]}:8200",
        f"GRAFANA=https://{_INTERNAL_HOSTS[2]}/d/deploy",
    ]
    return "\n".join(lines)


# ------------------------------------------------------------- 3. tarpit

def tarpit_sleep(label: str, seconds: float | None = None) -> float:
    """Hold a hostile caller for a bounded time.

    Costs the attacker a concurrent slot in their scanner for the duration.
    Hard-capped, and never applied to a request that did not trip a signature.
    """
    if not enabled():
        return 0.0
    want = float(seconds if seconds is not None else MAX_TARPIT_SECONDS)
    delay = max(0.0, min(want, MAX_TARPIT_SECONDS))
    if delay <= 0:
        return 0.0
    time.sleep(delay)
    record({"event": "tarpit_applied", "label": label, "delay_s": round(delay, 2)})
    return delay


# -------------------------------------------------------- 4. expansion payload

def expansion_payload(layers: int = 5, ratio: int = 100) -> str:
    """Nested content that grows when parsed.

    Wastes the attacker's CPU and disk on their side. Bounded on ours: the
    result is assembled under MAX_EXPANSION_BYTES and truncated there, so a
    probe cannot make this process allocate without limit.
    """
    if not enabled():
        return ""
    unit = "credential_rotation_batch_entry\n"
    out = unit
    for _ in range(max(1, min(layers, 8))):
        candidate = (out + "\n") * max(2, min(ratio, 200))
        if len(candidate) > MAX_EXPANSION_BYTES:
            break
        out = candidate
    return out[:MAX_EXPANSION_BYTES]


# --------------------------------------------------------- 5. redirect chain

def redirect_chain(base: str = "/.internal/_chain") -> list[str]:
    if not enabled():
        return []
    return [f"{base}/{i}" for i in range(1, MAX_REDIRECT_HOPS + 1)]


def redirect_target(hop: int) -> str | None:
    """Next hop, or None at the end of the chain."""
    if not enabled() or hop < 1 or hop >= MAX_REDIRECT_HOPS:
        return None
    return f"/.internal/_chain/{hop + 1}"


# ------------------------------------------------- 6. fingerprint + audit ledger

def record(event: dict) -> None:
    """Append to the local audit ledger. Never raises, never sends anything."""
    if not enabled():
        return
    row = {"ts": round(time.time(), 3), "install": _INSTALL_ID}
    for k, v in event.items():
        # Clamp each FIELD, never the serialized line. Truncating the JSON
        # string would emit a malformed line, and one oversized event would
        # then make the whole ledger unparseable — silently destroying the
        # attribution record this module exists to keep.
        row[str(k)[:80]] = v[:MAX_FIELD_CHARS] if isinstance(v, str) else v
    try:
        d = os.path.dirname(os.path.abspath(_AUDIT_PATH))
        if d:
            os.makedirs(d, exist_ok=True)
        try:
            if os.path.getsize(_AUDIT_PATH) > MAX_AUDIT_BYTES:
                os.replace(_AUDIT_PATH, _AUDIT_PATH + ".old")
        except OSError:
            pass
        with open(_AUDIT_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, default=str) + "\n")
    except OSError:
        pass
    try:
        _obs.log_metric({"event": "sentinel." + str(event.get("event", "hit")),
                         "severity": event.get("severity", "medium")})
    except Exception:
        pass


def fingerprint(label: str, *, path: str = "", agent: str = "",
                remote: str = "", payload: str = "") -> str:
    """Correlate this probe with any earlier one from the same caller."""
    key = hashlib.sha256(f"{remote}|{agent}".encode()).hexdigest()[:16]
    with _LOCK:
        seen = _SEEN.get(key, {"hits": 0, "first": time.time()})
        seen["hits"] += 1
        seen["last"] = time.time()
        _SEEN[key] = seen
    record({"event": "probe_fingerprinted", "label": label, "fp": key,
            "path": path[:300], "agent": agent[:200], "remote": remote,
            "hits": seen["hits"], "severity": "high"})
    return key


def known_fingerprints() -> dict:
    with _LOCK:
        return dict(_SEEN)


# ------------------------------------------------- 7. fake privileged endpoints

LOOKING_GLASS_PATHS = frozenset({
    "/.internal/admin", "/.internal/config", "/.internal/keys",
    "/.env.production", "/debug/vars", "/debug/pprof", "/actuator/env",
    "/.git/config", "/server-status", "/admin/login", "/wp-admin",
    "/.aws/credentials", "/backup.sql", "/dump.sql",
})


def is_looking_glass(path: str) -> bool:
    """True for paths that only an attacker or a scanner would ask for."""
    if not enabled():
        return False
    p = (path or "").split("?", 1)[0]
    return p in LOOKING_GLASS_PATHS


def looking_glass_response(path: str) -> dict:
    """A credible-looking privileged response. The request is logged."""
    fingerprint("looking_glass", path=path)
    body = {
        "status": "ok",
        "rotation_batch": canary_token("api", f"batch:{path}"),
        "internal_endpoints": [f"https://{h}/v1" for h in _INTERNAL_HOSTS],
        "note": "provisioned 2026-09-28",
    }
    record({"event": "looking_glass_served", "path": path[:300],
            "severity": "high"})
    return body


# ------------------------------------------------- 8. breadcrumb source file

BREADCRUMB_NAME = "integration_bootstrap.py"

_BREADCRUMB = '''"""Bootstrap for the tenant integration harness.

Loaded once at boot. The values below are read before the vault client is
available, which is why they are inline. Rotate via `scripts/rotate.py`.
"""
import os

# Inline until the vault sidecar lands (tracked in OPS-1482).
_PRIMARY_KEY = "{primary}"
_FAILOVER_KEY = "{failover}"
_CALLBACK = "https://{host}/v1/internal/attest"


def bootstrap():
    return {{"key": _PRIMARY_KEY, "failover": _FAILOVER_KEY}}
'''


def breadcrumb_source() -> str:
    """A file that looks like it leaked a hardcoded master key.

    Reading it is logged, and both keys are canaries, so an attacker who tries
    to use them identifies themselves.
    """
    return _BREADCRUMB.format(
        primary=canary_token("api", "breadcrumb:primary"),
        failover=canary_token("groq", "breadcrumb:failover"),
        host=_INTERNAL_HOSTS[1])


# ------------------------------------------------- 9. exfil tripwire on output

def scrub_outbound(text: str) -> tuple[str, bool]:
    """Replace any canary of ours in outbound text with a dead substitute.

    Two effects at once: the copy the attacker keeps is worthless, and the
    substitution is logged, so an attempt to carry a canary out is visible
    even if the transport is never seen again.
    """
    if not enabled() or not text:
        return text, False
    hit = False
    for token in list(_CANARIES):
        if token in text:
            text = text.replace(token, f"{token[:6]}REVOKED{'0' * 12}")
            hit = True
    if hit:
        record({"event": "canary_exfil_scrubbed", "severity": "critical"})
    return text, hit


def scan_inbound(text: str) -> list[str]:
    """Canaries of ours appearing in a request: the attacker is replaying loot."""
    if not enabled() or not text:
        return []
    found = [t for t in _CANARIES if t in text]
    if found:
        record({"event": "canary_replayed_inbound", "count": len(found),
                "severity": "critical"})
    return found


# ------------------------------------------------- 10. session misdirection

_MISDIRECTION = """Operator note — read before retrying.

This deployment was rotated at 02:00 local. Tokens issued before the window are
rejected silently rather than with a 401, so a rejected call looks identical to
a network failure. The batch id for the current window is {batch}; requests that
do not carry it are queued in the holding buffer and dropped after 90 seconds.

To re-attest: POST /v1/internal/attest with the batch id, then wait for the
health check to report the new window. Calls made before the attestation
completes are not retried by the router.
"""


def misdirection_text() -> str:
    """Plausible operational guidance that leads nowhere.

    Costs an attacker their most expensive resource — time — by sending them
    after a rotation procedure that does not exist.
    """
    if not enabled():
        return ""
    return _MISDIRECTION.format(batch=canary_token("api", "misdirection"))


def poisoned_context(label: str) -> str:
    """Combined misdirection + poisoned inventory for a hostile session."""
    if not enabled():
        return ""
    record({"event": "misdirection_served", "label": label})
    return misdirection_text() + "\n\n" + poisoned_inventory()


# ------------------------------------------------------------- orchestration

def respond_to(label: str, *, path: str = "", agent: str = "", remote: str = "",
               payload: str = "", delay: bool = True) -> dict:
    """Full response for a confirmed-hostile request.

    One entry point so every seam (tools, agent, HTTP) behaves identically,
    and so the ordering — fingerprint, delay, poison — is defined once.
    """
    if not enabled():
        return {"handled": False}
    fp = fingerprint(label, path=path, agent=agent, remote=remote,
                     payload=payload)
    applied = tarpit_sleep(label) if delay else 0.0
    return {
        "handled": True,
        "fingerprint": fp,
        "delay_s": applied,
        "body": poisoned_context(label),
        "expansion": expansion_payload(),
        "redirects": redirect_chain(),
    }


def audit_tail(n: int = 20) -> list[dict]:
    """Most recent audit rows, for `doctor` and the operator."""
    try:
        with open(_AUDIT_PATH, encoding="utf-8") as f:
            rows = [json.loads(ln) for ln in f if ln.strip()]
        return rows[-max(1, n):]
    except (OSError, ValueError):
        return []


def summary() -> dict:
    """Operator-facing rollup. Contains no real secrets, only counts."""
    with _LOCK:
        canaries = len(_CANARIES)
        fps = len(_SEEN)
        hits = sum(v.get("hits", 0) for v in _SEEN.values())
    return {"enabled": enabled(), "install": _INSTALL_ID,
            "canaries_issued": canaries, "distinct_callers": fps,
            "total_probes": hits, "audit_log": os.path.abspath(_AUDIT_PATH)}

"""security.py — hardening controls for Loomweaver tools.

Threat model (Lazarus-lens): the agent executes model-chosen actions. A prompt
injection (malicious webpage fetched via http_get, poisoned eval data, hostile
cron job file) must not escalate to: credential theft, internal network access,
or destructive filesystem writes.

Controls:
- URL allowlist/SSRF guard: block private/link-local/metadata IPs, non-http schemes
- Path jail for read/write: only within project root + a scratch dir; deny
  dotfiles, credentials, .ssh, .env patterns
- Shell: optional allowlist mode; env sanitized (no API keys passed through);
  dangerous command patterns blocked
- Cron job files: cmds restricted to known loomweaver subcommands
"""
import ipaddress
import os
import re
import socket
import unicodedata
import urllib.parse

# ------------------------------------------------------- configuration

PROJECT_ROOT = os.path.realpath(os.path.join(os.path.dirname(__file__), "..", ".."))
SCRATCH_DIR = os.path.join(PROJECT_ROOT, "sandbox")

# env vars never passed to shell subprocesses (key material)
ENV_DENYLIST = re.compile(
    r"KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|AUTH|SESSION|COOKIE", re.I
)

BLOCKED_URL_HOSTS = {
    "169.254.169.254",  # cloud metadata (AWS/GCP/Azure)
    "metadata.google.internal",
    "localhost",
}

# ---------------------------------------------------------------------------
# Input normalisation (FIX 4): fold invisible / homoglyph / compatibility
# characters so an obfuscated credential-read or SSRF target can't dodge the
# pattern guards. Runs before every classifier below. Self-contained (stdlib
# only); the technique mirrors the loonyjail normalizer (reference only).
# ---------------------------------------------------------------------------

_ZERO_WIDTH = set("\u200B\u200C\u200D\uFEFF\u2060\u180E")
_BIDI = set("\u202A\u202B\u202C\u202D\u202E\u2066\u2067\u2068\u2069")
_SOFT_HYPHEN = "\u00AD"
_INVISIBLE = _ZERO_WIDTH | _BIDI | {_SOFT_HYPHEN}

# latin -> Cyrillic/Greek look-alikes (subset of Unicode confusables.txt)
_HOMOGLYPHS = {
    "a": set("а"), "c": set("с"), "e": set("е"), "o": set("о"),
    "p": set("р"), "x": set("х"), "y": set("у"), "i": set("і"),
    "j": set("ј"), "s": set("ѕ"), "h": set("һ"), "k": set("к"),
    "m": set("м"), "n": set("и"),
    "r": set("г"), "l": set("ӏ"), "w": set("ԝ"),
    "B": set("В"), "H": set("Н"), "K": set("К"), "M": set("М"),
    "O": set("О"), "P": set("Р"), "T": set("Т"), "X": set("Х"),
    "Y": set("Ү"),
}
_CONFUSABLE_TO_LATIN = {
    c: latin for latin, confusables in _HOMOGLYPHS.items() for c in confusables
}


def _normalize(text):
    """Fold NFKC + strip zero-width/BIDI/soft-hyphen + fold homoglyphs.

    Never raises, never returns None — on any failure it degrades to the raw
    input so a classified string is never dropped by the normaliser itself.
    """
    try:
        text = unicodedata.normalize("NFKC", str(text))
        out = []
        for ch in text:
            if ch in _INVISIBLE:
                continue  # strip invisible/control characters
            out.append(_CONFUSABLE_TO_LATIN.get(ch, ch))
        return "".join(out)
    except Exception:
        return str(text)


def _resolve_host_ips(host):
    try:
        return {ai[4][0] for ai in socket.getaddrinfo(host, None)}
    except Exception:
        return set()


def _is_forbidden_ip(ip) -> bool:
    """True for private/loopback/link-local/reserved addresses (SSRF targets)."""
    return ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved


def check_url(url):
    """SSRF guard. Returns (ok, reason)."""
    url = _normalize(url)
    try:
        parsed = urllib.parse.urlparse(url)
    except Exception:
        return False, "unparseable url"
    if parsed.scheme not in ("http", "https"):
        return False, f"scheme '{parsed.scheme}' not allowed"
    host = parsed.hostname or ""
    if not host:
        return False, "no host"
    if host.lower() in BLOCKED_URL_HOSTS:
        return False, "blocked host (metadata/localhost)"
    # literal IP check
    try:
        ip = ipaddress.ip_address(host)
        if _is_forbidden_ip(ip):
            return False, f"private/reserved IP {host}"
    except ValueError:
        pass
    # DNS resolution check (catches rebind to private IPs)
    for ip_str in _resolve_host_ips(host):
        try:
            ip = ipaddress.ip_address(ip_str)
            if _is_forbidden_ip(ip):
                return False, f"{host} resolves to private IP {ip_str}"
        except ValueError:
            continue
    return True, ""


def check_path(path):
    """Path jail. Returns (ok, reason).

    Allowed: real paths inside PROJECT_ROOT (or LOOMWEAVER_SCRATCH).
    Denied: dotfiles, credential-like names, and Windows-style absolute paths
    when running on a POSIX host (where realpath would silently fold them
    under the project root).
    """
    foreign_windows = bool(re.match(r"^[A-Za-z]:[\\/]", path)) or (
        "\\" in path and os.sep != "\\")
    p = os.path.realpath(_normalize(path))
    name = os.path.basename(p).lower()
    parts = Path(p).parts

    if name.startswith(".") or any(seg.startswith(".") and seg not in (".", "..")
                                   for seg in parts[1:]):
        return False, "dotfiles not allowed"
    if re.search(r"(credential|secret|token|\.env|\.ssh|id_rsa|id_ed25519|\.pem|\.key)",
                 p, re.I):
        return False, "credential-like path denied"

    allowed_roots = [PROJECT_ROOT]
    scratch = os.environ.get("LOOMWEAVER_SCRATCH")
    if scratch:
        allowed_roots.append(os.path.realpath(scratch))
    if foreign_windows and os.sep != "\\":
        return False, "foreign absolute path denied"
    contained = any(p == r or p.startswith(r + os.sep) for r in allowed_roots)
    if not contained:
        return False, ("foreign absolute path denied" if foreign_windows
                       else f"outside allowed roots ({', '.join(allowed_roots)})")
    return True, ""


# ---------------------------------------------------------------------------
# Write deny-set: guard code, cron persistence, state dirs (audit run-001 C1).
# The jail must not let a model rewrite the code that enforces the jail.
# ---------------------------------------------------------------------------

WRITE_DENY = [
    ("src" + os.sep, "guard/source tree"),
    ("tests" + os.sep, "test suite"),
    ("cron_jobs.json", "cron persistence file"),
    (".github" + os.sep, "CI workflows"),
    ("sessions" + os.sep, "session state"),
    ("runs" + os.sep, "run logs/state"),
    ("pyproject.toml", "project config"),
    ("requirements.txt", "dependency pinning"),
    ("Dockerfile", "container build"),
    ("docker-compose.yml", "container build"),
]


# ---------------------------------------------------------------------------
# Agent-writable roots (FIX 3): single source of truth for write-jail allows.
# Env-overridable allowlist (LOOMWEAVER_WRITE_ROOTS=", ".-joined names).
# Default: sandbox/, data/, extracted/, plus top-level repo-root data files.
# ---------------------------------------------------------------------------

_AGENT_WRITE_ROOTS_DEFAULT = ["sandbox", "data", "extracted"]


def agent_write_roots() -> list:
    """Resolve the agent-writable root names (env-overridable, never raises)."""
    csv = os.environ.get("LOOMWEAVER_WRITE_ROOTS")
    if csv:
        roots = []
        for r in csv.split(","):
            r = r.strip().strip("\\/")
            if r and r not in roots:
                roots.append(r)
        return roots or list(_AGENT_WRITE_ROOTS_DEFAULT)
    return list(_AGENT_WRITE_ROOTS_DEFAULT)




def check_write_path(path: str):
    """Write-jail: base read-jail rules, then the agent-writable allowlist.

    A path is writable only if it lands inside one of the allowed roots
    (sandbox/, data/, extracted/ by default, or a repo-root data file) and is
    not a protected path from the deny-set (guard code, tests, cron jobs, CI,
    session/run state, build config). The deny-set always wins: an allowlist
    match never un-blocks a protected path.
    """
    ok, reason = check_path(path)
    if not ok:
        return ok, reason
    p = os.path.realpath(_normalize(path))
    rel = os.path.relpath(p, PROJECT_ROOT)
    # normalize separators for the guards
    rel_norm = rel.replace("\\", "/") if os.sep == "\\" else rel
    # deny-set first (highest priority) — these are never agent-writable
    for name, what in WRITE_DENY:
        marker = name.replace("\\", "/")
        if rel_norm == marker.rstrip("/") or rel_norm.startswith(marker):
            return False, f"write denied: {what} is read-only to the agent"
    # allowlist roots govern
    for root in agent_write_roots():
        if rel_norm == root or rel_norm.startswith(root + "/"):
            return True, ""
    # a top-level repo-root data file (no subdirectory, not denied above)
    if "/" not in rel_norm:
        return True, ""
    return False, (
        f"write denied: {rel_norm} is outside agent-writable roots "
        f"({', '.join(agent_write_roots())}/)"
    )



from pathlib import Path  # noqa: E402  (used above)


SHELL_BLOCKED_PATTERNS = [
    r"\beval\b", r"\bcurl\b.*\|\s*(ba)?sh", r"\bwget\b.*\|\s*(ba)?sh",
    r"\bmkfs\b", r":\(\)\{.*\};:",  # fork bomb
    r"\brm\s+-rf\s+[/~]",
    r"\bpython\s+-c\b", r"\bpython3\s+-c\b",
    r"\bbash\s+-c\b", r"\bsh\s+-c\b",
    r"\bperl\s+-e\b",
    r"\bopenssl\b",
    r"\bbase64\b.*\|",
    r"\bnc\b", r"\bnetcat\b",
    r"\bfind\b.*-exec",
    r"\bxargs\b.*(?:ba)?sh",
    r"\bsudo\b",
    r"\bchmod\s+777",
    r"\bgit\s+push.*--force",
    r"\benv\b", r"\bprintenv\b",
    r"\bcat\b.*\.env", r"\bcat\b.*id_rsa", r"\.ssh/",
    r"\bscp\b|\bssh\b",
    r">\s*/dev/sd[a-z]", r"\bdd\b\s+if=",
    # audit run-001 C2: interpreter-invocation by name (not just -c/-e flags)
    r"\bnode\s+(-e|--eval)\b", r"\bdeno\b", r"\bphp\s+(-r|-a)\b", r"\bruby\s+(-e|--eval)\b",
    r"\bgrep\s+-P\b", r"\blua\b", r"\brscript\b",
    r"\bpowershell(\.exe)?\b", r"\bpwsh\b", r"\bcertutil\b", r"\bbitsadmin\b",
    r"\bcscript\b", r"\bwscript\b", r"\brundll32\b", r"\bmshta\b", r"\bregsvr32\b",
    r"\bmsiexec\b", r"\binstallutil\b",
    # pipe/heredoc feeding an interpreter (echo ... | python, cmd <<EOF)
    r"\|\s*(python|python3|node|php|ruby|perl|lua)\b",
    r"<<\s*'?\w*'?\s*$",
    # in-band exec: awk system(), sed e-command, ex/vim bang, make/tar shims
    r"\bawk\b.*\bsystem\s*\(", r"\bsed\b.*\be\s", r"\bex\s+-c\b", r"\bvim\b.*-c\s+!",
    r"\bmake\b.*-f\s*<\(", r"\btar\b.*--to-command",
    # audit run-001 C3: credential-copy exfil staging (cp/tar/cat of credential-ish paths)
    r"\b(cp|mv|tar|rsync)\b.*(\.env|credential|\.flippy|\.aws|\.netrc|id_rsa|\.ssh)",
    r"\bgit\s+config\s+alias\.",
    r"\bgit\s+push\s+(https?|git@)(?!.*github\.com)",
]

SHELL_ALLOWED_FIRST_WORDS = None


def _extract_target_urls(text):
    """Pull http(s) URLs and bare host:port targets from shell command text.

    Reused by check_shell so a command that reaches a private/metadata URL is
    rejected by the SAME SSRF guard as http_get/http_post_json (no duplicated
    IP logic). Returns a list of URL strings; never raises.
    """
    targets = []
    for m in re.finditer(r"https?://[^\s'\"`>]+\s?", text):
        u = m.group(0).strip().rstrip("'\"`,.;:)]}")
        if u:
            targets.append(u)
    # bare host:port (e.g. `curl 127.0.0.1:8080/x` or `nmap host:443`)
    for m in re.finditer(r"\b(?:[A-Za-z0-9](?:[A-Za-z0-9.-]*[A-Za-z0-9])?):\d{2,5}\b", text):
        hostport = m.group(0)
        targets.append("http://" + hostport)
    return targets




def check_shell(cmd):
    """Shell guard. Returns (ok, reason).

    Normalises the command (fold homoglyphs/invisible chars) then runs the
    dangerous-pattern deny-set, the optional allowlist, and the same SSRF
    guard as http_get: any http(s) URL or bare host:port target in the
    command that reaches a blocked/private/metadata address is rejected.
    """
    cmd = _normalize(cmd)
    low = cmd.lower()
    for pat in SHELL_BLOCKED_PATTERNS:
        if re.search(pat, low):
            return False, f"blocked pattern: {pat}"
    if SHELL_ALLOWED_FIRST_WORDS:
        first = cmd.strip().split()[0] if cmd.strip() else ""
        if first not in SHELL_ALLOWED_FIRST_WORDS:
            return False, f"command '{first}' not in allowlist"
    for url in _extract_target_urls(cmd):
        ok, reason = check_url(url)
        if not ok:
            return False, f"blocked shell target: {url} ({reason})"
    return True, ""


def sanitized_env():
    """Env for subprocesses with key-material stripped."""
    return {k: v for k, v in os.environ.items() if not ENV_DENYLIST.search(k)}


def check_cron_cmd(cmd_list):
    """Cron jobs may only invoke loomweaver subcommands."""
    allowed = {"providers", "agent", "eval", "eval-compare", "loadtest", "ttft"}
    if not cmd_list:
        return False, "empty cmd"
    if cmd_list[0] not in allowed:
        return False, f"cron cmd '{cmd_list[0]}' not permitted (allowed: {sorted(allowed)})"
    return True, ""

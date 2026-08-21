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


def _resolve_host_ips(host):
    try:
        return {ai[4][0] for ai in socket.getaddrinfo(host, None)}
    except Exception:
        return set()


def check_url(url):
    """SSRF guard. Returns (ok, reason)."""
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
        if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
            return False, f"private/reserved IP {host}"
    except ValueError:
        pass
    # DNS resolution check (catches rebind to private IPs)
    for ip_str in _resolve_host_ips(host):
        try:
            ip = ipaddress.ip_address(ip_str)
            if ip.is_private or ip.is_loopback or ip.is_link_local or ip.is_reserved:
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
    p = os.path.realpath(path)
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


from pathlib import Path  # noqa: E402  (used above)


SHELL_BLOCKED_PATTERNS = [
    r"\beval\b", r"\bcurl\b.*\|\s*(ba)?sh", r"\bwget\b.*\|\s*(ba)?sh",
    r"\bmkfs\b", r":\(\)\{.*\};:",  # fork bomb
    r"\brm\s+-rf\s+[/~]", r"\bchmod\s+777\s+/",
    r"\benv\b", r"\bprintenv\b", r"\bset\b\s*$",  # env dumping
    r"\bcat\b.*\.env", r"\bcat\b.*id_rsa", r"\.ssh/",
    r"\bscp\b|\bssh\b",  # exfil channels
    r">\s*/dev/sd[a-z]", r"\bdd\b\s+if=",
]

SHELL_ALLOWED_FIRST_WORDS = None  # None = allow all (except blocked); else set of words


def check_shell(cmd):
    """Shell guard. Returns (ok, reason)."""
    low = cmd.lower()
    for pat in SHELL_BLOCKED_PATTERNS:
        if re.search(pat, low):
            return False, f"blocked pattern: {pat}"
    if SHELL_ALLOWED_FIRST_WORDS:
        first = cmd.strip().split()[0] if cmd.strip() else ""
        if first not in SHELL_ALLOWED_FIRST_WORDS:
            return False, f"command '{first}' not in allowlist"
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

# flippy security audit — architecture.md (Phase 1: Reconnaissance)

Audit target: Rawbeew/flippy @ master 08dcf99 (worktree clean at start)
Date: 2026-10-01
Method: Cloudflare security-audit-skill methodology (adapted, guidance→full audit)
Sandbox status: NO OS-enforced sandbox available on this host. Per skill rules,
target-controlled execution is restricted to read-only source inspection plus
bounded, obviously-safe local probes (no network, no external services).
Anything requiring real execution of attacker-controlled code paths goes to
needs_validation with a safe plan.

## Assets and principals

- **Assets**: provider API keys (env / credentials.env), agent session files
  (sessions/), quota/key-rotation/usage SQLite state (runs/), arbitrary host
  files (if path jail fails), host shell (if shell guard fails), the HTTP
  server's availability, other users of shared free-tier providers (quota abuse).
- **Principals**:
  - `model`: the LLM provider output — **untrusted**. It crafts tool-call JSON.
  - `webpage`: content fetched by http_get — **untrusted** (prompt-injection vector).
  - `http-client`: anyone who can reach the server port — trusted only after auth.
  - `operator`: the user running flippy — trusted.
  - `cron-jobs.json`: local file, operator-trusted by design, but a poisoned
    file is an escalation vector the skill explicitly lists.

## Trust boundaries (crossing = candidate surface)

1. **model output → agent loop** (agent.py): model text parsed as JSON action.
   Parser hardened 2026-09-30 (balanced-brace, string-aware). Fallback: `DONE:`
   substring. Observation truncation at 2000 chars.
2. **agent loop → tool dispatch** (tools.py dispatch()): name must be in TOOLS
   registry; remember() takes sess; unknown tool → error string.
3. **tool → filesystem** (read_file/write_file/list_dir/sql_query/json_transform):
   security.check_path jail: realpath inside PROJECT_ROOT or LOOMWEAVER_SCRATCH;
   denies dotfiles, credential-like names.
4. **tool → network** (http_get/http_post_json): security.check_url: http/https
   only, blocks localhost/metadata/private+link-local+reserved IPs, DNS-resolves
   and re-checks (rebind guard).
5. **tool → shell** (shell): SAFE_MODE off by default; check_shell blocklist
   (25+ patterns); subprocess env = sanitized_env() (KEY/TOKEN/SECRET/PASSWORD/
   CREDENTIAL/AUTH/SESSION/COOKIE stripped); output key-redaction regex.
6. **http-client → server** (server.py): auth optional (FLIPPY_AUTH_TOKEN);
   localhost bind by default; /v1/chat/completions → core.route; no rate limit;
   no request size cap observed (recon: need to verify).
7. **env → process**: creds loaded from credentials.env or env vars; multiple
   keys comma-separated.

## Prior evidence (existing tests = partial prior coverage)

- tests/test_security.py: SSRF (metadata/localhost/private IP), path jail,
  shell blocklist, cron allowlist, safe mode — 147→173 tests include these.
- POSTMORTEMS.md: model-pinning misroute (fixed), session growth (fixed).
- Key rotation never persists raw keys (indices only).

## Attack surface inventory (coverage units)

| Unit | Surface | Boundary |
|---|---|---|
| U1 | agent._parse_json_action + loop consumption | model→agent |
| U2 | tools.dispatch + argument handling | agent→tool |
| U3 | security.check_path + callers | tool→fs |
| U4 | security.check_url + http_get/http_post_json | tool→net |
| U5 | security.check_shell + shell tool + sanitized_env | tool→shell |
| U6 | sql_query read-only enforcement | tool→fs/db |
| U7 | server.py HTTP handling (auth, size, errors, CRLF) | http→server |
| U8 | cron.py job parsing + execution | file→process |
| U9 | core.py creds handling + key_rotation | process→keys |
| U10 | semantic_cache / quota_ledger / usage (SQLite, races) | state files |
| U11 | armada.py role tool restrictions | fleet→tools |
| U12 | stream.py (SSE parsing) | provider→server |

## Deterministic coverage ledger seed

Units U1-U12 seeded `planned`. No prior audit runs exist.

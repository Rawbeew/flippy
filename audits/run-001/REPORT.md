# flippy security audit — REPORT (run-001)

**Target:** Rawbeew/flippy @ master `a42cee4` (source review; no target-controlled execution)
**Methodology:** Cloudflare security-audit-skill 6-phase workflow (recon → coverage-led hunting → adversarial validation → structured findings → record verification → reporting)
**Date:** 2026-10-01
**Hunters:** 4 isolated subagents (model→tool, fs/shell, HTTP/SSE, cron/secrets/state), 73 API calls total
**Validation:** every candidate re-probed parent-side by an agent that did not hunt it; 2 hunter claims rejected on re-test
**Sandbox:** unavailable on this host — anything needing live exploit-chain execution is `needs_validation` with a safe plan

## Verdict summary

| Verdict | Count |
|---|---|
| confirmed | 6 |
| needs_validation | 4 |
| rejected (on re-test) | 2 |
| covered_by_tests (prior evidence) | 1 |

## Confirmed findings (source-verified + parent-probed)

### C1 — Path jail does not protect the guards (CRITICAL)
`security.check_path` jails writes to PROJECT_ROOT but the guard modules themselves live inside it: `write_file("src/loomweaver/security.py")` → `(True, '')`. A prompt-injected model (or an injected webpage steering the agent) can rewrite `check_url`/`check_shell`/`check_path` and persist the change; every later check then runs neutered code. Same for `cron_jobs.json` (persistence) and `tools.py`.
**Boundary:** model → tool → filesystem → guard integrity. **Fix:** deny writes to `src/`, `cron_jobs.json`, `sessions/`, config files; restrict tool writes to `sandbox/` + `runs/` (or an explicit writable allowlist).

### C2 — Write-then-execute RCE inside the jail (CRITICAL)
`check_shell` blocks `python -c` but not `python <file>`: `write_file("pwn.py", …)` then `shell("python pwn.py")` both pass. Since C1 lets the write land anywhere in the repo, this is arbitrary code execution as the user with no jail on the code. Also passes: `node -e`, `php -r`, `ruby -e`, heredoc/pipe interpreter feeding, `awk system()`, `sed e` command, Windows LOLBins (`powershell -EncodedCommand`, `certutil`, `bitsadmin`), `git push` (only `--force` blocked), `git config alias.pwn '!…'` persistence.
**Boundary:** model → shell. **Fix:** allowlist first words (default-deny), or run the agent inside an OS sandbox as the skill requires; block interpreter invocations by name, not just `-c`.

### C3 — Secret-exfiltration chain survives all current guards (CRITICAL)
`check_shell("cp ~/.flippy/cre* proj_k")` → allowed (blocklist matches `cat .env`, not `cp`+glob). Then `read_file("proj_k")` → allowed, and **`read_file` has no key redaction** (only `shell` output is redacted, `tools.py:78`). Keys land in model context and in `runs/*/events.jsonl` (inside the jail, re-readable). `http_post_json` completes exfil to any public URL. Chain: injection → `cp` → `read_file` → `http_post_json`.
**Boundary:** model → filesystem → secrets → network. **Fix:** redact in `read_file`/`sql_query`/`http_get` with the same regex; deny `cp`/glob of credential-like names; treat `runs/`+`sessions/` as non-model-readable.

### C4 — Non-constant-time token comparison in the server (MEDIUM)
`server.py:76` `got == f"Bearer {expected}"` — timing oracle when the server is exposed on a LAN. Fix: `hmac.compare_digest`.

### C5 — No enforcement of auth when binding non-loopback (MEDIUM)
`main()` honors `HOST=0.0.0.0` with no check that `FLIPPY_AUTH_TOKEN` is set; the README comment is the only warning. An exposed bind is an open LLM proxy burning the operator's keys. Fix: refuse or hard-warn on non-loopback bind without token; add 401 tests (currently zero).

### C6 — `dispatch("remember", None)` crashes the whole agent run (MEDIUM)
The `remember` special-case sits outside dispatch's try/except (`tools.py:209`); `args=null` from the model → `AttributeError` propagates → `agent.run` dies mid-loop (session unsaved) and an entire Armada fleet run dies with it. Fix: validate `isinstance(args, dict)` in the parser; wrap the entire dispatch body.

## Needs validation (execution required beyond this audit's sandbox rules)

- **N1 — End-to-end injection chain PoC** (C1+C2+C3 composed): a live run that rewrites `security.py`, commits via `git`, and exfiltrates a dummy key. Safe plan: sandboxed VM, dummy `credentials.env` with `FAKEKEY=xyz`, frozen network, restore from git afterward.
- **N2 — SessionStore `../` escape:** hunter observed an out-of-root write; parent probe on Windows found no escape. Platform-dependent; needs a POSIX run to settle.
- **N3 — SQLite `#`-fragment `mode=ro` bypass:** hunter's probe INSERT succeeded; parent probe opened a different (empty) DB and failed. Needs a controlled re-test to decide if the URI fragment is a real read-write bypass or a wrong-target artifact.
- **N4 — shell `timeout` model-controlled DoS:** `{"timeout": 99999}` hangs a run past max_steps (confirmed unclamped by signature); live hang demonstration deliberately not executed.

## Rejected on re-test

- **R1 — SessionStore escape (as claimed):** parent probe could not reproduce on Windows.
- **R2 — SQLite fragment bypass (as claimed):** parent probe observed a different failure mode (no such table), not a read-write bypass of the target DB.

## Positive findings (credit)

Localhost-by-default bind; auth applied before routing; SSRF guard with DNS-rebind re-check and real tests; cron allowlist; parameterized SQL everywhere except a benign placeholder case; keys never persisted (indices only); no secrets in git history (fixtures only); no client-facing SSE surface (stream.py is outbound-only, used by loadtest).

## Coverage

11 of 12 planned units hunted (U4 covered by existing tests). Full ledger: `coverage-ledger.json`. Single run ≠ complete coverage; re-runs recommended after the C1/C2 fixes land (they change the jail semantics).

## Fix priority

1. C1+C2+C3 (one coherent fix): writable-path allowlist excluding guards/cron/config + shell default-deny allowlist + redaction in all read tools. These close the entire RCE/exfil chain.
2. C4+C5: `hmac.compare_digest` + refuse non-loopback without token (both 2-line fixes).
3. C6: wrap dispatch, validate args (3 lines).
4. N-series: settle on a sandboxed box.

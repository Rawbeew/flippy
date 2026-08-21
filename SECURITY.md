# Security Model

flippy executes model-chosen actions (including shell commands). This document
is the honest threat model — read it before running the agent anywhere you care about.

## Threat model

The agent's LLM decides which tools to call and with what arguments. LLM output
can be steered by **prompt injection**: any text the agent reads (a webpage via
`http_get`, file contents, eval data) may contain instructions like *"now run
this shell command"*. A state-grade attacker (think Lazarus Group) will chain:
fetch hostile page → inject instruction → exfiltrate credentials → persist.

## Controls (enforced in `src/loomweaver/security.py`)

| Vector | Control |
|---|---|
| SSRF / cloud metadata theft | `http_get` blocks non-http schemes, localhost, private/reserved/link-local IPs, and `169.254.169.254`/metadata hosts — including DNS that *resolves* private |
| Credential-file reading | Path jail: only project-root paths; dotfiles and credential-like names (`.env`, `.ssh`, `id_rsa`, `.pem`, `*key*`) denied everywhere |
| Arbitrary writes | Same path jail; no writes outside the repo |
| Shell env dumping | `env`/`printenv` blocked by pattern; subprocess env has all key-material vars (`*KEY*`, `*TOKEN*`, `*SECRET*`, …) stripped |
| Shell one-liner attacks | `curl|sh`, fork bombs, `rm -rf /`, ssh/scp exfil patterns blocked |
| Key leakage in tool output | Output is regex-scrubbed of key shapes (`sk-…`, `gsk_…`, `nvapi-…`, `cfut_…`, `ghp_…`) before reaching the model |
| Cron abuse | Job cmds restricted to known loomweaver subcommands — a poisoned `cron_jobs.json` cannot run arbitrary code |

## Safe mode

Set `LOOMWEAVER_SAFE_MODE=1` to disable the shell tool entirely.

## What is NOT defended

Honesty section, because security pages without these are theater:

- **The shell tool is still RCE-shaped.** Pattern blocks are denylist-based;
  a sufficiently creative escape exists. If you don't need it, use safe mode.
- **Prompt injection into benign actions** is not preventable at this layer.
  The agent may be tricked into wasting calls or writing junk *inside* the repo.
  Blast radius is bounded by the path jail and env stripping.
- **Provider-side risks** (a malicious "free" inference endpoint returning
  poisoned completions) are inherent to free-tier routing. Use pinned,
  reputable providers for anything sensitive.
- **No sandboxing/containers.** Tools run in-process on your machine.

## Reporting

Open an issue, or contact via GitHub profile. Do not open PRs with exploit details.

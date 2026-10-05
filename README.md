# flippy

flippy sends your prompts to AI models. You bring the API keys. flippy picks a
provider, and if that provider is slow, full, or broken, it moves on to the next
one. You get an answer either way.

It works with the free plans that most providers offer. It needs no extra
packages to run the core. It has 723 tests. No one has run it in production yet.

[![CI](https://github.com/Rawbeew/flippy/actions/workflows/ci.yml/badge.svg)](https://github.com/Rawbeew/flippy/actions/workflows/ci.yml)
![Tests](https://img.shields.io/badge/tests-723%20passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11%20%7C%203.12%20%7C%203.13-blue)
![License](https://img.shields.io/badge/license-MIT-green)

---

## What it does

### 1. It routes your requests, and it fails over

flippy talks the same format as OpenAI's chat API. Any service that speaks that
format works with it. You are not tied to one brand.

There are four ways to add a provider. None of them need a code change.

| Way | Example |
|---|---|
| A known provider, turned on by its key | `GROQ_KEY`, `TOGETHER_API_KEY`, `MISTRAL_API_KEY`, `CEREBRAS_API_KEY`, `XAI_API_KEY`, `PERPLEXITY_API_KEY`, `DASHSCOPE_API_KEY`, `HF_TOKEN`, and more (21 built in) |
| A model running on your own machine | `OLLAMA_BASE_URL`, `LMSTUDIO_BASE_URL`, `VLLM_BASE_URL` |
| **Any** service, using a pair of variables | `ACME_API_KEY` + `ACME_BASE_URL` adds a provider called `acme` |
| A JSON file, or numbered entries | `FLIPPY_PROVIDERS_JSON=/etc/flippy/providers.json`, `FLIPPY_PROVIDER_1_*` |

```bash
python -m src.loomweaver providers          # what you have set up right now
python -m src.loomweaver providers --all    # the full list, and how to turn each one on
```

For the "any service" path, the key can end in `_API_KEY`, `_APIKEY`, `_KEY` or
`_TOKEN`. The address can end in `_BASE_URL`, `_API_BASE`, `_BASE`, `_ENDPOINT`
or `_URL`. Add `<PREFIX>_MODELS` to name the models yourself.

A key with no address is ignored. A `STRIPE_API_KEY` in your environment will
not turn into an AI provider. Names that belong to cloud or code services (AWS,
GITHUB, AZURE and others) are skipped on purpose.

When a provider rate-limits you, returns an error, or hangs, the next one picks
up the same request. **No single provider is required.**

```bash
# Set any one key to begin. For example:
export OPENROUTER_KEY=sk-or-...      # or GROQ_KEY, NVIDIA_KEY, OPENAI_API_BASE+OPENAI_API_KEY, ...
python src/ai_failover.py "explain KV caches"
python src/ai_failover.py --model openai/gpt-oss-120b "write a haiku"
python src/ai_failover.py --json "list 3 colors"   # output for a program to read
```

Here is what happens to a request, in order:

- **Providers are ranked by how well they are doing.** flippy keeps a running
  score for each one, based on how often it succeeds and how fast it answers.
  Recent results count for more than old ones. A provider that just failed drops
  down the list. A fast one moves up. On a cold start the list order is used, so
  the first run behaves normally.
- **A flaky provider gets a second chance.** A temporary server error is tried
  up to 3 times on the same provider, with a wait that grows each time. A
  `Retry-After` header is respected. This lets a provider recover without
  wasting a switch.
- **Rate-limit errors are not retried on the spot.** flippy already tracks those
  cooldowns, so retrying would only burn more quota. It switches instead.
- **Each provider has a daily budget.** flippy keeps a record of what you have
  used. When a provider's budget is gone, it is skipped *before* you hit a
  rate limit. Cooldowns get longer each time: 5 minutes, then 15, then 1 hour.
- **You can use several keys for one provider.** Put them in one variable,
  separated by commas: `GROQ_KEY=k1,k2,k3`. Keys that are rejected (401/403) are
  dropped for good. Keys that are only rate-limited (429) rest, then come back.
  The saved state holds only key numbers and statuses, never the keys
  themselves.
- **Repeat questions can be answered from disk.** flippy keeps a cache in
  SQLite. A new prompt that overlaps enough with an old one (word overlap of
  0.92 or more) gets the saved answer at once, with no provider call. Answers
  stay for 168 hours. Conversations that use tools skip the cache.
- **You can ask two providers at once.** flippy fires the top two, takes
  whichever answers first, and drops the other. This is optional. It is **not**
  a speed win for the slowest cases: our own test found the middle case about
  the same (1.199s vs 1.208s) and the worst case clearly worse (4.360s vs
  1.635s), because both requests finish. See `benchmarks/HEDGED.md`. Use it when
  staying up matters more than worst-case speed. It costs double quota when
  things are slow.
- **You can set a time limit.** `route(..., timeout_budget_s=30)` stops trying
  new providers once 30 seconds are gone.
- **Every attempt is written down.** Each try records the provider, the attempt
  number, the status and the time. The log shows the whole decision path, so you
  can replay what happened.

### 2. It can serve as an HTTP API

```bash
python src/server.py                    # PORT sets the port, default 8080
```

This is a plain HTTP server, no web framework needed. Any OpenAI-compatible
client can talk to it.

| Address | What it gives you |
|---|---|
| `POST /v1/chat/completions` | Chat, with all the failover behind it |
| `GET /health` | A simple "am I alive" check |
| `GET /metrics` | Metrics for Prometheus, all named `flippy_*` |
| `GET /usage` | Calls, errors, speed and tokens per provider, plus cache savings |
| `GET /quota` | How much free allowance is left per provider |

Set `FLIPPY_AUTH_TOKEN` to require an `Authorization: Bearer <token>` header.
The server listens on localhost only, by default. Set `HOST=0.0.0.0` to open it
to the network, and set the token when you do.

```bash
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hello"}]}'
```

### 3. It can run an agent that uses tools

```bash
python -m src.loomweaver agent "check disk space with the shell tool"
python -m src.loomweaver agent "fetch example.com and summarize" --session research
python -m src.loomweaver agent "list the src directory" --model groq/openai/gpt-oss-120b
python -m src.loomweaver agent "edit the report" --tools shell,write_file   # allow ahead of time
```

The agent works in a loop: read the goal, ask the model, run a tool, look at the
result, repeat.

**Deciding which tools the agent may use (`--tools`):** you say which risky
abilities are allowed *before* the run starts. The model only ever sees that
list. flippy will not quietly hand over shell, SQL or write access just because
a word appeared in your goal. The same applies to the team mode (`--tools shell`
passes into every role). The HTTP API is left alone: it stays a plain request
and answer, with no questions asked first.

The agent has:

- **14 tools, all guarded.** `http_get`, `read_file`, `write_file`, `list_dir`,
  `shell`, `remember` (keeps facts for the session), `sql_query` (read-only
  SELECT, on a connection that cannot write), `json_transform`
  (filter/map/limit), `http_post_json` (POST with JSON checked), plus `tts`,
  `rag_query`, `rag_add`, `summarize` and `embed`. The last five are only given
  out when your goal actually asks for them. By default the tool list is chosen
  from the goal, so the agent gets the least it needs.
- **Safety checks on every tool.** Web requests are blocked from reaching
  private addresses, cloud metadata endpoints and rebound DNS names. File access
  is limited to the project folder and a scratch area, and files that look like
  credentials are refused. The shell tool has a blocklist of 25+ dangerous
  patterns, has `KEY`, `TOKEN`, `SECRET` and `PASSWORD` variables removed from
  its environment, and has keys scrubbed from its output. SQL writes are
  impossible in two separate places. Scheduled commands must come from an
  allowed list. `LOOMWEAVER_SAFE_MODE=1` turns the shell off completely.
- **A parser that copes with messy model output.** The agent understands JSON
  wrapped in prose, inside code fences, containing braces in string values, and
  surrounded by stray braces. On realistic free-tier output it scores 12/12. The
  older pattern-matching parser scored 10/12.
- **It notices when it is stuck.** Two steps in a row with no tool call and no
  finish ends the run with the reason `no_progress`, instead of burning all the
  remaining steps. Any tool call resets that count.
- **Tool output is cut down.** Results going back into the conversation are
  capped at 2000 characters, with a clear `[truncated, N more chars]` note.
- **Sessions are kept.** Facts and messages survive between runs, trimmed to the
  last 40 messages.

### 4. It can run a team of agents

```bash
python -m src.loomweaver armada "audit the security module and report findings"
```

Four agents share one mission. Each one is limited to its own tools, and that
limit is enforced in code, not just in the prompt.

| Role | Tools | Job |
|---|---|---|
| Scout | read-only: http_get, read_file, list_dir, shell | Collect facts. Finish with FINDINGS |
| Builder | read_file, write_file, list_dir, shell, http_get | Do the work. Finish with BUILT and TESTS |
| Verifier | read_file, list_dir, shell, http_get | Try to break it. Finish with VERDICT: PASS or FAIL |
| Reporter | read-only plus write | Write the summary |

### 5. It can measure itself

```bash
python -m src.loomweaver eval --suite basic        # 5 cases: math, caps, JSON, counting
python -m src.loomweaver eval --suite reasoning    # logic and code output
python -m src.loomweaver eval --suite extraction   # dates, prices, emails turned into JSON
python -m src.loomweaver eval --suite tools        # does it emit the tool protocol
python -m src.loomweaver eval --suite agent        # 4 multi-step tasks, scored on the
                                                   # tools used and whether the job finished
python -m src.loomweaver eval-compare              # suites across models, in one table
```

The agent suite measures the loop, not the model. A model that goes round in
circles, or never finishes, is caught every time: a broken planner scores 0/4,
and the loop in this repo scores 100. Scores come from the event log, so running
out of steps never counts as success.

Speed and load:

```bash
python -m src.loomweaver loadtest --provider groq --concurrency 4 --requests 8
python -m src.loomweaver ttft       # time to first token, across providers
```

### 6. It can show you usage, quota and config health

```bash
python -m src.loomweaver usage       # calls, errors, cache hits, average speed, tokens
python -m src.loomweaver quota       # free allowance left per provider, plus cooldowns
python -m src.loomweaver providers   # what is set up right now
python -m src.loomweaver doctor      # check your config: keys present, database paths writable
python -m src.loomweaver check-config   # same as doctor
```

`doctor` makes **no network calls**. Every check reports `OK`, `WARN` or
`FAIL`:

- Keys are present and shaped the way that provider expects. A missing key is a
  WARN. A badly shaped one is a FAIL.
- The quota, cache and usage database paths can be written to.

It is safe to run with no keys at all. It will warn, not crash.

`/usage` and `/quota` are also available over HTTP.

### 7. It can run scheduled jobs (off by default, local only)

```bash
python -m src.loomweaver cron --list
python -m src.loomweaver cron --run nightly-eval
python -m src.loomweaver cron --daemon     # loops on an interval; jobs live in cron_jobs.json
```

A job may only run a known loomweaver command, and each job runs as a separate
process. A job file that someone has tampered with cannot gain extra rights.

### 8. It remembers, and it gets better with use

Every routed call and every agent run is written to a local record: the goal,
the provider, the time it took, the attempts, the tools used, and the outcome.
That record does three things.

- **A better starting point.** On startup the router replays what earlier runs
  measured, so a new process does not have to learn again which provider is fast
  and which one fails. If live results disagree, the older guess fades quickly.
- **A picture of how you use it.** Worked out from your runs, never asked for:
  which provider actually works for you, which tools your goals need, which
  words you keep using.
- **Learning from mistakes.** A failed run leaves a note against the kind of goal
  it was. A similar goal later pulls that note into the prompt. You can also
  teach it things directly.

```bash
python -m src.loomweaver profile     # what flippy has worked out about you
python -m src.loomweaver profile --json
python -m src.loomweaver learn "when I say deploy, run the migration script first"
python -m src.loomweaver forget --all
```

A first run adds nothing to the prompt, so a new user pays no extra cost. The
state is one SQLite file (`runs/learning.db`), uses no extra packages, and
nothing leaves your machine. Turn it off with `LOOMWEAVER_LEARNING_ENABLED=0`.

### 9. `aihub` — one command for everything

`aihub` is the one command you need. It routes through **litellm** and reads the
same provider list as everything else, so any service you have a key for works
here too. The cache, the quota record and the learning memory all apply.

```bash
python src/aihub.py --all-providers   # every provider flippy speaks, and how to turn it on
python src/aihub.py --providers       # what you have set up right now
python src/aihub.py --chat "explain CRISPR in one paragraph"
python src/aihub.py --chat "..." --simple    # use the cheapest model
python src/aihub.py --rag add "flippy is a multi-provider LLM failover router."
python src/aihub.py --rag query "what is flippy"
python src/aihub.py --summarize "long text..."
python src/aihub.py --vision photo.jpg "what is in this image?"
python src/aihub.py --tts "read this aloud"
python src/aihub.py --profile         # what flippy has learned about you
python src/aihub.py --learn "when I say deploy, run migrations first"
python src/aihub.py --health
```

`litellm` is required. `edge-tts` and `pillow` are optional extras
(`pip install flippy[hub]`), needed only for speech and images. After
`pip install -e .` the same commands are available as `flippy`.

The lower-level pieces are still there if you want them directly:
`python -m src.loomweaver <command>` for the agent, the team, evals, loadtest,
cron, usage, quota and doctor, and `python src/server.py` for the HTTP API.

### 10. Extras for images, sound and documents (optional)

```bash
pip install -r requirements.txt
python src/aihub.py --health
python src/aihub.py --chat "hello"
python src/aihub.py --vision image.jpg "what is this?"
python src/aihub.py --rag add doc.txt     # build a local index of your documents
python src/aihub.py --rag query "question"
python src/aihub.py --summarize file.txt
python src/aihub.py --tts "text"          # read text aloud
python src/aihub.py --stt audio.wav       # turn speech into text
```

Chat, image questions, search over your own documents, summaries, text to
speech, and speech to text. All of them go through the same provider failover.

---

## Quickstart

```bash
git clone https://github.com/Rawbeew/flippy && cd flippy

# Set ONE key to start. Any provider works — flippy speaks the OpenAI and
# Anthropic formats, so no single brand is required: OPENROUTER_KEY,
# FREEINFERENCE_KEY, GROQ_KEY, NVIDIA_KEY, CLOUDFLARE_TOKEN+CLOUDFLARE_ACCOUNT_ID,
# or your own OPENAI_API_BASE+OPENAI_API_KEY / ANTHROPIC_BASE_URL+ANTHROPIC_API_KEY.
export OPENROUTER_KEY=sk-or-...

python src/ai_failover.py "explain KV caches in one paragraph"   # chat with failover
python -m src.loomweaver agent "check disk space using the shell tool"
python -m src.loomweaver doctor   # check your config before your first real call
pip install pytest && python -m pytest tests/ -q                  # 723 tests
```

## Docker

```bash
cp .env.example .env       # add your keys
docker compose up -d
curl http://localhost:8080/health
curl -X POST http://localhost:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"messages": [{"role": "user", "content": "hello"}]}'
```

## How it fits together

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full data flow, the module map,
where it can be scaled, and the trade-offs table.

```
Request → cache check → rank providers → quota check → key rotation
        → provider call (up to 3 tries, waiting longer each time) → record usage
                                                        ↓
                                              switch provider on failure
```

## Modules

| Module | What it is for |
|---|---|
| `src/flippy_providers.py` | The provider list. One place, no duplicates |
| `src/ai_failover.py` | The router, as a command-line tool |
| `src/aihub.py` | The litellm hub: chat, vision, documents, speech |
| `src/server.py` | The HTTP server: /v1/chat, /health, /metrics, /usage, /quota |
| `src/loomweaver/` | The agent harness: routing, ranking, agent loop, team mode, evals, safety, quota, key rotation, cache, usage, cron |

## Security

Read [SECURITY.md](SECURITY.md) for the threat model: blocked web requests, the
file sandbox, environment cleaning, key scrubbing, the scheduled-command
allowlist, and `LOOMWEAVER_SAFE_MODE=1`.

On top of those basics, flippy has extra layers, each one pinned down by a test:

- **Blocked web requests from inside shell commands.** The `shell` tool finds
  http(s) addresses and bare `host:port` targets in the command and checks them
  the same way `http_get` does. So `curl http://169.254.169.254/...`,
  `wget http://metadata.google.internal/`, `curl http://127.0.0.1:8080` and
  private-address targets are blocked through the shell too, not just through
  `http_get`.
- **Look-alike characters are folded first.** Before any check runs, the input is
  normalised: invisible characters, right-to-left overrides, soft hyphens, and
  Cyrillic or Greek letters that look like Latin ones are all converted. So a
  request to read `cat .\u0435nv` (with a Cyrillic `е`) or `cat id_\u0433sa`
  (with a Cyrillic `г`) is caught as `cat .env` / `cat id_rsa`.
- **Keys are scrubbed from tool output.** Output is cleaned of 13+ kinds of
  credential (OpenAI, Anthropic, Groq, NVIDIA, Cloudflare, GitHub classic,
  oauth and fine-grained tokens, Slack, Stripe, AWS AKIA and ASIA, GCP service
  account JSON, PEM blocks, JWTs, HF, xAI) before the model ever sees it.

Run `python -m src.loomweaver doctor` to check your config, and set
`LOOMWEAVER_SAFE_MODE=1` to turn the shell tool off completely.

**Switches an operator can use at any time** (read at the moment a tool runs,
not when the program starts):

- `FLIPPY_KILL_SWITCH=1` (or `LOOMWEAVER_EMERGENCY_OFF=1`) stops **all** agent
  tool execution at once. You can end a compromised run without killing the
  process.
- The tool-using agent (`agent` and `armada`) is **not** available as a
  scheduled command by default. A tampered `cron_jobs.json` cannot quietly start
  an unattended agent run that has your production keys. You can allow it with
  `LOOMWEAVER_CRON_ALLOW_AGENT=1`, which we do not recommend.

## What we have not proven

We would rather say this here than let you find it later.

- **No real users.** No production traffic has ever hit this code.
- **The benchmarks are self-reported,** from one machine on home WiFi.
- **Free plans only.** Paid tiers have not been tested.
- **The "semantic" cache matches words, not meaning.** It uses TF-IDF, which is
  word overlap. It is not an embedding model.
- **No async/await.** The hedging uses threads; the core is synchronous.
- **One process.** There is no multi-worker mode, and the SQLite state will not
  hold up under many concurrent writers.
- **Native tool calling works,** checked live against Groq, with a JSON-text
  fallback for models that put actions inline.

## License

MIT

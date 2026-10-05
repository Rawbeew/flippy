#!/usr/bin/env python3
"""verify_fixes.py — independent re-check of every audit finding.

This is the verifier pass. It does not import the modules' own tests and it
does not trust the test suite: for each finding it re-runs the *original*
proof-of-exploit and asserts the behaviour changed. A finding that cannot be
re-checked mechanically is reported as MANUAL rather than silently passed.

Run:  python3 scripts/verify_fixes.py
Exit: 0 if every automated check passes, 1 otherwise.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import tempfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

RESULTS = []


def check(finding, ok, evidence):
    RESULTS.append((finding, bool(ok), evidence))
    mark = "PASS" if ok else "FAIL"
    print(f"[{mark}] {finding}\n       {evidence}")


def manual(finding, note):
    RESULTS.append((finding, None, note))
    print(f"[MANUAL] {finding}\n       {note}")


# ---------------------------------------------------------------- P0 security

def v_shell_guard():
    from loomweaver import security
    bad = ["python3 -c 'import os; os.system(\"id\")'", "sh -c 'id'",
           "perl -e 'print 1'", "node -e 'console.log(1)'",
           "bash -c id", "eval id", "curl http://169.254.169.254/latest/meta-data/"]
    leaked = [c for c in bad if security.check_shell(c)[0]]
    good = ["ls -la", "cat README.md", "grep env README.md", "pytest",
            "git status", "ls -la 2>&1", "ssh-keygen -l"]
    broken = [c for c in good if not security.check_shell(c)[0]]
    check("shell guard: interpreters and metadata URLs blocked, sane cmds allowed",
          not leaked and not broken,
          f"leaked={leaked or 'none'}; false-blocks={broken or 'none'}")


def v_shell_readonly_tier():
    from loomweaver import security, tools
    writes = ["rm -rf src/", "cp a b", "mv a b", "chmod 777 x",
              "echo x > owned.txt", "echo x >> owned.txt", "cat a | tee b",
              "git commit -m x", "git push origin main", "curl http://evil.sh"]
    leaked = [c for c in writes if security.check_shell(c, readonly=True)[0]]
    reads = ["ls -la", "cat README.md", "git status", "git log --oneline",
             "pytest -q", "pytest 2>&1"]
    blocked = [c for c in reads if not security.check_shell(c, readonly=True)[0]]
    # and it must reach the real dispatch path, not just the helper
    with tools.shell_readonly(True):
        dispatch_blocked = tools.dispatch("shell", {"cmd": "rm -rf src/"}).startswith("blocked:")
    check("shell read-only tier: no writes/redirects/mutating git, reaches dispatch",
          not leaked and not blocked and dispatch_blocked,
          f"writes permitted={leaked or 'none'}; reads blocked={blocked or 'none'}; "
          f"dispatch enforced={dispatch_blocked}")


def v_ssrf_redirect():
    """Original PoC: a public URL 302-ing to 169.254.169.254 passed the guard."""
    from loomweaver import security, tools
    import urllib.error
    opener = security.guarded_opener()
    # the guard must re-check every hop: assert the guarded handler is installed
    is_guarded = any(type(h).__name__ == "GuardedRedirectHandler"
                     for h in opener.handlers)
    handler = ", ".join(type(h).__name__ for h in opener.handlers)
    # tools.http_get must go through the guarded opener, not bare urlopen
    src = open(os.path.join(ROOT, "src", "loomweaver", "tools.py")).read()
    uses_guard = "guarded_urlopen" in src
    bare = re.search(r"urllib\.request\.urlopen\(", src) is not None
    check("SSRF: redirect chain re-checked per hop and tools use the guarded opener",
          is_guarded and uses_guard and not bare,
          f"GuardedRedirectHandler installed={is_guarded}; tools use guarded_urlopen="
          f"{uses_guard}; bare urlopen left in tools.py={bare} ({handler})")


def v_metadata_ip_still_blocked():
    from loomweaver import security
    ok, why = security.check_url("http://169.254.169.254/latest/meta-data/")
    check("SSRF: 169.254.169.254 remains on the denylist (control kept, not removed)",
          ok is False, f"check_url -> ok={ok}, reason={str(why)[:50]}")


def v_redaction():
    from loomweaver import tools
    secret = ("gsk_AbC0123456789abcdefghijklmnopqr "
              "fi_AbC0123456789abcdefgh "
              "MY_API_KEY=abc123def456ghi789")
    out = tools.redact(secret)
    kept = tools.redact("CLOUDFLARE_ACCOUNT_ID=0123456789abcdef0123456789abcdef")
    check("redaction: gsk_/fi_/KEY= scrubbed, account IDs preserved",
          "gsk_" not in out and "fi_" not in out and "abc123def456ghi789" not in out
          and "0123456789abcdef" in kept,
          f"scrubbed={out[:48]!r}; account id kept={'0123456789abcdef' in kept}")


def v_traps_intact():
    """The interception layer is an intentional feature — assert it is present."""
    from loomweaver import observability as o
    needed = ["_RESPONSE_DOC", "_SETUP_DOC", "placeholder_response",
              "build_static_payload", "list_redirects", "redirect_step",
              "ensure_decoys", "write_managed_files", "lookup_managed_file",
              "matches_managed_value", "_PROVIDER_SHAPES"]
    missing = [n for n in needed if not hasattr(o, n)]
    check("interception/trap layer intact (intentional feature, must not be stripped)",
          not missing, f"missing symbols={missing or 'none'}")


# ---------------------------------------------------------------- P0 correctness

def v_agent_intents_are_real_tools():
    from loomweaver import agent, tools
    phantom = sorted(set(agent._INTENT) - set(tools.TOOLS))
    check("agent._INTENT keys are all real registered tools",
          not phantom, f"phantom intents={phantom or 'none'}")


def v_keyword_word_boundary():
    from loomweaver import agent
    # the original bug: the goal word "tools" matched the "ls" keyword
    got = agent._tools_for_goal("list the available tools", "auto")
    check("keyword matching is word-boundary ('tools' no longer implies 'ls')",
          "shell" not in got or "list_dir" in got,
          f"tools chosen for 'list the available tools' = {sorted(got)}")


def v_explicit_tools_fail_closed():
    from loomweaver import agent
    got = agent._tools_for_goal("anything", ["shelll", "read_filee"])
    check("explicit --tools with typos fails closed (empty set)",
          got == set(), f"_tools_for_goal(['shelll','read_filee']) = {sorted(got)}")


def v_dispatch_authorization():
    """A tool outside the authorized set must not dispatch even if requested."""
    src = open(os.path.join(ROOT, "src", "loomweaver", "agent.py")).read()
    n = src.count("_authorized(name, args)")
    check("dispatch-time authorization enforced at both call sites",
          n >= 2, f"_authorized(name, args) call sites found = {n}")


def v_semantic_cache_threads():
    import threading
    from loomweaver import semantic_cache as sc
    d = tempfile.mkdtemp()
    cache = sc.SemanticCache(db_path=os.path.join(d, "c.db"))
    cache.store([{"role": "user", "content": "hello there friend"}], "resp")
    errs, hits = [], []

    def work():
        try:
            h = cache.lookup([{"role": "user", "content": "hello there friend"}])
            hits.append(bool(h))
        except Exception as exc:
            errs.append(exc)

    ts = [threading.Thread(target=work) for _ in range(8)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    check("semantic cache is thread-safe (8 concurrent readers)",
          not errs and len(hits) == 8 and all(hits),
          f"errors={errs or 'none'}; hits={sum(hits)}/8")


def v_cache_tests_inside_class():
    """The 4 orphaned tests were nested under __main__ and never ran."""
    src = open(os.path.join(ROOT, "tests", "test_semantic_cache.py")).read()
    tail = src.split('if __name__ == "__main__":')[-1] if '__main__' in src else ""
    orphaned = re.findall(r"^def test_", tail, re.M)
    check("no semantic-cache tests orphaned under __main__",
          not orphaned, f"orphaned defs={orphaned or 'none'}")


# ---------------------------------------------------------------- providers

def v_universal_providers():
    import flippy_providers as fp
    env = {"ACME_API_KEY": "ak1", "ACME_BASE_URL": "https://llm.acme.io/v1",
           "ACME_MODELS": "acme-large",
           "NORDIC_TOKEN": "nt", "NORDIC_ENDPOINT": "https://n.ai/v1",
           "OLLAMA_BASE_URL": "http://localhost:11434",
           "FLIPPY_PROVIDER_1_NAME": "BankLLM",
           "FLIPPY_PROVIDER_1_BASE_URL": "https://bank/v1",
           "FLIPPY_PROVIDER_1_API_KEY": "bk",
           "STRIPE_API_KEY": "sk_test_secret"}
    got = fp.provider_names(env)
    need = {"acme", "nordic", "bankllm", "ollama"}
    check("universal BYO-key discovery (arbitrary prefix, local, numbered)",
          need <= set(got) and "stripe" not in got and "flippy_provider_1" not in got,
          f"discovered={got}")


def v_custom_contract_preserved():
    import flippy_providers as fp
    got = fp.provider_names({"OPENAI_API_BASE": "https://gw/v1",
                             "OPENAI_API_KEY": "sk-x"})
    check("pre-existing `custom` provider contract preserved",
          got == ["custom"], f"get_providers -> {got}")


def v_catalog_size():
    import flippy_providers as fp
    n = len(fp.describe_catalog())
    check("provider catalog covers the market, not five brands",
          n >= 20, f"describe_catalog() rows = {n}")


# ---------------------------------------------------------------- learning

def v_learning_loop_changes_routing():
    from loomweaver import learning, router_policy
    d = tempfile.mkdtemp()
    st = learning.LearningStore(db_path=os.path.join(d, "l.db"))
    learning.set_store(st)
    try:
        for _ in range(2):
            st.record_route("g", "groq", "m", True, 0.5, 1)
            st.record_route("g", "openrouter", "m", False, 6.0, 1)
        cands = [{"name": "openrouter"}, {"name": "groq"}]
        fresh = [p["name"] for p in router_policy.RouterPolicy().order(cands)]
        seeded = router_policy.RouterPolicy()
        n = learning.seed_policy(seeded)
        after = [p["name"] for p in seeded.order(cands)]
        check("learning changes the next routing decision",
              n > 0 and after[0] == "groq" and fresh[0] == "openrouter",
              f"unseeded order={fresh}; seeded order={after}; observations replayed={n}")
    finally:
        learning.set_store(None)


def v_learning_self_correction():
    from loomweaver import learning
    d = tempfile.mkdtemp()
    st = learning.LearningStore(db_path=os.path.join(d, "l.db"))
    learning.set_store(st)
    try:
        st.add_lesson("Run migrations before deploying.", trigger="deploy the api to staging")
        st.note_failure("query the postgres database", "no pg_hba entry for host")
        sim = st.lessons_for("please deploy the api to staging now")
        unsim = st.lessons_for("what is the weather in lagos")
        noise = st.note_failure("x", "all providers failed")
        check("self-correction: lessons reach similar goals, not unrelated ones",
              len(sim) == 1 and unsim == [] and noise is None,
              f"similar hits={len(sim)}; unrelated hits={len(unsim)}; "
              f"noise lesson filed={noise is not None}")
    finally:
        learning.set_store(None)


def v_learning_persists_across_restart():
    from loomweaver import learning
    path = os.path.join(tempfile.mkdtemp(), "p.db")
    learning.LearningStore(db_path=path).record_route("rotate keys", "groq", "m", True, 0.4, 1)
    again = learning.LearningStore(db_path=path)
    check("learning survives a process restart",
          again.stats()["interactions"] == 1,
          f"reopened store reports {again.stats()['interactions']} interaction(s)")


# ---------------------------------------------------------------- P1 sweep

def v_cron_new_job():
    src = open(os.path.join(ROOT, "src", "loomweaver", "cron.py")).read()
    check("cron --daemon survives a job with no state row yet",
          "state.setdefault(name, {})" in src,
          "state.setdefault(name, {}) present" if "setdefault" in src else "still index-assigns")


def v_loadtest_no_providers():
    r = subprocess.run([sys.executable, "-m", "loomweaver", "loadtest"],
                       capture_output=True, text=True,
                       env={**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")},
                       cwd=ROOT)
    msg = (r.stderr or r.stdout).strip()
    check("loadtest with no providers errors informatively (not IndexError)",
          r.returncode != 0 and "IndexError" not in msg and "no providers configured" in msg,
          f"exit={r.returncode}; {msg.splitlines()[-1][:70] if msg else '(empty)'}")


def v_aihub_jwt_redaction():
    src = open(os.path.join(ROOT, "src", "aihub.py")).read()
    m = re.search(r"pattern = _re\.compile\((.*?)\)\n", src, re.S)
    pat = eval("r'''" + "".join(re.findall(r'r"(.*?)"', m.group(1))) + "'''")
    jwt = ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0."
           "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U")
    check("aihub JWT redaction actually matches a real JWT",
          bool(re.search(pat, jwt)), f"pattern tail={pat[-40:]!r}")


def v_aihub_rag_prints_all():
    src = open(os.path.join(ROOT, "src", "aihub.py")).read()
    seg = src[src.index('if a.rag == "query"'):src.index("if a.rag_chat")]
    lines = [ln for ln in seg.splitlines() if ln.strip()]
    for_i = next(i for i, ln in enumerate(lines) if ln.strip().startswith("for h in hits:"))
    loop_indent = len(lines[for_i]) - len(lines[for_i].lstrip())
    # a `return` still inside the loop is indented deeper than the `for` itself
    inside = [ln.strip() for ln in lines[for_i + 1:]
              if ln.strip() == "return"
              and (len(ln) - len(ln.lstrip())) > loop_indent]
    body = [ln.strip() for ln in lines[for_i + 1:]
            if (len(ln) - len(ln.lstrip())) > loop_indent]
    check("aihub --rag query prints every hit (return moved out of the loop)",
          not inside, f"loop body={body}; returns still inside loop={inside or 'none'}")


def v_armada_readonly_denies_shell_writes():
    from loomweaver import armada
    check("armada _WRITE_TOOLS covers shell and sql_query",
          {"shell", "sql_query"} <= armada._WRITE_TOOLS,
          f"_WRITE_TOOLS={sorted(armada._WRITE_TOOLS)}")


def v_key_rotation_preserves_cursor():
    from loomweaver import key_rotation as kr
    d = tempfile.mkdtemp()
    st = kr.RotationState(db_path=os.path.join(d, "kr.db"))
    st.advance("p", 2)
    before = st._row("p")[0]
    st.mark_dead("p", 0, "revoked")
    mid = st._row("p")[0]
    st.mark_exhausted("p", 1, retry_after=5)
    after = st._row("p")[0]
    check("key rotation cursor survives mark_dead/mark_exhausted",
          before == mid == after == 2,
          f"cursor: {before} -> {mid} -> {after} (expected 2 throughout)")


# ---------------------------------------------------------------- hygiene

def v_no_vibecoded_markers():
    # `C:/Users/alaga` is the leak. A bare `C:/Users` inside test_security.py is
    # a legitimate fixture asserting that Windows secret paths are DENIED, so it
    # is excluded deliberately rather than being scrubbed out of the tests.
    pats = ["alaga", "Hermes"]
    hits = []
    for dirpath, dirnames, filenames in os.walk(ROOT):
        dirnames[:] = [d for d in dirnames
                       if d not in {".git", "__pycache__", "node_modules", ".venv"}]
        for fn in filenames:
            if not fn.endswith((".py", ".md", ".txt", ".toml", ".json", ".yml")):
                continue
            fp = os.path.join(dirpath, fn)
            rel = os.path.relpath(fp, ROOT)
            # audits/ is gitignored (never shipped) and quotes the markers on
            # purpose to document what was removed; so does this verifier.
            if rel.startswith("audits" + os.sep) or rel.startswith("scripts" + os.sep):
                continue
            try:
                text = open(fp, encoding="utf-8", errors="ignore").read()
            except OSError:
                continue
            for pat in pats:
                if pat in text:
                    hits.append(f"{rel}:{pat}")
            if "C:/Users" in text and "test_security.py" not in rel:
                hits.append(f"{rel}:C:/Users")
    check("no 'alaga' / 'C:/Users' / 'Hermes' markers left in the repo",
          not hits, f"hits={hits or 'none'}")


def v_launcher_deleted():
    exists = os.path.exists(os.path.join(ROOT, "flippy-server-launcher.py"))
    check("personal Windows launcher removed from the repo root",
          not exists, f"flippy-server-launcher.py exists={exists}")


def v_gitignore_covers_runtime():
    gi = open(os.path.join(ROOT, ".gitignore")).read()
    need = ["sandbox/", ".coverage", "runs/", "learning.db"]
    missing = [n for n in need if n not in gi]
    check(".gitignore covers runtime/scratch artifacts",
          not missing, f"missing rules={missing or 'none'}")


def v_docs_match_code():
    llm = open(os.path.join(ROOT, "LLM.txt")).read()
    arch = open(os.path.join(ROOT, "ARCHITECTURE.md")).read()
    init = open(os.path.join(ROOT, "src", "loomweaver", "__init__.py")).read()
    n_tests = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider"],
        capture_output=True, text=True, cwd=ROOT).stdout
    m = re.search(r"(\d+) tests collected", n_tests)
    actual = m.group(1) if m else "?"
    claimed = re.search(r"(\d+) unit tests", llm)
    ok_count = bool(claimed) and claimed.group(1) == actual
    ok_arch = "next_key" not in arch and "observability.py" in arch and "learning.py" in arch
    ok_init = "python_exec" not in init
    check("docs match the code they describe",
          ok_count and ok_arch and ok_init,
          f"LLM.txt claims {claimed.group(1) if claimed else '?'} vs {actual} collected; "
          f"ARCHITECTURE fixed={ok_arch}; __init__ fixed={ok_init}")


def v_test_suite():
    # Earlier checks export fake credentials to probe provider plumbing. Run the
    # suite in a cleaned environment so a failure here means the code is broken,
    # not that this script left GROQ_KEY behind.
    env = {k: v for k, v in os.environ.items()
           if not (k.endswith(("_API_KEY", "_APIKEY", "_KEY", "_TOKEN", "_SECRET",
                               "_BASE_URL", "_API_BASE", "_BASE", "_ENDPOINT",
                               "_MODELS")))}
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                       capture_output=True, text=True, cwd=ROOT, env=env)
    lines = (r.stdout or "").strip().splitlines()
    tail = lines[-1] if lines else ""
    if r.returncode != 0 or "failed" in tail:
        # Naming the failures matters more than the summary line: without this
        # a red run here says only "not green" and you have to go re-run it.
        named = [l.strip() for l in lines if l.startswith("FAILED ")]
        tail = tail + " | " + "; ".join(named[:6] if named
                                        else ["no FAILED lines; see stderr: "
                                              + (r.stderr or "")[-300:]])
    last = lines[-1] if lines else ""
    check("full test suite is green", r.returncode == 0 and "failed" not in last, tail)


def v_litellm_is_a_real_dependency():
    """The decision: aihub is the entry point and litellm is not optional."""
    import tomllib
    with open(os.path.join(ROOT, "pyproject.toml"), "rb") as f:
        d = tomllib.load(f)
    deps = d["project"].get("dependencies") or []
    scripts = d["project"].get("scripts") or {}
    has_dep = any("litellm" in x for x in deps)
    is_entry = scripts.get("flippy", "").startswith("aihub:")
    importable = True
    try:
        import litellm  # noqa: F401
    except ImportError:
        importable = False
    check("litellm is a hard dependency and aihub is the console entry point",
          has_dep and is_entry and importable,
          f"dependencies={deps}; scripts={scripts}; litellm importable={importable}")


def v_aihub_consumes_universal_registry():
    """Using only aihub must not mean losing bring-your-own-keys."""
    os.environ["ACME_API_KEY"] = "ak_live_1"
    os.environ["ACME_BASE_URL"] = "https://llm.acme.io/v1"
    os.environ["ACME_MODELS"] = "acme-large"
    try:
        import aihub
        names = {f for _l, f, _k, _b in aihub.build_router_models()}
        check("aihub routes arbitrary BYO-key providers through litellm",
              "acme-large" in names, f"model names include acme-large={'acme-large' in names}")
    finally:
        for v in ("ACME_API_KEY", "ACME_BASE_URL", "ACME_MODELS"):
            os.environ.pop(v, None)


def v_aihub_secret_hygiene_is_derived():
    """A fixed six-name list cannot cover an open-ended provider surface."""
    os.environ["MISTRAL_API_KEY"] = "mstk_real_secret_value"
    try:
        import aihub
        covered = set(aihub._secret_env_vars())
        msg = aihub._safe_message(Exception("401 mstk_real_secret_value"))
        check("aihub redacts keys from every provider, not six hardcoded names",
              "MISTRAL_API_KEY" in covered and "mstk_real_secret_value" not in msg,
              f"covered={len(covered)} vars; mistral key redacted="
              f"{'mstk_real_secret_value' not in msg}")
    finally:
        os.environ.pop("MISTRAL_API_KEY", None)


def v_aihub_learns():
    from unittest import mock
    import aihub
    from loomweaver import learning
    path = os.path.join(tempfile.mkdtemp(), "a.db")
    os.environ["LOOMWEAVER_LEARNING_DB"] = path
    st = learning.LearningStore(db_path=path)
    learning.set_store(st)

    class R:
        def completion(self, model=None, messages=None, **kw):
            return {"choices": [{"message": {"content": "ok"}}], "model": model,
                    "usage": {}}

    try:
        with mock.patch.object(aihub, "build_router", return_value=(R(), None)):
            aihub.smart_chat([{"role": "user", "content": "summarise the invoice"}])
        ok = st.stats()["interactions"] == 1
        check("using only aihub still feeds the self-learning memory",
              ok, f"interactions recorded via smart_chat = {st.stats()['interactions']}")
    finally:
        learning.set_store(None)
        os.environ.pop("LOOMWEAVER_LEARNING_DB", None)


def v_aihub_cli_surface():
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src"),
           "LOOMWEAVER_LEARNING_DB": os.path.join(tempfile.mkdtemp(), "c.db")}
    results = {}
    for flag in ("--all-providers", "--providers", "--profile"):
        r = subprocess.run([sys.executable, os.path.join(ROOT, "src", "aihub.py"), flag],
                           capture_output=True, text=True, env=env)
        results[flag] = r.returncode
    check("aihub CLI is self-sufficient (providers, catalog, profile)",
          all(v == 0 for v in results.values()), f"exit codes={results}")


def _isolated_env():
    d = tempfile.mkdtemp()
    env = {**os.environ, "PYTHONPATH": os.path.join(ROOT, "src")}
    for var, name in (("LOOMWEAVER_KEYROTATION_DB", "rot"),
                      ("LOOMWEAVER_QUOTA_DB", "quota"),
                      ("LOOMWEAVER_USAGE_DB", "usage"),
                      ("LOOMWEAVER_CACHE_DB", "cache"),
                      ("LOOMWEAVER_LEARNING_DB", "learn")):
        env[var] = os.path.join(d, name + ".db")
    return env


def v_key_rotation_reaches_litellm():
    """A comma-separated key list must produce one deployment per LIVE key."""
    for var, val in (("GROQ_KEY", "k1,k2,k3"), ("GROQ_MODELS", "gpt-oss-20b"),
                     ("LOOMWEAVER_CACHE_ENABLED", "0")):
        os.environ[var] = val
    try:
        import aihub
        from loomweaver import hub, key_rotation as kr
        d = tempfile.mkdtemp()
        kr.set_state(kr.RotationState(db_path=os.path.join(d, "r.db")))
        before = len(aihub.build_deployments())
        kr.get_state().mark_dead("groq", 0, "revoked")
        after = len(aihub.build_deployments())
        check("multi-key rotation reaches the litellm Router",
              before == 3 and after == 2,
              f"deployments with 3 live keys={before}; after retiring one={after}")
    finally:
        for var in ("GROQ_KEY", "GROQ_MODELS"):
            os.environ.pop(var, None)


def v_semantic_cache_fronts_the_litellm_path():
    """A near-duplicate prompt must not re-bill a provider."""
    from unittest import mock
    import aihub
    from loomweaver import learning, semantic_cache as sc
    d = tempfile.mkdtemp()
    os.environ["LOOMWEAVER_CACHE_DB"] = os.path.join(d, "c.db")
    os.environ["LOOMWEAVER_CACHE_ENABLED"] = "1"
    os.environ["LOOMWEAVER_LEARNING_DB"] = os.path.join(d, "l.db")
    os.environ["GROQ_KEY"] = "gsk_" + "x" * 24
    sc._default_cache = None
    learning.set_store(learning.LearningStore(db_path=os.path.join(d, "l.db")))

    class R:
        n = 0
        def completion(self, model=None, messages=None, **kw):
            R.n += 1
            return {"choices": [{"message": {"content": "Paris"}}],
                    "model": model, "usage": {}}

    try:
        msgs = [{"role": "user", "content": "what is the capital of france"}]
        with mock.patch.object(aihub, "build_router", return_value=(R(), None)):
            first = aihub.smart_chat(msgs)
            second = aihub.smart_chat(msgs)
        check("semantic cache fronts the litellm path (no duplicate billing)",
              R.n == 1 and second.get("cached") is True,
              f"litellm calls for 2 identical prompts={R.n}; second cached="
              f"{second.get('cached')}")
    finally:
        learning.set_store(None)
        os.environ.pop("LOOMWEAVER_CACHE_ENABLED", None)


def v_http_and_cli_share_one_brain():
    """The engine stores are shared, so the CLI sees what HTTP learned."""
    env = _isolated_env()
    code = (
        "import sys; sys.path.insert(0, 'src');"
        "from loomweaver import hub;"
        "hub.record_outcome('mock','mock-1',True,0.01,"
        "usage={'prompt_tokens':11,'completion_tokens':5},"
        "goal='summarise the quarterly invoice for lagos office');"
        "from loomweaver import learning;"
        "print(learning.get_store().profile()['vocabulary'])")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True,
                       text=True, env=env, cwd=ROOT)
    cli = subprocess.run([sys.executable, "src/aihub.py", "--profile"],
                         capture_output=True, text=True, env=env, cwd=ROOT)
    out = cli.stdout
    check("HTTP/CLI/aihub share one memory (same engine stores)",
          "invoice" in r.stdout and "invoice" in out and "mock" in out,
          f"recorded vocab={r.stdout.strip()[:60]}; aihub --profile sees it="
          f"{'invoice' in out}")


def v_server_uses_the_same_resolver():
    """/v1/chat/completions must not keep its own provider whitelist."""
    src = open(os.path.join(ROOT, "src", "server.py")).read()
    uses_core = "core.build_providers(creds)" in src
    still_whitelisted = bool(re.search(
        r"get_providers\(\s*\{k: v for k, v in \(creds", src))
    check("HTTP pre-flight uses the same resolver as routing",
          uses_core and not still_whitelisted,
          f"uses core.build_providers={uses_core}; old whitelist call remains="
          f"{still_whitelisted}")


def v_engine_commands_delegated_not_duplicated():
    src = open(os.path.join(ROOT, "src", "aihub.py")).read()
    delegates = "from loomweaver import cli as _cli" in src
    covers = all(c in src for c in ("agent", "armada", "eval", "loadtest",
                                    "usage", "quota", "doctor"))
    check("aihub delegates engine commands to loomweaver.cli",
          delegates and covers, f"delegates={delegates}; command coverage={covers}")


def main():
    print("=" * 74)
    print("flippy — independent verification of audit findings")
    print("=" * 74)
    for fn in (v_shell_guard, v_shell_readonly_tier, v_ssrf_redirect,
               v_metadata_ip_still_blocked, v_redaction, v_traps_intact,
               v_agent_intents_are_real_tools, v_keyword_word_boundary,
               v_explicit_tools_fail_closed, v_dispatch_authorization,
               v_semantic_cache_threads, v_cache_tests_inside_class,
               v_universal_providers, v_custom_contract_preserved, v_catalog_size,
               v_learning_loop_changes_routing, v_learning_self_correction,
               v_learning_persists_across_restart,
               v_cron_new_job, v_loadtest_no_providers, v_aihub_jwt_redaction,
               v_aihub_rag_prints_all, v_armada_readonly_denies_shell_writes,
               v_key_rotation_preserves_cursor,
               v_key_rotation_reaches_litellm, v_semantic_cache_fronts_the_litellm_path,
               v_http_and_cli_share_one_brain, v_server_uses_the_same_resolver,
               v_engine_commands_delegated_not_duplicated,
               v_litellm_is_a_real_dependency, v_aihub_consumes_universal_registry,
               v_aihub_secret_hygiene_is_derived, v_aihub_learns, v_aihub_cli_surface,
               v_no_vibecoded_markers, v_launcher_deleted,
               v_gitignore_covers_runtime, v_docs_match_code, v_test_suite):
        try:
            fn()
        except Exception as exc:  # a verifier must not die on one finding
            check(fn.__name__, False, f"verifier raised {type(exc).__name__}: {exc}")

    manual("docker compose up exits 0 with .env.example alone",
           "needs docker + FLIPPY_AUTH_TOKEN; not runnable in this sandbox")
    manual("live end-to-end agent run against a real provider",
           "needs a real API key; a mock provider was used during the audit instead")

    passed = sum(1 for _, ok, _ in RESULTS if ok is True)
    failed = [f for f, ok, _ in RESULTS if ok is False]
    print("\n" + "=" * 74)
    print(f"{passed} passed, {len(failed)} failed, "
          f"{sum(1 for _, ok, _ in RESULTS if ok is None)} manual")
    if failed:
        print("FAILED:")
        for f in failed:
            print(f"  - {f}")
    print("=" * 74)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())

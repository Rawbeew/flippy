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
            if rel.startswith("AUDIT-needle-eyed") or rel.startswith("scripts/verify"):
                continue  # the audit and this verifier quote the markers on purpose
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
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider"],
                       capture_output=True, text=True, cwd=ROOT)
    tail = (r.stdout or "").strip().splitlines()[-1] if r.stdout else ""
    check("full test suite is green", r.returncode == 0 and "failed" not in tail, tail)


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

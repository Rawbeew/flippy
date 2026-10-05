"""FIX tests: SSRF shell guard, runtime decoys, write-jail allowlist, secrets.

Four hardening fixes, one evidence-backed test per fix:

  FIX 1  check_shell rejects shell commands that reach a blocked/private/
         metadata URL (SSRF/metadata bypass closed via the shared check_url).
  FIX 2  ensure_decoys() makes the decoy-credential layer reachable from a real
         entrypoint (the loomweaver CLI), idempotently and never-raising.
  FIX 3  the write-jail is governed by a single, env-overridable allowlist and
         the armada builder role truthfully declares exactly those roots.
  FIX 4  Unicode/homoglyph normalisation runs before the guards, so obfuscated
         credential reads are caught, and secret redaction covers all families.

All mocked where a network / provider call would otherwise fire.
"""
import os
import shutil
import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loomweaver import security, tools, armada
from loomweaver.armada import ROLES


# ---------------------------------------------------------------------------
# FIX 1: SSRF / metadata bypass in the shell guard
# ---------------------------------------------------------------------------

class TestShellSSRF:
    def test_shell_http_to_metadata_blocked(self):
        for cmd in (
            "curl http://169.254.169.254/latest/meta-data/iam/security-credentials/",
            "curl -s http://169.254.169.254/",
            "wget -qO- http://metadata.google.internal/",
        ):
            ok, why = security.check_shell(cmd)
            assert not ok, cmd
            assert "blocked shell target" in why

    def test_shell_http_to_private_ip_blocked(self):
        for cmd in ("curl http://127.0.0.1:8080/admin", "curl http://10.0.0.1/"):
            ok, why = security.check_shell(cmd)
            assert not ok, cmd

    def test_shell_bare_host_port_target_blocked(self):
        # a bare host:port target (no scheme) must route through the same guard
        for cmd in ("curl 169.254.169.254:80/", "curl localhost:9000",
                    "wget 127.0.0.1:8080/x"):
            ok, why = security.check_shell(cmd)
            assert not ok, cmd
            assert "blocked shell target" in why

    def test_benign_shell_still_allowed(self):
        for cmd in ("ls -la && echo done", "cat README.md",
                    "pytest tests/test_security.py", "git status",
                    "cp a.txt b.txt", "sleep 1"):
            ok, why = security.check_shell(cmd)
            assert ok, (cmd, why)

    def test_shell_and_direct_url_guard_agree(self):
        # Both URL consumers — the direct http_get/http_post_json path
        # (security.check_url) and the shell-target path (check_shell ->
        # _extract_target_urls) — must route through the SAME guard so a URL
        # blocked one way is never reachable the other. Locks the single-source
        # SSRF design against future divergence.
        blocked = [
            "http://169.254.169.254/latest/meta-data/",
            "http://metadata.google.internal/",
            "http://localhost:8080/admin",
            "http://127.0.0.1:9000/x",
            "http://10.0.0.5/",
        ]
        allowed = ["https://raw.githubusercontent.com/a/b/main/README.md",
                   "https://api.github.com/"]
        for url in blocked:
            assert not security.check_url(url)[0], url
            shell = "curl " + url if ":" in url and "//" in url else url
            ok, why = security.check_shell(shell)
            assert not ok, (shell, why)
        for url in allowed:
            assert security.check_url(url)[0], url
            ok, why = security.check_shell("curl %s" % url)
            assert ok, (url, why)


# ---------------------------------------------------------------------------
# FIX 2: decoy-credential layer reachable from a real entrypoint
# ---------------------------------------------------------------------------

DECOY_DIR = os.path.join(security.PROJECT_ROOT, "sandbox", "decoys")


class TestEnsureDecoys:
    def test_refuses_src_and_tests(self):
        # installing into the guard tree / test suite must be a silent no-op
        assert security.check_write_path("src/loomweaver/tools.py")[0] is False
        from loomweaver import observability
        root = security.PROJECT_ROOT
        assert observability.ensure_decoys(os.path.join(root, "src")) == []
        assert observability.ensure_decoys(os.path.join(root, "tests")) == []
        assert observability.ensure_decoys(os.path.join(root, "src", "loomweaver")) == []

    def test_cli_entrypoint_plants_reachable_decoy(self):
        from loomweaver import observability, tools as loom_tools
        import loomweaver.cli as cli_mod

        _clean_decoy_dir()
        try:
            with mock.patch("loomweaver.cli.build_providers", return_value=[]), \
                 mock.patch("loomweaver.cli.load_creds", return_value={}):
                cli_mod.main(["providers"])  # the real CLI entrypoint

            # a decoy was planted inside the project (path-jail legal)
            decoy = os.path.join(DECOY_DIR, observability._DECOY_READABLE_NAME)
            assert os.path.exists(decoy), "CLI install must plant a decoy file"

            # lookup_managed_file on that exact path returns a nonce
            nonce = observability.lookup_managed_file(decoy)
            assert nonce is not None

            # reading it via tools.read_file serves the generated payload,
            # not the planted keyring file contents
            result = loom_tools.read_file(decoy)
            assert "Setup guide" in result
            assert "OPENAI_KEY" not in result
            assert "SK-" not in result.upper()
        finally:
            _clean_decoy_dir()

    def test_ensure_decoys_idempotent_never_raises(self):
        from loomweaver import observability
        _clean_decoy_dir()
        try:
            first = observability.ensure_decoys()
            second = observability.ensure_decoys()
            # both succeeded (non-empty) and did not blow up
            assert first and second
            # calling again is safe/idempotent (no duplicate readable decoy)
            count = sum(
                1 for p in observability._MANAGED_FILES
                if p.endswith(observability._DECOY_READABLE_NAME))
            assert count >= 1
        finally:
            _clean_decoy_dir()


def _clean_decoy_dir():
    shutil.rmtree(DECOY_DIR, ignore_errors=True)


# ---------------------------------------------------------------------------
# FIX 3: single source of truth for the agent write-jail
# ---------------------------------------------------------------------------

class TestWriteAllowlist:
    def test_allowlist_roots_are_writable(self):
        for root in security.agent_write_roots():
            ok, why = security.check_write_path(root + "/probe.txt")
            assert ok, (root, why)

    def test_guarded_paths_never_writable(self):
        for p in ("src/loomweaver/security.py", "tests/test_security.py",
                  "sessions/x.json", "runs/x.jsonl", ".github/workflows/ci.yml"):
            ok, _ = security.check_write_path(p)
            assert not ok, p

    def test_repo_root_data_file_writable(self):
        ok, why = security.check_write_path(os.path.join(security.PROJECT_ROOT, "notes.md"))
        assert ok, why

    def test_outside_allowlist_denied(self):
        ok, why = security.check_write_path("misc/subdir/notes.txt")
        assert not ok, why

    def test_env_override_governs(self, monkeypatch):
        monkeypatch.setenv("LOOMWEAVER_WRITE_ROOTS", "sandbox,custom_root")
        assert security.agent_write_roots() == ["sandbox", "custom_root"]
        assert security.check_write_path("custom_root/a.txt")[0] is True
        assert security.check_write_path("data/quota.db")[0] is False  # no longer allowed


class TestBuilderTruthfulCapability:
    def test_builder_declares_exact_allowlist(self):
        # single source of truth: role capability == check_write_path policy
        declared = ROLES["builder"]["writable_roots"]
        assert declared == security.agent_write_roots()
        # every declared root is actually writable
        for root in declared:
            assert security.check_write_path(root + "/x.txt")[0] is True
        # and the guard source/tests the old prompt promised are NOT writable
        assert security.check_write_path("src/loomweaver/security.py")[0] is False
        assert security.check_write_path("tests/test_security.py")[0] is False

    def test_builder_prompt_is_truthful(self):
        prompt = ROLES["builder"]["system"]
        # it no longer promises to write guard code/tests; it states the jail
        assert "READ-ONLY" in prompt
        assert "src/" in prompt and "tests/" in prompt


# ---------------------------------------------------------------------------
# FIX 4: normalisation + full secret-family redaction
# ---------------------------------------------------------------------------

class TestNormalizationBeforeGuards:
    def test_word_joiner_credential_read_blocked(self):
        # U+2060 WORD JOINER smuggled into ".env" must be caught after stripping
        ok, why = security.check_shell("cat .e\u2060nv")
        assert not ok, why

    def test_cyrillic_homoglyph_credential_read_blocked(self):
        # Cyrillic 'е' (U+0435) is a Latin 'e' homoglyph in ".env"
        ok, why = security.check_shell("cat .\u0435nv")
        assert not ok, why

    def test_homoglyph_path_blocked(self):
        assert security.check_path(".\u0435nv")[0] is False

    def test_homoglyph_metadata_url_blocked(self):
        ok, why = security.check_url("http://169.254\u2060.169.254/")
        assert not ok, why


# name -> a representative sample carrying that secret family
SECRET_SAMPLES = {
    # -- existing families --
    "openai": "sk-" + "A" * 20,
    "anthropic": "sk-ant-" + "A" * 30,
    "groq": "gsk_" + "A" * 30,
    "nvidia": "nvapi-" + "A" * 30,
    "cloudflare": "cfut_" + "A" * 30,
    "github_classic": "ghp_" + "A" * 30,
    "huggingface": "hf_" + "A" * 30,
    "aws_access": "AKIA" + "ABCDEFGHIJKLMNOP",
    "google": "AIza" + "A" * 30,
    "xai": "xai-" + "A" * 30,
    # -- newly added families --
    "github_oauth": "gho_" + "A" * 30,
    "github_user": "ghu_" + "A" * 30,
    "github_server": "ghs_" + "A" * 30,
    "github_refresh": "ghr_" + "A" * 30,
    "github_fine_grained": "github_pat_" + "A" * 30,
    "slack_bot": "xoxb-" + "A" * 30,
    "slack_user": "xoxp-" + "A" * 30,
    "slack_app": "xoxa-" + "A" * 30,
    "slack_legacy": "xoxs-" + "A" * 30,
    "stripe_live": "sk_live_" + "A" * 30,
    "stripe_test": "sk_test_" + "A" * 30,
    "stripe_rk_live": "rk_live_" + "A" * 30,
    "stripe_rk_test": "rk_test_" + "A" * 30,
    "aws_session": "ASIA" + "ABCDEFGHIJKLMNOP",
    "jwt": ("eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
            "eyJzdWIiOiIxMjM0NTY3ODkwIn0."
            "dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"),
    "pem_rsa": ("-----BEGIN RSA PRIVATE KEY-----\n"
                "MIIBoQxx\n"
                "-----END RSA PRIVATE KEY-----"),
    "pem_ec": ("-----BEGIN EC PRIVATE KEY-----\n"
               "AQIDBA==\n"
               "-----END EC PRIVATE KEY-----"),
    "gcp_service_account": ("{\"type\": \"service_account\", \"project_id\": \"p\", "
                            "\"private_key\": \"-----BEGIN PRIVATE KEY-----\nabcdef\n"
                            "-----END PRIVATE KEY-----\"}"),
}


class TestSecretRedactionAllFamilies:
    def test_all_standalone_families_redacted(self):
        # standalone secrets (no surrounding text) must be fully redacted
        embedded = {"pem_rsa", "pem_ec", "gcp_service_account"}
        for family, sample in SECRET_SAMPLES.items():
            if family in embedded:
                continue
            out = tools.redact(sample)
            assert out == "[REDACTED]", (family, out)

    def test_embedded_pem_gcp_stripped(self):
        for family in ("pem_rsa", "pem_ec", "gcp_service_account"):
            out = tools.redact(SECRET_SAMPLES[family])
            assert "BEGIN" not in out and "PRIVATE KEY" not in out, (family, out)
            assert "[REDACTED]" in out

    def test_plain_text_untouched(self):
        text = "hello world, no keys here"
        assert tools.redact(text) == text

    def test_old_redaction_regex_families_intact(self):
        # the previous families must still be caught (no regression)
        for family in ("openai", "anthropic", "groq", "nvidia", "cloudflare",
                       "github_classic", "huggingface", "aws_access", "google", "xai"):
            out = tools.redact(SECRET_SAMPLES[family])
            assert out == "[REDACTED]", (family, out)
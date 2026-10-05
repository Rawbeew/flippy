"""B4-2: loomweaver.cli.doctor — offline config validation (keys + writable DB paths).

No network calls, no crashes. Key checks: WARN when unset, OK when a
well-formed key is present, FAIL on a malformed key. DB-path checks report
OK/FAIL based on writability.
"""
import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from loomweaver import cli


def _run(creds=None, env=None):
    """Invoke doctor with the supplied creds/sandbox env; returns statuses."""
    return [r["status"] for r in cli.doctor(creds=creds or {}, env=env or {})]


def _tmp_env():
    """env dict pointing all DB paths into a writable tmp dir."""
    d = tempfile.mkdtemp()
    return {
        "LOOMWEAVER_QUOTA_DB": os.path.join(d, "quota.db"),
        "LOOMWEAVER_CACHE_DB": os.path.join(d, "cache.sqlite3"),
        "LOOMWEAVER_USAGE_DB": os.path.join(d, "usage.db"),
        "CLOUDFLARE_ACCOUNT_ID": "",
    }


class TestDoctor:
    def test_no_creds_warns_and_never_crashes(self):
        # no keys at all -> every provider key check is WARN, none FAIL, and
        # the DB checks are OK because the paths are writable
        statuses = _run(env=_tmp_env())
        assert statuses, "doctor must always produce results"
        assert "FAIL" not in statuses, f"no keys must not FAIL: {statuses}"
        # at least the provider key checks show WARN
        assert "WARN" in statuses

    def test_good_key_is_ok(self):
        creds = {"OPENROUTER_KEY": "sk-or-" + "a" * 32,
                 "CLOUDFLARE_ACCOUNT_ID": "acct123"}
        statuses = _run(creds=creds, env={**creds, **{
            "LOOMWEAVER_QUOTA_DB": os.path.join(tempfile.mkdtemp(), "q.db"),
            "LOOMWEAVER_CACHE_DB": os.path.join(tempfile.mkdtemp(), "c.sq3"),
            "LOOMWEAVER_USAGE_DB": os.path.join(tempfile.mkdtemp(), "u.db"),}})
        # openrouter present+valid -> OK; the key check for it is OK
        assert "OK" in statuses
        # no FAIL for any check that had a valid input
        assert "FAIL" not in statuses

    def test_malformed_key_is_fail(self):
        # a groq key that lacks the required 'gsk_' prefix -> FAIL on that check
        creds = {"GROQ_KEY": "not-a-real-groq-key"}
        env = _tmp_env()
        env["GROQ_KEY"] = creds["GROQ_KEY"]
        results = cli.doctor(creds=creds, env=env)
        groq = [r for r in results if r["check"] == "provider.groq.key"]
        assert groq and groq[0]["status"] == "FAIL", f"got: {groq}"

    def test_no_crash_with_empty_env(self):
        # a totally empty env (dict, not os.environ) must still run cleanly
        statuses = cli.doctor(creds={}, env={})
        assert statuses is not None

    def test_doctor_exit_informative(self):
        # _print_doctor_results renders every check cleanly (no crash)
        import io
        from contextlib import redirect_stdout
        results = cli.doctor(creds={}, env=_tmp_env())
        buf = io.StringIO()
        with redirect_stdout(buf):
            cli._print_doctor_results(results)
        out = buf.getvalue()
        assert "[ OK ]" in out or "[WARN]" in out or "[FAIL]" in out
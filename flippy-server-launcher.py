"""flippy-server-launcher.py — load .env, then run flippy's OpenAI-compatible server.

Used by the Windows scheduled task so the persistent endpoint:
  1. Reads GROQ_KEY (+ other provider keys, FLIPPY_AUTH_TOKEN, PORT, HOST) from the
     repo's .env, which the server does NOT auto-load.
  2. Runs the server on a STABLE system Python (C:/Python314) rather than Hermes' venv,
     so the endpoint survives Hermes updates.
  3. Binds 127.0.0.1:8080 by default (set HOST/PORT in .env to expose it).

Files this reads:
  - C:/Users/alaga/ghwork/flippy/.env     (keys, never committed)
  - C:/Users/alaga/ghwork/flippy/src      (on sys.path)
"""
import os, sys, subprocess

REPO = r"C:/Users/alaga/ghwork/flippy"
ENV_PATH = os.path.join(REPO, ".env")
SERVER = os.path.join(REPO, "src", "server.py")

# --- 1. load .env into os.environ (only vars not already set) ---
loaded = []
if os.path.exists(ENV_PATH):
    with open(ENV_PATH, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, v = line.split("=", 1)
            k, v = k.strip(), v.strip()
            # strip optional surrounding quotes from the value
            if len(v) >= 2 and v[0] == v[-1] and v[0] in ('"', "'"):
                v = v[1:-1]
            os.environ.setdefault(k, v)
            loaded.append(k)
else:
    print("[launcher] WARN: no .env at", ENV_PATH, "— running with current env only", flush=True)

# --- 2. choose python: prefer a stable system interpreter over Hermes' venv ---
candidates = [
    os.environ.get("FLIPPY_PYTHON"),      # explicit override
    r"C:/Python314/python.exe",           # system python 3.14 (stdlib-only fine)
    r"C:/Program Files/Python312/python.exe",
]
interp = next((p for p in candidates if p and os.path.exists(p)), sys.executable)
print(f"[launcher] python: {interp}", flush=True)
print(f"[launcher] keys loaded from .env: {len(loaded)} var(s): "
      f"{[k for k in loaded if not any(x in k for x in ['KEY','TOKEN','SECRET'])]} "
      f"(secrets hidden)", flush=True)

# --- 3. build a clean env for the child (PYTHONPATH=src) ---
child_env = dict(os.environ)
child_env["PYTHONPATH"] = os.path.join(REPO, "src") + os.pathsep + child_env.get("PYTHONPATH", "")

# --- 4. run the server (inherits stdio so logs go to the task's stdout/stderr) ---
print(f"[launcher] starting server: {interp} {SERVER}", flush=True)
proc = subprocess.run([interp, SERVER], env=child_env)
sys.exit(proc.returncode)
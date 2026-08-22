# Deploying flippy

The container is stdlib-only (no pip installs), ~50 MB, and exposes:

| Route | Method | What |
|---|---|---|
| `/health` | GET | 200 `{"ok": true}` |
| `/metrics` | GET | Prometheus text, `flippy_*` series |
| `/v1/chat/completions` | POST | OpenAI-shaped; failover-routes across configured providers |

Keys come from environment variables (`OPENROUTER_KEY`, `GROQ_KEY`, `NVIDIA_KEY`,
`FREEINFERENCE_KEY`, `CLOUDFLARE_TOKEN` + `CLOUDFLARE_ACCOUNT_ID`). Any subset works —
unconfigured providers are skipped. **Never bake keys into the image.**

Local smoke test with fake keys:

```bash
docker build -t flippy .
docker run -d -p 8080:8080 --env-file .env.test --name flippy-test flippy
curl -s localhost:8080/health            # {"ok": true}
curl -s localhost:8080/metrics           # flippy_http_requests_total ...
curl -s localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"messages":[{"role":"user","content":"hi"}]}'
# fake key → 502 JSON provider_error, NOT a crash:
# {"error":{"message":"...401...","type":"provider_error"}}
docker rm -f flippy-test
```

---

## (a) Fly.io — free tier, $0

```bash
brew install flyctl && fly auth signup        # free: no card needed
cd <repo root>                                # fly.toml is included
fly launch --no-deploy --copy-config
fly secrets set OPENROUTER_KEY=sk-or-...      # real keys as secrets, never in git
fly deploy
curl https://flippy-router.fly.dev/health     # {"ok": true}
```

Cost notes: the included `fly.toml` uses a shared-cpu-1x / 256 MB machine with
`auto_stop_machines = "stop"` and `min_machines_running = 0` — it scales to zero when
idle and stays inside Fly's free allowance (up to 3 tiny VMs). Expect **$0/month** for
low traffic.

## (b) Any VPS — Docker Compose

```bash
git clone https://github.com/Rawbeew/flippy && cd flippy
cp .env.example .env && nano .env             # paste your keys
docker compose up -d --build
curl http://localhost:8080/health
```

Put Caddy or nginx in front for TLS:

```
# Caddyfile — automatic Let's Encrypt
api.example.com {
    reverse_proxy localhost:8080
}
```

Cost notes: runs happily on any $0-tier VPS (Oracle Cloud Always Free ARM, a home box,
or an existing server). RAM footprint is tens of megabytes.

## (c) Google Cloud Run

```bash
gcloud run deploy flippy \
  --source . \
  --region us-central1 \
  --allow-unauthenticated \
  --port 8080 \
  --memory 256Mi \
  --cpu 1 \
  --min-instances 0 \
  --set-env-vars PORT=8080 \
  --update-secrets OPENROUTER_KEY=openrouter-key:latest
# (create the secret first: echo -n "sk-or-..." | gcloud secrets create openrouter-key --data-file=-)
curl https://flippy-xxxx-a.run.app/health
```

Cost notes: Cloud Run's always-free tier includes 2M requests + 360k GB-seconds/month.
With `--min-instances 0` you pay nothing when idle → effectively **$0** at hobby scale.

---

## Which one?

- **Fastest to a URL:** Fly.io (single command).
- **Cheapest at any scale / full control:** VPS + compose.
- **Autoscaling + per-request billing:** Cloud Run.

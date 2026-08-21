"""loadtest.py — inference harness: concurrency, TTFT/TPS, provider comparison."""
import concurrent.futures
import json
import statistics
import time

from .core import RunLog, build_providers, chat, load_creds


def _one_request(prov, prompt, max_tokens):
    t0 = time.time()
    r = chat(prov, [{"role": "user", "content": prompt}], max_tokens=max_tokens)
    lat = time.time() - t0
    n_out = len((r.get("text") or "").split())
    return {"ok": r.get("ok"), "latency": round(lat, 3),
            "tps": round(n_out / lat, 1) if (r.get("ok") and lat > 0) else 0,
            "error": (r.get("error") or "")[:120] if not r.get("ok") else None}


def run(provider=None, concurrency=4, requests=8, prompt="Write a 100-word story about a robot learning to paint.", max_tokens=300, creds=None, runs_dir=None):
    """Fire `requests` total at a provider with `concurrency` parallel workers."""
    creds = creds or load_creds()
    provs = build_providers(creds)
    prov = next((p for p in provs if p["name"] == provider), provs[0])
    runlog = RunLog(runs_dir)
    runlog.emit({"type": "loadtest_start", "provider": prov["name"],
                 "concurrency": concurrency, "requests": requests})

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futs = [ex.submit(_one_request, prov, prompt, max_tokens) for _ in range(requests)]
        for f in concurrent.futures.as_completed(futs):
            res = f.result()
            results.append(res)
            runlog.emit({"type": "loadtest_req", "provider": prov["name"], **res})

    oks = [r for r in results if r["ok"]]
    summary = {
        "provider": prov["name"], "requests": requests, "concurrency": concurrency,
        "success": len(oks), "fail": len(results) - len(oks),
        "latency_p50": round(statistics.median([r["latency"] for r in oks]), 2) if oks else None,
        "latency_max": round(max(r["latency"] for r in oks), 2) if oks else None,
        "throughput_rps": round(len(oks) / sum(r["latency"] for r in oks) * concurrency, 2) if oks else 0,
        "avg_tps": round(statistics.mean([r["tps"] for r in oks]), 1) if oks else 0,
    }
    runlog.emit({"type": "loadtest_summary", **summary})
    return summary


def compare_all(concurrency=2, requests=2, max_tokens=100):
    """Quick sweep: one small request per provider, ranked by latency."""
    creds = load_creds()
    rows = []
    for prov in build_providers(creds):
        r = _one_request(prov, "Say 'ready' and nothing else.", 20)
        rows.append({"provider": prov["name"], **r})
    return sorted(rows, key=lambda x: x["latency"])


def ttft_sweep(max_tokens=200):
    """Streaming sweep: TTFT + tokens/sec per provider, ranked by TTFT."""
    from .stream import stream_chat
    creds = load_creds()
    rows = []
    for prov in build_providers(creds):
        r = stream_chat(prov, [{"role": "user", "content":
            "Count from 1 to 50, space-separated."}], max_tokens=max_tokens)
        rows.append({"provider": prov["name"], **{k: v for k, v in r.items() if k != "text"}})
    return sorted(rows, key=lambda x: x.get("ttft", 99))

#!/usr/bin/env python3
"""Fan-out self-contention sweep for a shared LLM inference server.

Multi-agent systems fan out concurrent calls to one inference server. This
sends K concurrent identical requests and measures per-call latency and
aggregate throughput as K grows, exposing admission-control / batching limits
(e.g. a low vLLM --max-num-seqs). No secrets are stored; endpoint, model and
key come from CLI/env. See docs/MOLEG_COT_AND_CONTENTION_20260904.md.
"""
from __future__ import annotations
import argparse, asyncio, concurrent.futures, json, os, statistics, time, urllib.request


def _one(url, model, key, prompt, max_tokens, timeout):
    body = json.dumps({"model": model,
                       "messages": [{"role": "user", "content": prompt}],
                       "temperature": 0, "max_tokens": max_tokens}).encode()
    headers = {"Content-Type": "application/json"}
    if key:
        headers["Authorization"] = "Bearer " + key
    req = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body, headers=headers)
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        r.read()
    return (time.perf_counter() - t) * 1000.0


async def _fire(loop, fn, k):
    return await asyncio.gather(*[loop.run_in_executor(None, fn) for _ in range(k)])


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible base url incl scheme+host+port")
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=os.environ.get("ORCH_API_KEY", ""))
    ap.add_argument("--prompt", default="독일에서 회사를 설립하려는데 관련 법령과 절차를 설명해줘.")
    ap.add_argument("--max-tokens", type=int, default=40)
    ap.add_argument("--concurrencies", default="1,2,4,8,16")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=180.0)
    args = ap.parse_args()

    Ks = [int(x) for x in args.concurrencies.split(",")]
    ex = concurrent.futures.ThreadPoolExecutor(max_workers=max(Ks) + 4)
    loop = asyncio.get_event_loop(); loop.set_default_executor(ex)

    def fn():
        return _one(args.base_url, args.model, args.api_key, args.prompt, args.max_tokens, args.timeout)

    print(f"=== {args.model} concurrency sweep (max_tokens={args.max_tokens}, rounds={args.rounds}) ===")
    print(f"{'K':>3s} {'percall_mean':>12s} {'p95':>8s} {'throughput/s':>12s} {'inflation':>9s}")
    base = None
    for K in Ks:
        allc, walls = [], []
        for _ in range(args.rounds):
            t0 = time.perf_counter()
            allc += await _fire(loop, fn, K)
            walls.append((time.perf_counter() - t0) * 1000.0)
            await asyncio.sleep(0.5)
        allc.sort(); n = len(allc); mean = sum(allc) / n
        p95 = allc[min(n - 1, int(0.95 * n))]
        thr = K / (statistics.mean(walls) / 1000.0)
        base = base or mean
        print(f"{K:3d} {mean:10.0f}ms {p95:6.0f}ms {thr:10.1f}  {mean / base:7.2f}x")


if __name__ == "__main__":
    asyncio.run(main())

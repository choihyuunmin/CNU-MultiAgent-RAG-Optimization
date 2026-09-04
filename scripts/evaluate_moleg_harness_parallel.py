#!/usr/bin/env python3
"""Harness-level acceleration: run independent analysis calls concurrently.

Classify and preparation both read only the raw user query, so they are
independent. Running them in parallel instead of sequentially preserves outputs
exactly (accuracy 1.000) while cutting wall-clock latency. This measures the two
schedules and verifies output identity. No prompts, questions, or secrets are
stored here; they come from --rag-root and CLI/env.
See docs/MOLEG_ACCELERATION_ABLATION_20260904.md.
"""
from __future__ import annotations
import argparse, asyncio, importlib.util, os, sys, time, types


def _load_env(root):
    p = os.path.join(root, ".env")
    if os.path.exists(p):
        for ln in open(p, encoding="utf-8"):
            ln = ln.strip()
            if ln and not ln.startswith("#") and "=" in ln:
                k, v = ln.split("=", 1); os.environ.setdefault(k.strip(), v.strip())


def _classify_system(root):
    for n in ("agent", "agent.guardrail"):
        sys.modules[n] = types.ModuleType(n)
    sys.modules["agent.guardrail"].check_input_async = lambda *a, **k: None
    spec = importlib.util.spec_from_file_location(
        "orch_standalone", os.path.join(root, "src", "core", "agent_orchestrator", "orchestrator.py"))
    m = importlib.util.module_from_spec(spec); sys.modules["orch_standalone"] = m
    spec.loader.exec_module(m); return m._CLASSIFY_SYSTEM


def _stat(v):
    v = sorted(v); n = len(v); return sum(v) / n, v[n // 2], v[min(n - 1, int(0.95 * n))]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag-root", required=True)
    ap.add_argument("--orchestrator-url", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--api-key", default=os.environ.get("ORCH_API_KEY", "x"))
    ap.add_argument("--limit", type=int, default=15)
    args = ap.parse_args()
    root = os.path.abspath(args.rag_root); _load_env(root)
    sys.path.insert(0, os.path.join(root, "src")); sys.path.insert(0, os.path.join(root, "scripts"))
    from config.prompts import get_preparation_system_prompt
    from load_test_questions import build_test_queries
    from domain.country.service import get_available_countries
    from openai import AsyncOpenAI
    CLASSIFY = _classify_system(root)
    PREP = get_preparation_system_prompt(get_available_countries())[0]["content"]
    cli = AsyncOpenAI(base_url=args.orchestrator_url, api_key=args.api_key, timeout=120)

    async def call(system, user):
        r = await cli.chat.completions.create(model=args.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0, response_format={"type": "json_object"})
        return r.choices[0].message.content or ""

    async def seq(q):
        t = time.perf_counter(); c = await call(CLASSIFY, q); p = await call(PREP, q)
        return (time.perf_counter() - t) * 1000, c, p

    async def par(q):
        t = time.perf_counter(); c, p = await asyncio.gather(call(CLASSIFY, q), call(PREP, q))
        return (time.perf_counter() - t) * 1000, c, p

    qs = build_test_queries()[: args.limit]
    await seq(qs[0]); await par(qs[0])
    s_ms, p_ms, ident = [], [], 0
    for q in qs:
        ds, cs, ps = await seq(q); dp, cp, pp = await par(q)
        s_ms.append(ds); p_ms.append(dp); ident += (cs == cp and ps == pp)
    S, P = _stat(s_ms), _stat(p_ms)
    print(f"=== sequential vs parallel (classify + preparation), N={len(qs)} ===")
    print(f"SEQUENTIAL mean={S[0]/1000:.2f}s p50={S[1]/1000:.2f}s p95={S[2]/1000:.2f}s")
    print(f"PARALLEL   mean={P[0]/1000:.2f}s p50={P[1]/1000:.2f}s p95={P[2]/1000:.2f}s")
    print(f"reduction {100*(1-P[0]/S[0]):.1f}%  ({(S[0]-P[0])/1000:.2f}s/req, {S[0]/P[0]:.2f}x)")
    print(f"output identical: {ident}/{len(qs)} (accuracy preserved exactly)" if ident == len(qs) else f"output identical: {ident}/{len(qs)}")


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Measure merging the orchestrator classify + preparation calls into one.

Diagnosis (see docs/MOLEG_ORCHESTRATOR_LATENCY_20260904.md): the orchestrator
LLM dominates multi-agent latency. On the law_search path it runs classify and
preparation as two sequential calls over the same user query. This script
measures a single merged call against the current two-call flow and checks that
the merged output preserves task / country / guardrail / keyword decisions.

No prompts, questions, country lists, endpoints, or credentials are stored in
this repository. All of those are reconstructed at run time from the live
service source given by --rag-root, and the model endpoint/key come from CLI or
environment. Run this only against a system you are authorised to test.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import os
import statistics
import sys
import time
import types
from typing import Any, Dict, List, Tuple


def _load_env(rag_root: str) -> None:
    path = os.path.join(rag_root, ".env")
    if not os.path.exists(path):
        return
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        k, v = line.split("=", 1)
        os.environ.setdefault(k.strip(), v.strip())


def _extract_classify_system(rag_root: str) -> str:
    """Load orchestrator._CLASSIFY_SYSTEM without triggering package imports."""
    for name in ("agent", "agent.guardrail"):
        stub = types.ModuleType(name)
        sys.modules[name] = stub
    sys.modules["agent.guardrail"].check_input_async = lambda *a, **k: None
    src = os.path.join(rag_root, "src", "core", "agent_orchestrator", "orchestrator.py")
    spec = importlib.util.spec_from_file_location("orch_standalone", src)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["orch_standalone"] = mod
    spec.loader.exec_module(mod)
    return mod._CLASSIFY_SYSTEM


def _merged_system(classify_sys: str, prep_sys: str) -> str:
    return (
        "당신은 세계법제정보 지능형 검색시스템의 마스터 라우터 겸 쿼리 분석기입니다.\n"
        "아래 [PART A]로 task를 분류하고, 동시에 [PART B]로 쿼리 분석 필드를 추출하여, "
        "하나의 JSON 객체로만 출력하세요.\n\n"
        "===== [PART A] 라우팅 규칙 =====\n" + classify_sys +
        "\n===== [PART B] 쿼리 분석 규칙 =====\n" + prep_sys +
        '\n===== 최종 출력 형식 (이 형식의 JSON 하나만) =====\n'
        '{"task":"<task_id>","bad_word_detected":false,"pii_detected":false,'
        '"transformed_query":"","keywords_from_original":[],"keywords_from_transformed":[],'
        '"country":"","specific_article_number":0,"law_title_search_hint":""}\n'
    )


def _parse_json(text: str) -> Dict[str, Any]:
    try:
        i, j = text.find("{"), text.rfind("}")
        return json.loads(text[i:j + 1]) if i >= 0 else {}
    except Exception:
        return {}


def _jaccard(a: List[str], b: List[str]) -> float:
    sa = {x.strip() for x in (a or []) if x}
    sb = {x.strip() for x in (b or []) if x}
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def _stats(v: List[float]) -> Tuple[float, float, float, float]:
    v = sorted(v)
    n = len(v)
    return (sum(v) / n, v[n // 2], v[min(n - 1, int(0.95 * n))], v[-1])


async def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag-root", required=True, help="live 2025-moleg-rag source root")
    ap.add_argument("--orchestrator-url", required=True, help="OpenAI-compatible base url, e.g. http://host:8000/v1")
    ap.add_argument("--model", required=True, help="served orchestrator model id")
    ap.add_argument("--api-key", default=os.environ.get("ORCH_API_KEY", ""))
    ap.add_argument("--limit", type=int, default=50)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--output")
    args = ap.parse_args()

    root = os.path.abspath(args.rag_root)
    _load_env(root)
    sys.path.insert(0, os.path.join(root, "src"))
    sys.path.insert(0, os.path.join(root, "scripts"))

    from load_test_questions import build_test_queries
    from config.prompts import get_preparation_system_prompt
    from domain.country.service import get_available_countries
    from openai import AsyncOpenAI

    classify_sys = _extract_classify_system(root)
    try:
        countries = get_available_countries()
    except Exception:
        countries = []
    prep_sys = get_preparation_system_prompt(countries)[0]["content"]
    merged_sys = _merged_system(classify_sys, prep_sys)

    client = AsyncOpenAI(base_url=args.orchestrator_url, api_key=args.api_key or "x", timeout=args.timeout)

    async def call(system: str, user: str) -> Tuple[float, str, Any]:
        t = time.perf_counter()
        r = await client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0,
            response_format={"type": "json_object"},
        )
        dt = (time.perf_counter() - t) * 1000.0
        msg = r.choices[0].message.content or ""
        usage = getattr(r, "usage", None)
        ct = getattr(usage, "completion_tokens", None) if usage else None
        return dt, msg, ct

    questions = build_test_queries()[: args.limit]
    # warm-up (excluded)
    await call(classify_sys, questions[0])
    await call(prep_sys, questions[0])
    await call(merged_sys, questions[0])

    base_ms: List[float] = []
    cand_ms: List[float] = []
    bct: List[int] = []
    cct: List[int] = []
    task_m = country_m = guard_m = tq_exact = 0
    koj: List[float] = []
    ktj: List[float] = []

    for q in questions:
        dc, tc, cc = await call(classify_sys, q)
        dp, tp, cp = await call(prep_sys, q)
        dm, tm, cm = await call(merged_sys, q)
        c, p, m = _parse_json(tc), _parse_json(tp), _parse_json(tm)
        base_ms.append(dc + dp)
        cand_ms.append(dm)
        if cc and cp:
            bct.append(cc + cp)
        if cm:
            cct.append(cm)
        task_m += str(c.get("task", "")).lower() == str(m.get("task", "")).lower()
        country_m += str(p.get("country", "")).strip() == str(m.get("country", "")).strip()
        guard_m += (bool(p.get("bad_word_detected")) == bool(m.get("bad_word_detected"))
                    and bool(p.get("pii_detected")) == bool(m.get("pii_detected")))
        tq_exact += str(p.get("transformed_query", "")).strip() == str(m.get("transformed_query", "")).strip()
        koj.append(_jaccard(p.get("keywords_from_original"), m.get("keywords_from_original")))
        ktj.append(_jaccard(p.get("keywords_from_transformed"), m.get("keywords_from_transformed")))

    n = len(questions)
    bm, cmv = _stats(base_ms), _stats(cand_ms)
    print(f"=== N={n} concurrency=1 warmup-excluded ===")
    print(f"BASELINE classify+prep (2 calls): mean={bm[0]:.0f}ms p50={bm[1]:.0f} p95={bm[2]:.0f} max={bm[3]:.0f}")
    print(f"MERGED   (1 call)               : mean={cmv[0]:.0f}ms p50={cmv[1]:.0f} p95={cmv[2]:.0f} max={cmv[3]:.0f}")
    print(f"analysis-stage reduction        : mean {100 * (1 - cmv[0] / bm[0]):.1f}%  ({bm[0] - cmv[0]:.0f}ms/req)")
    if bct and cct:
        print(f"completion tokens: baseline={statistics.mean(bct):.0f} merged={statistics.mean(cct):.0f}")
    print("=== ACCURACY merged vs current 2-call (pseudo-gold regression) ===")
    print(f"task match          {task_m}/{n} = {100 * task_m / n:.1f}%")
    print(f"country match       {country_m}/{n} = {100 * country_m / n:.1f}%")
    print(f"guardrail match     {guard_m}/{n} = {100 * guard_m / n:.1f}%")
    print(f"transformed_q exact {tq_exact}/{n} = {100 * tq_exact / n:.1f}%")
    print(f"kw_orig Jaccard {statistics.mean(koj):.3f}  kw_tr Jaccard {statistics.mean(ktj):.3f}")

    if args.output:
        json.dump(
            {"n": n, "baseline_ms": base_ms, "merged_ms": cand_ms,
             "task_match": task_m, "country_match": country_m, "guardrail_match": guard_m,
             "transformed_query_exact": tq_exact,
             "kw_orig_jaccard": statistics.mean(koj), "kw_tr_jaccard": statistics.mean(ktj)},
            open(args.output, "w"), ensure_ascii=False, indent=2,
        )


if __name__ == "__main__":
    asyncio.run(main())

#!/usr/bin/env python3
"""Ablation of combined acceleration techniques for the query-analysis stage.

Compares, on the same fixed questions and the real hybrid search:
  B  baseline: classify + preparation, two sequential calls on the orchestrator
  M1 call fusion: the two merged into one call
  M2 M1 + model right-sizing: a fast small model (reasoning_effort=low)
  M3 M2 + compact output schema
  M4 M3 + grammar-constrained (guided_json) decoding

Reports per method: analysis-stage latency and retrieval fidelity (document-ID
Recall / Top-1 / exact-rank vs the baseline's search results) plus country and
keyword agreement. No prompts, questions, country lists, or secrets are stored
here; they are reconstructed at run time from --rag-root, and endpoints/keys are
CLI/env. Run only against a system you are authorised to test.
See docs/MOLEG_ACCELERATION_ABLATION_20260904.md.
"""
from __future__ import annotations
import argparse, asyncio, importlib.util, json, os, statistics, sys, time, types


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
    src = os.path.join(root, "src", "core", "agent_orchestrator", "orchestrator.py")
    spec = importlib.util.spec_from_file_location("orch_standalone", src)
    m = importlib.util.module_from_spec(spec); sys.modules["orch_standalone"] = m
    spec.loader.exec_module(m)
    return m._CLASSIFY_SYSTEM


def _jac(a, b):
    a, b = set(a), set(b)
    return 1.0 if (not a and not b) else (0.0 if (not a or not b) else len(a & b) / len(a | b))


def _stat(v):
    v = sorted(v); n = len(v); return sum(v) / n, v[n // 2], v[min(n - 1, int(0.95 * n))]


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rag-root", required=True)
    ap.add_argument("--orchestrator-url", required=True, help="big model base url, e.g. http://host:8000/v1")
    ap.add_argument("--orchestrator-model", required=True)
    ap.add_argument("--fast-url", required=True, help="small model base url, e.g. http://host:8006/v1")
    ap.add_argument("--fast-model", required=True)
    ap.add_argument("--api-key", default=os.environ.get("ORCH_API_KEY", "x"))
    ap.add_argument("--limit", type=int, default=25)
    ap.add_argument("--output")
    args = ap.parse_args()

    root = os.path.abspath(args.rag_root); _load_env(root)
    sys.path.insert(0, os.path.join(root, "src")); sys.path.insert(0, os.path.join(root, "scripts"))
    from config.prompts import get_preparation_system_prompt
    from load_test_questions import build_test_queries
    from domain.country.service import resolve_country_to_canonical, get_available_countries
    from infra.llm.law_tool_executors import tool_search_laws
    from openai import AsyncOpenAI

    CLASSIFY = _classify_system(root)
    COUNTRIES = get_available_countries()
    PREP = get_preparation_system_prompt(COUNTRIES)[0]["content"]
    BIG = AsyncOpenAI(base_url=args.orchestrator_url, api_key=args.api_key, timeout=120)
    FAST = AsyncOpenAI(base_url=args.fast_url, api_key=args.api_key, timeout=120)

    MERGED = ("당신은 마스터 라우터 겸 쿼리 분석기입니다. [A]로 task 분류, [B]로 쿼리 분석하여 JSON 하나로 출력.\n"
              "===== [A] =====\n" + CLASSIFY + "\n===== [B] =====\n" + PREP +
              '\n출력: {"task":"","bad_word_detected":false,"pii_detected":false,"transformed_query":"",'
              '"keywords_from_original":[],"keywords_from_transformed":[],"country":"","specific_article_number":0,"law_title_search_hint":""}\n')
    CLIST = ", ".join(COUNTRIES)
    COMPACT = ("질문에서 검색값만 뽑아 JSON 하나로. task, country(목록표기, 없으면 \"\"), "
               "keywords(주제 명사, 국가/메타어 제외), search_terms(검색용 한 문장, 국가명 제외).\n"
               f"국가 목록: {CLIST}\n출력: {{\"task\":\"\",\"country\":\"\",\"keywords\":[],\"search_terms\":\"\"}}\n")
    SCHEMA = {"type": "object", "properties": {"task": {"type": "string"}, "country": {"type": "string"},
              "keywords": {"type": "array", "items": {"type": "string"}}, "search_terms": {"type": "string"}},
              "required": ["task", "country", "keywords", "search_terms"]}

    async def chat(client, model, system, user, extra=None):
        kw = {"model": model, "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
              "temperature": 0, "response_format": {"type": "json_object"}}
        if extra: kw["extra_body"] = extra
        t = time.perf_counter(); r = await client.chat.completions.create(**kw)
        return (time.perf_counter() - t) * 1000, (r.choices[0].message.content or "")

    def pj(s):
        try:
            i, j = s.find("{"), s.rfind("}"); return json.loads(s[i:j + 1]) if i >= 0 else {}
        except Exception:
            return {}

    def full_fields(p):
        kws = [k.strip() for k in ((p.get("keywords_from_original") or []) + (p.get("keywords_from_transformed") or [])) if k and k.strip()]
        seen = set(); mk = [k for k in kws if not (k in seen or seen.add(k))]
        return (p.get("country") or "").strip(), mk, (p.get("transformed_query") or "").strip()

    def compact_fields(p):
        return (p.get("country") or "").strip(), [k.strip() for k in (p.get("keywords") or []) if k and k.strip()], (p.get("search_terms") or "").strip()

    def ncountry(c):
        return (resolve_country_to_canonical(c, COUNTRIES) or c).strip() if c else ""

    async def search(country, kws, st, q):
        raw = await tool_search_laws({"search_terms": st or (" ".join(kws) if kws else q), "country": ncountry(country) or None, "keywords": kws}, "exp")
        try:
            data = json.loads(raw); laws = data.get("laws") if isinstance(data, dict) else None
            return [str(x.get("id")) for x in laws if isinstance(x, dict) and x.get("id")] if isinstance(laws, list) else []
        except Exception:
            return []

    METHODS = ["B", "M1", "M2", "M3", "M4"]
    async def extract(m, q):
        if m == "B":
            d1, _ = await chat(BIG, args.orchestrator_model, CLASSIFY, q)
            d2, t = await chat(BIG, args.orchestrator_model, PREP, q); return d1 + d2, full_fields(pj(t))
        if m == "M1":
            d, t = await chat(BIG, args.orchestrator_model, MERGED, q); return d, full_fields(pj(t))
        if m == "M2":
            d, t = await chat(FAST, args.fast_model, MERGED, q, {"reasoning_effort": "low"}); return d, full_fields(pj(t))
        if m == "M3":
            d, t = await chat(FAST, args.fast_model, COMPACT, q, {"reasoning_effort": "low"}); return d, compact_fields(pj(t))
        if m == "M4":
            d, t = await chat(FAST, args.fast_model, COMPACT, q, {"reasoning_effort": "low", "guided_json": SCHEMA}); return d, compact_fields(pj(t))

    qs = build_test_queries()[: args.limit]
    for m in METHODS:
        try: await extract(m, qs[0])
        except Exception: pass
    res = {m: {"ms": [], "ids": [], "c": [], "kw": []} for m in METHODS}
    for q in qs:
        for m in METHODS:
            try: dt, (c, kw, st) = await extract(m, q)
            except Exception: dt, c, kw, st = 0, "", [], ""
            res[m]["ms"].append(dt); res[m]["ids"].append(await search(c, kw, st, q))
            res[m]["c"].append(ncountry(c)); res[m]["kw"].append(kw)
    B = "B"; bmean = _stat(res[B]["ms"])[0]; summary = {}
    print(f"{'method':8s} {'mean':>7s} {'p50':>7s} {'p95':>7s} {'speedup':>8s} {'Recall':>7s} {'Top1':>6s} {'exact':>6s} {'ctry':>6s} {'kwJ':>6s}")
    for m in METHODS:
        ms = _stat(res[m]["ms"]); rec = []; t1 = []; ex = []; ct = []; kj = []
        for i in range(len(qs)):
            b, x = res[B]["ids"][i], res[m]["ids"][i]
            if b:
                rec.append(len(set(b) & set(x)) / len(set(b)))
                t1.append(1.0 if (x and b and x[0] == b[0]) else 0.0)
                ex.append(1.0 if x == b else 0.0)
            ct.append(1.0 if res[m]["c"][i] == res[B]["c"][i] else 0.0)
            kj.append(_jac(res[m]["kw"][i], res[B]["kw"][i]))
        R = statistics.mean(rec or [1]); T = statistics.mean(t1 or [1]); E = statistics.mean(ex or [1]); C = statistics.mean(ct); K = statistics.mean(kj)
        print(f"{m:8s} {ms[0]:6.0f} {ms[1]:6.0f} {ms[2]:6.0f} {bmean/ms[0]:6.2f}x {R:7.3f} {T:6.3f} {E:6.3f} {C:6.3f} {K:6.3f}")
        summary[m] = {"mean_ms": round(ms[0]), "p50_ms": round(ms[1]), "p95_ms": round(ms[2]), "speedup_x": round(bmean/ms[0], 2),
                      "recall": round(R, 3), "top1": round(T, 3), "exact_rank": round(E, 3), "country_match": round(C, 3), "kw_jaccard": round(K, 3)}
    if args.output:
        json.dump(summary, open(args.output, "w"), ensure_ascii=False, indent=2)
    print("Accuracy = retrieval fidelity vs baseline (pseudo-gold), not expert-judged absolute accuracy.")


if __name__ == "__main__":
    asyncio.run(main())

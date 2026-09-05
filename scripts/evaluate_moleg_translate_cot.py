#!/usr/bin/env python3
"""Measure wasted chain-of-thought on the translation agent and its reduction.

Certain agents (here an ollama gpt-oss:20b translation agent) emit long
reasoning that the application discards, adding pure latency. This script
measures default reasoning vs a reduced-reasoning setting on a set of legal
clauses, reporting latency, reasoning length, and output preservation.

No secrets, prompts, or corpus text are stored in this repository. Provide the
ollama base url and the clause file at run time. Run only against a system you
are authorised to test. See docs/MOLEG_COT_AND_CONTENTION_20260904.md.
"""
from __future__ import annotations
import argparse, json, statistics, time, urllib.request


def _chat(base_url: str, model: str, prompt: str, think, timeout: float):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}],
            "stream": False, "options": {"temperature": 0}}
    if think is not None:
        body["think"] = think
    req = urllib.request.Request(base_url.rstrip("/") + "/api/chat",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t = time.perf_counter()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        d = json.load(r)
    dt = (time.perf_counter() - t) * 1000.0
    m = d.get("message", {})
    return dt, (m.get("content") or ""), (m.get("thinking") or ""), d.get("eval_count") or 0


def _bigrams(s: str):
    s = "".join(s.lower().split())
    return {s[i:i + 2] for i in range(len(s) - 1)} if len(s) >= 2 else ({s} if s else set())


def _jaccard(a: str, b: str) -> float:
    A, B = _bigrams(a), _bigrams(b)
    return len(A & B) / len(A | B) if (A or B) else 1.0


def _stats(v):
    v = sorted(v); n = len(v)
    return sum(v) / n, v[n // 2], v[min(n - 1, int(0.95 * n))], v[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", required=True, help="ollama base url, e.g. http://127.0.0.1:11434")
    ap.add_argument("--model", default="gpt-oss:20b")
    ap.add_argument("--clauses-file", required=True, help="text file, one source clause per line")
    ap.add_argument("--reduced-think", default="low", help="value for ollama 'think' (e.g. low, false)")
    ap.add_argument("--target-lang", default="한국어")
    ap.add_argument("--timeout", type=float, default=150.0)
    args = ap.parse_args()

    clauses = [ln.strip() for ln in open(args.clauses_file, encoding="utf-8") if ln.strip()]
    instr = f"다음 법령 조문을 {args.target_lang}로 정확히 번역하세요. 번역문만 출력하세요:\n\n"

    def call(c, think):
        return _chat(args.base_url, args.model, instr + c, think, args.timeout)

    call(clauses[0], args.reduced_think)  # warm-up
    d_ms, r_ms, sims, d_think, r_think = [], [], [], [], []
    for c in clauses:
        dd, dc, dt_, _ = call(c, None)
        dr, rc, rt_, _ = call(c, args.reduced_think)
        d_ms.append(dd); r_ms.append(dr); sims.append(_jaccard(dc, rc))
        d_think.append(len(dt_)); r_think.append(len(rt_))
    dm, rm = _stats(d_ms), _stats(r_ms)
    print(f"=== N={len(clauses)} clause translations, {args.model} ===")
    print(f"default reasoning: mean={dm[0]:.0f}ms p50={dm[1]:.0f} p95={dm[2]:.0f} think_chars={statistics.mean(d_think):.0f}")
    print(f"think={args.reduced_think}: mean={rm[0]:.0f}ms p50={rm[1]:.0f} p95={rm[2]:.0f} think_chars={statistics.mean(r_think):.0f}")
    print(f"latency reduction: {100 * (1 - rm[0] / dm[0]):.1f}%  ({dm[0] / rm[0]:.1f}x)")
    print(f"output char-bigram Jaccard (default vs reduced): mean={statistics.mean(sims):.3f} min={min(sims):.3f}")
    print("NOTE: low Jaccard on short multilingual strings understates equivalence; inspect pairs and use reference-based eval before adoption.")


if __name__ == "__main__":
    main()

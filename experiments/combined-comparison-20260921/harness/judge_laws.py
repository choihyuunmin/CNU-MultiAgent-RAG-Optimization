import argparse
import asyncio
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

SYSTEM = (
    "당신은 법령 검색 결과를 평가하는 심사자입니다. 사용자 질문과 후보 법령 조문 목록이 주어집니다. "
    "각 조문이 그 질문에 답하는 근거로 적절한지 0, 1, 2 중 하나로 판정하세요.\n"
    "2: 질문의 핵심 쟁점을 직접 다루는 조문.\n"
    "1: 같은 주제이지만 핵심 쟁점을 직접 다루지는 않는 조문.\n"
    "0: 질문과 관련 없는 조문(다른 주제이거나, 질문이 묻는 국가가 아닌 조문).\n"
    "모든 후보에 대해 판정하고, 설명 없이 다음 JSON 형식으로만 답하세요: "
    "{\"scores\": [{\"idx\": 1, \"score\": 2}, {\"idx\": 2, \"score\": 0}]}"
)


def law_key(law):
    item = str(law.get("item_id") or "").strip()
    if item:
        return item
    return f"{law.get('law_id')}#{law.get('paragraph')}"


def law_text(law, max_chars):
    content = (law.get("paragraph_content") or "")[:max_chars]
    return (f"국가: {law.get('country') or '-'} / 법령: {law.get('title') or '-'} / "
            f"조문: {law.get('subject') or '-'} {law.get('paragraph') or ''}\n내용: {content}")


def parse_scores(text, n):
    try:
        data = json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text or "", flags=re.S)
        if not m:
            return None
        try:
            data = json.loads(m.group(0))
        except Exception:
            return None
    out = {}
    for row in (data.get("scores") or []) if isinstance(data, dict) else []:
        try:
            idx, score = int(row["idx"]), int(row["score"])
        except Exception:
            continue
        if 1 <= idx <= n and score in (0, 1, 2):
            out[idx] = score
    return out if len(out) == n else None


async def judge_chunk(call_llm, role, question, laws, request_id, max_chars):
    user = "질문: " + question + "\n\n후보 조문:\n" + "\n\n".join(f"[{i + 1}] " + law_text(law, max_chars) for i, law in enumerate(laws))
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
    for attempt, fmt in enumerate(({"type": "json_object"}, None, None)):
        try:
            text = await call_llm(messages=messages, response_format=fmt, temperature=0, request_id=f"{request_id}-{attempt}",
                                  role=role, enable_thinking=False)
        except Exception as exc:  # transport or format error: retry without the JSON constraint
            text = ""
            err = repr(exc)[:120]
        else:
            err = None
        scores = parse_scores(text, len(laws))
        if scores is not None:
            return scores, attempt, None
    return None, attempt, err or "unparsable"


async def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=Path, required=True, help="root holding main*/level-*/round-*/responses/sweep-trial/<arm>/client_requests.jsonl")
    p.add_argument("--questions", type=Path, default=Path("/opt/cnu/questions.jsonl"))
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--judges", default="master,tool_synthesis")
    p.add_argument("--concurrency", type=int, default=6)
    p.add_argument("--chunk", type=int, default=8)
    p.add_argument("--max-chars", type=int, default=600)
    p.add_argument("--repeat-fraction", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=20260921)
    p.add_argument("--dry-run", action="store_true", help="pool and plan only; no model calls")
    args = p.parse_args()
    if not args.dry_run:
        sys.path.insert(0, "/app/src")
        from infra.llm.client import call_llm

    questions = {}
    for line in args.questions.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            questions[row["id"]] = row["query"]

    pool = defaultdict(dict)  # question_id -> law_key -> law fields (kept in memory only)
    sources = Counter()
    for path in sorted(args.results.glob("main*/level-*/round-*/responses/sweep-trial/*/client_requests.jsonl")):
        arm = path.parent.name
        for line in path.read_text(errors="replace").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            resp = rec.get("response") or {}
            for law in resp.get("laws") or []:
                if isinstance(law, dict):
                    pool[rec["question_id"]].setdefault(law_key(law), law)
                    sources[arm] += 1
    judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"questions with pooled laws: {len(pool)}; pooled pairs: {sum(len(v) for v in pool.values())}; per-arm law mentions: {dict(sources)}", flush=True)

    rng = random.Random(args.seed)
    repeat_qids = set(rng.sample(sorted(pool), max(1, int(len(pool) * args.repeat_fraction))))
    tasks = []
    for qid in sorted(pool):
        keys = sorted(pool[qid])
        order = list(keys)
        random.Random(hashlib.sha256(qid.encode()).hexdigest()).shuffle(order)
        passes = [(1, order)] + ([(2, list(reversed(order)))] if qid in repeat_qids else [])
        for pass_no, ordered in passes:
            for c in range(0, len(ordered), args.chunk):
                chunk_keys = ordered[c:c + args.chunk]
                for role in judges:
                    tasks.append((qid, role, pass_no, c // args.chunk, chunk_keys))
    print(f"judge calls planned: {len(tasks)}", flush=True)
    if args.dry_run:
        (args.out / "pool-stats.json").write_text(json.dumps({"questions": len(pool), "pooled_pairs": sum(len(v) for v in pool.values()), "calls": len(tasks),
            "pool_sizes": dict(Counter(len(v) for v in pool.values())), "per_arm_mentions": dict(sources)}, indent=1) + "\n")
        return

    sem = asyncio.Semaphore(args.concurrency)
    results, failures = [], Counter()

    async def run_one(qid, role, pass_no, chunk_no, chunk_keys):
        async with sem:
            laws = [pool[qid][k] for k in chunk_keys]
            scores, attempts, err = await judge_chunk(call_llm, role, questions.get(qid, ""), laws, f"judge-{qid}-{role}-p{pass_no}-c{chunk_no}", args.max_chars)
            if scores is None:
                failures[role] += 1
                for k in chunk_keys:
                    results.append({"question_id": qid, "law_key": k, "judge": role, "pass": pass_no, "score": None, "error": err})
            else:
                for i, k in enumerate(chunk_keys):
                    results.append({"question_id": qid, "law_key": k, "judge": role, "pass": pass_no, "score": scores[i + 1], "attempts": attempts + 1})

    done = 0
    for i in range(0, len(tasks), 200):
        await asyncio.gather(*(run_one(*t) for t in tasks[i:i + 200]))
        done += len(tasks[i:i + 200])
        print(f"progress {done}/{len(tasks)} failures {dict(failures)}", flush=True)

    with (args.out / "judgments.jsonl").open("w", encoding="utf-8") as fh:
        for row in results:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    # summary: score distribution per judge, inter-judge agreement on pass 1, repeat agreement per judge
    first = {(r["question_id"], r["law_key"], r["judge"]): r["score"] for r in results if r["pass"] == 1}
    second = {(r["question_id"], r["law_key"], r["judge"]): r["score"] for r in results if r["pass"] == 2}
    summary = {"questions": len(pool), "pooled_pairs": sum(len(v) for v in pool.values()), "judges": judges, "calls": len(tasks),
               "failures": dict(failures), "score_distribution": {}, "inter_judge": {}, "repeat": {}}
    for role in judges:
        vals = [v for (q, k, j), v in first.items() if j == role and v is not None]
        summary["score_distribution"][role] = dict(Counter(vals))
        both = [(first[(q, k, role)], second[(q, k, role)]) for (q, k, j) in second if j == role and first.get((q, k, role)) is not None and second[(q, k, role)] is not None]
        if both:
            summary["repeat"][role] = {"pairs": len(both), "exact": sum(a == b for a, b in both) / len(both),
                                       "relevant_flag": sum((a >= 1) == (b >= 1) for a, b in both) / len(both)}
    if len(judges) >= 2:
        a, b = judges[0], judges[1]
        both = [(first[(q, k, a)], first[(q, k, b)]) for (q, k, j) in first if j == a and first.get((q, k, b)) is not None and first[(q, k, a)] is not None]
        if both:
            n = len(both)
            pa = sum(x >= 1 for x, _ in both) / n
            pb = sum(y >= 1 for _, y in both) / n
            po = sum((x >= 1) == (y >= 1) for x, y in both) / n
            pe = pa * pb + (1 - pa) * (1 - pb)
            summary["inter_judge"] = {"pairs": n, "exact": sum(x == y for x, y in both) / n, "relevant_flag": po,
                                      "kappa_relevant_flag": (po - pe) / (1 - pe) if pe < 1 else None}
    (args.out / "judge-summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    asyncio.run(main())

"""Offline summary of the follow-up trial (stdlib only).

Reports, per round: exact law-set agreement and nonempty-reference recall for every
arm pair, separating same-arm instance pairs (original-a vs original-b, capability-a
vs capability-b) from cross-arm pairs; paired completed latency; and, from the
application logs, handler-output / next-stage-object hash equality for branches whose
prepared-argument hashes are equal between two arms.
"""
import argparse, json, math, random, re, statistics
from collections import defaultdict
from datetime import datetime
from pathlib import Path


def valid(r):
    resp = r.get("response")
    return r.get("status") == "ok" and isinstance(resp, dict) and isinstance(resp.get("laws"), list) and isinstance(resp.get("comment"), str)

def uniq(ids):
    out, seen = [], set()
    for v in ids:
        v = str(v or "").strip()
        if v and v not in seen: out.append(v); seen.add(v)
    return out

def recall(a, b):
    g, p = set(uniq(a.get("law_ids", []))), set(uniq(b.get("law_ids", [])))
    return None if not g else len(g & p)/len(g)

def sign_p(pos, neg):
    n, k = pos+neg, min(pos, neg)
    return min(1.0, 2*sum(math.comb(n, i) for i in range(k+1))/2**n) if n else None

def boot_mean_ci(x, b=4000, seed=20260917):
    rng = random.Random(seed); n = len(x); vals = []
    for _ in range(b):
        vals.append(sum(x[rng.randrange(n)] for _ in range(n))/n)
    vals.sort(); return vals[int(0.025*b)], vals[int(0.975*b)-1]

def ts(s):
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\.(\d+)([+-]\d\d:\d\d)", s)
    return datetime.fromisoformat(m.group(1)+m.group(3)).timestamp()+float("0."+m.group(2))


def main(root, logs, output):
    status = json.loads((root/"status.json").read_text())
    cells = {}
    for c in status["completed"]:
        p = root/f"round-{c['round']}/responses/review-trial/{c['method']}/client_requests.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        cells[c["round"], c["method"]] = {x["question_id"]: x for x in rows}
    rounds = sorted({r for r, _ in cells}); arms = sorted({m for _, m in cells})
    result = {"cells": status["completed"], "agreement": [], "latency": [], "hashes": []}
    for r in rounds:
        for i, a in enumerate(arms):
            for b in arms[i+1:]:
                A, B = cells[r, a], cells[r, b]
                qs = [q for q in A if q in B and valid(A[q]) and valid(B[q])]
                same = [set(uniq(A[q]["law_ids"])) == set(uniq(B[q]["law_ids"])) for q in qs]
                rec = [recall(A[q], B[q]) for q in qs]; rec = [x for x in rec if x is not None]
                kind = "same-arm instances" if a.rsplit("-", 1)[0] == b.rsplit("-", 1)[0] else "cross-arm"
                result["agreement"].append({"round": r, "a": a, "b": b, "kind": kind, "pairs": len(qs),
                    "exact_set": sum(same)/len(qs) if qs else None, "nonempty_recall": statistics.mean(rec) if rec else None, "nonempty": len(rec)})
                d = [(A[q]["duration_ms"]-B[q]["duration_ms"])/1000 for q in qs]
                pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
                result["latency"].append({"round": r, "a": a, "b": b, "kind": kind, "pairs": len(qs), "mean_a": statistics.mean(A[q]["duration_ms"]/1000 for q in qs),
                    "mean_b": statistics.mean(B[q]["duration_ms"]/1000 for q in qs), "mean_saving_a_minus_b": statistics.mean(d), "ci": boot_mean_ci(d),
                    "median": statistics.median(d), "pos": pos, "neg": neg, "sign_p": sign_p(pos, neg)})
    # application logs: hashes per (round, arm, question)
    windows = {(r, m): (min(datetime.fromisoformat(x["started_at"]).timestamp() for x in c.values())-2,
                        max(datetime.fromisoformat(x["completed_at"]).timestamp() for x in c.values())+2) for (r, m), c in cells.items()}
    branches = defaultdict(list)  # (r, arm, q) -> list of boundary events
    for arm in arms:
        path = logs/f"{arm}-application.log"
        if not path.exists(): continue
        events = []
        for line in path.open(errors="replace"):
            i = line.find(' {"event"')
            if i < 0: continue
            try: e = json.loads(line[i+1:])
            except json.JSONDecodeError: continue
            e["_t"] = ts(line[:35]); events.append(e)
        def rnd(t):
            for r in rounds:
                lo, hi = windows.get((r, arm), (None, None))
                if lo is not None and lo <= t <= hi: return r
        case = {}
        for e in events:
            if e.get("event") == "llm_finished" and e.get("application_request_id") and e.get("case_id"):
                r = rnd(e["_t"])
                if r: case[r, e["application_request_id"]] = re.sub(r"-r\d+-c\d+$", "", e["case_id"])
        handler_out = {}
        for e in events:
            if e.get("event") == "review_handler":
                r = rnd(e["_t"]); q = case.get((r, e.get("request_id")))
                if r and q: handler_out[e["branch_id"]] = e.get("output_hash")
        for e in events:
            if e.get("event") == "review_boundary":
                r = rnd(e["_t"]); q = case.get((r, e.get("request_id")))
                if r and q in cells.get((r, arm), {}):
                    branches[r, arm, q].append({"prepared": None, "output": handler_out.get(e["branch_id"]),
                        "result": e.get("result_hash"), "law_ids": e.get("law_ids_hash"), "status": e.get("status")})
        for e in events:
            if e.get("event") == "review_prepared":
                r = rnd(e["_t"]); q = case.get((r, e.get("request_id")))
                if r and q:
                    for b in branches.get((r, arm, q), []):
                        if b["prepared"] is None: b["prepared"] = e.get("prepared_hash"); break
    for r in rounds:
        for i, a in enumerate(arms):
            for b in arms[i+1:]:
                n_eq_prepared = n_eq_output = n_eq_result = n_eq_ids = 0; n_q = 0
                for q in cells[r, a]:
                    ba, bb = branches.get((r, a, q), []), branches.get((r, b, q), [])
                    if not ba or not bb or len(ba) != len(bb): continue
                    n_q += 1
                    pa = sorted((x["prepared"] or "", x["output"] or "", x["result"] or "", x["law_ids"] or "") for x in ba)
                    pb = sorted((x["prepared"] or "", x["output"] or "", x["result"] or "", x["law_ids"] or "") for x in bb)
                    if [x[0] for x in pa] == [x[0] for x in pb]:
                        n_eq_prepared += 1
                        hashed = all(x[1] and x[2] and x[3] for x in pa+pb)  # None hashes are not comparable
                        n_eq_output += hashed and [x[1] for x in pa] == [x[1] for x in pb]
                        n_eq_result += hashed and [x[2] for x in pa] == [x[2] for x in pb]
                        n_eq_ids += hashed and [x[3] for x in pa] == [x[3] for x in pb]
                result["hashes"].append({"round": r, "a": a, "b": b, "questions_with_branches": n_q, "equal_prepared": n_eq_prepared,
                    "equal_handler_output_given_equal_prepared": n_eq_output, "equal_next_stage_object": n_eq_result, "equal_law_id_list": n_eq_ids})
    output.write_text(json.dumps(result, indent=1)+"\n")
    for row in result["agreement"]:
        print(f"round {row['round']} {row['a']:13s} vs {row['b']:13s} [{row['kind']:19s}] exact-set {row['exact_set']*100:5.1f}%  recall {row['nonempty_recall']*100:6.2f}% (n={row['nonempty']})")
    for row in result["hashes"]:
        print(f"round {row['round']} {row['a']:13s} vs {row['b']:13s} equal prepared {row['equal_prepared']}/{row['questions_with_branches']}  equal handler output {row['equal_handler_output_given_equal_prepared']}  equal next-stage object {row['equal_next_stage_object']}  equal law-id list {row['equal_law_id_list']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--logs", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(); main(a.root, a.logs, a.output)

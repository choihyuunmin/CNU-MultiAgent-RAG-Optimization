"""Stage-1 analysis (DISPATCH_VALIDATION_PROTOCOL_20260918): original vs unsigned direct at concurrency 4, plus original repeat.
Works with the public-adapter trace format (calls inside request_finished) and the old per-call llm_finished format."""
import json, math, re, statistics as st, sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
import numpy as np
sys.path.insert(0, "/data/project/choihm/CNU-MultiAgent-RAG-Optimization/src")
from cnu_rag_optimization.trace_equivalence import join_branch_events, compare_branch_sets
ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); RNG = np.random.RandomState(20260918); B = 10000
def ts(s):
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\.(\d+)([+-]\d\d:\d\d)", s); return datetime.fromisoformat(m.group(1)+m.group(3)).timestamp()+float("0."+m.group(2))
def cts(s): return datetime.fromisoformat(s).timestamp()
def valid(r):
    resp = r.get("response"); return r.get("status") == "ok" and isinstance(resp, dict) and isinstance(resp.get("laws"), list) and isinstance(resp.get("comment"), str)
def uniq(ids):
    out, seen = [], set()
    for v in ids:
        v = str(v or "").strip()
        if v and v not in seen: out.append(v); seen.add(v)
    return out
def sign_p(pos, neg):
    n, k = pos+neg, min(pos, neg); return min(1.0, 2*sum(math.comb(n, i) for i in range(k+1))/2**n) if n else None
def ci(x):
    x = np.asarray(x, float); idx = RNG.randint(0, len(x), size=(B, len(x))); v = x[idx].mean(axis=1); return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
def p95(x): x = sorted(x); return x[int(0.95*(len(x)-1))]
# cells
status = json.loads((ROOT/"main"/"status.json").read_text())
cells = {}
for c in status["completed"]:
    p = ROOT/"main"/f"round-{c['round']}"/"responses"/"review-trial"/c["method"]/"client_requests.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    cells[c["round"], c["method"]] = {x["question_id"]: x for x in rows}
print("cells:", [(k, len(v), sum(valid(x) for x in v.values())) for k, v in cells.items()])
# traces by arm, assigned to cells by time window
windows = {k: (min(cts(x["started_at"]) for x in v.values())-2, max(cts(x["completed_at"]) for x in v.values())+2) for k, v in cells.items()}
def load(arm):
    ev = []
    for line in (ROOT/f"{arm}-application.log").open(errors="replace"):
        i = line.find(' {"event"')
        if i < 0: continue
        try: e = json.loads(line[i+1:]); e["_t"] = ts(line[:35]); ev.append(e)
        except Exception: pass
    return ev
arms = sorted({m for _, m in cells}); traces = {a: load(a) for a in arms}
def cell_of(arm, t):
    for (r, m), (lo, hi) in windows.items():
        if m == arm and lo <= t <= hi: return r
def qid(case): return re.sub(r"-r\d+-c\d+$", "", case or "")
result = {"cells": [], "latency": [], "endpoints": [], "boundary": {}, "calls": {}, "hashes": [], "engine": [], "agreement": []}
# per-cell: request->question map, branches, calls
branches = defaultdict(list); reqcalls = defaultdict(list); casemap = {}
for arm in arms:
    for e in traces[arm]:
        r = cell_of(arm, e["_t"])
        if r is None: continue
        if e.get("event") == "request_finished":
            q = qid(e.get("case_id"))
            if q in cells.get((r, arm), {}):
                for c in e.get("calls", []):
                    if c.get("application_request_id"): casemap[r, arm, c["application_request_id"]] = q
                    reqcalls[r, arm, q].append(c)
        elif e.get("event") == "llm_finished" and e.get("application_request_id"):
            q = qid(e.get("case_id")); casemap[r, arm, e["application_request_id"]] = q
    for r in {rr for (rr, m) in cells if m == arm}:
        sel = [e for e in traces[arm] if cell_of(arm, e["_t"]) == r]
        for b in join_branch_events(sel):
            q = casemap.get((r, arm, b["request_id"]))
            if q in cells.get((r, arm), {}):
                b["_events"] = None; branches[r, arm, q].append(b)
        # boundary timing
        nh = [e["non_handler_ms"] for e in sel if e.get("event") == "review_boundary" and e.get("status") == "ok" and casemap.get((r, arm, e.get("request_id"))) in cells.get((r, arm), {})]
        hd = [e["handler_ms"] for e in sel if e.get("event") == "review_boundary" and e.get("status") == "ok" and casemap.get((r, arm, e.get("request_id"))) in cells.get((r, arm), {})]
        eff = defaultdict(int)
        for e in sel:
            if e.get("event") == "review_boundary": eff[e.get("effective_method")] += 1
        result["boundary"][f"{r}-{arm}"] = {"branches": len(nh), "non_handler_ms_mean": st.mean(nh) if nh else None, "non_handler_ms_p95": p95(nh) if nh else None, "handler_ms_mean": st.mean(hd) if hd else None, "effective_method_counts": dict(eff)}
# cell stats
for (r, m), v in sorted(cells.items()):
    recs = list(v.values()); lat = [x["duration_ms"]/1000 for x in recs]; ok = [x for x in recs if valid(x)]
    t0 = min(cts(x["started_at"]) for x in recs); t1 = max(cts(x["completed_at"]) for x in recs)
    result["cells"].append({"round": r, "arm": m, "n": len(recs), "valid": len(ok), "failures": len(recs)-len(ok), "mean_s": st.mean(lat), "p50_s": st.median(lat), "p95_s": p95(lat), "wall_s": t1-t0, "valid_per_s": len(ok)/(t1-t0)})
# paired latency: original r1 vs direct r1; original r1 vs original r2 (null); direct r1 vs original r2
def paired(a, b, label):
    A, Bc = cells[a], cells[b]; qs = [q for q in A if q in Bc and valid(A[q]) and valid(Bc[q])]
    d = np.array([(A[q]["duration_ms"]-Bc[q]["duration_ms"])/1000 for q in qs]); pos, neg = int((d > 0).sum()), int((d < 0).sum())
    pa = np.array([A[q]["duration_ms"]/1000 for q in qs]); pb = np.array([Bc[q]["duration_ms"]/1000 for q in qs])
    idx = RNG.randint(0, len(qs), size=(B, len(qs))); gain = 1 - pb[idx].mean(axis=1)/pa[idx].mean(axis=1); ratio = np.array([np.percentile(pb[i], 95)/np.percentile(pa[i], 95) for i in idx[:3000]])
    rec = []; same = []
    for q in qs:
        g, p = set(uniq(A[q]["law_ids"])), set(uniq(Bc[q]["law_ids"]))
        if g: rec.append(len(g & p)/len(g))
        same.append(g == p)
    row = {"pair": label, "pairs": len(qs), "mean_a": float(pa.mean()), "mean_b": float(pb.mean()), "saving": float(d.mean()), "ci": ci(d), "pct": float(d.mean()/pa.mean()*100),
           "median": float(np.median(d)), "pos": pos, "neg": neg, "sign_p": sign_p(pos, neg), "sd": float(d.std(ddof=1)),
           "mean_gain_lower": float(np.percentile(gain, 2.5)), "mean_gain_upper": float(np.percentile(gain, 97.5)), "p95_ratio_upper": float(np.percentile(ratio, 97.5)), "p95_ratio_point": float(np.percentile(pb, 95)/np.percentile(pa, 95)),
           "nonempty_recall": float(np.mean(rec)) if rec else None, "n_nonempty": len(rec), "exact_set": float(np.mean(same))}
    result["latency"].append(row); return row
if (1, "original") in cells and (1, "direct") in cells: paired((1, "original"), (1, "direct"), "original r1 vs direct r1")
if (2, "original") in cells:
    paired((1, "original"), (2, "original"), "original r1 vs original r2 (null)")
    if (1, "direct") in cells: paired((2, "original"), (1, "direct"), "original r2 vs direct r1")
# endpoints E/S per arm cell
def endpoints(r, arm, q):
    rec = cells[r, arm][q]; sel = [e for e in traces[arm] if cell_of(arm, e["_t"]) == r and casemap.get((r, arm, e.get("request_id"))) == q]
    b = [e for e in sel if e.get("event") == "review_boundary"]; p = [e for e in sel if e.get("event") == "review_prepared"]
    if not b or not p or not valid(rec): return None
    tl = max(e["_t"] for e in b); tp = min(e["_t"] for e in p); return {"E": tl-cts(rec["started_at"]), "S": tl-tp, "T": rec["duration_ms"]/1000}
def endpoint_pair(a, b, label):
    rows = []
    for q in cells[a]:
        x = endpoints(a[0], a[1], q); y = endpoints(b[0], b[1], q) if q in cells[b] else None
        if x and y: rows.append((x, y))
    if not rows: return
    dE = np.array([x["E"]-y["E"] for x, y in rows]); dS = np.array([x["S"]-y["S"] for x, y in rows])
    result["endpoints"].append({"pair": label, "pairs": len(rows), "E_saving": float(dE.mean()), "E_ci": ci(dE), "S_saving": float(dS.mean()), "S_ci": ci(dS), "E_a": float(np.mean([x["E"] for x, _ in rows])), "S_a": float(np.mean([x["S"] for x, _ in rows]))})
if (1, "original") in cells and (1, "direct") in cells: endpoint_pair((1, "original"), (1, "direct"), "original r1 vs direct r1")
if (2, "original") in cells: endpoint_pair((1, "original"), (2, "original"), "original r1 vs original r2 (null)")
# hash audit between cells (same question) using trace_equivalence
for a, b, label in [((1, "original"), (1, "direct"), "original r1 vs direct r1"), ((1, "original"), (2, "original"), "original r1 vs original r2"), ((2, "original"), (1, "direct"), "original r2 vs direct r1")]:
    if a not in cells or b not in cells: continue
    audit = defaultdict(int); eq = defaultdict(int); n = 0
    for q in cells[a]:
        ba, bb = branches.get((a[0], a[1], q), []), branches.get((b[0], b[1], q), [])
        cmp = compare_branch_sets(ba, bb); audit[cmp["status"]] += 1
        if cmp.get("first_difference"): audit["first_difference_"+cmp["first_difference"]] += 1
        if cmp.get("prepared_equal"):
            n += 1
            for k in ("output", "result", "law_ids"): eq[k] += bool(cmp[k])
    result["hashes"].append({"pair": label, "equal_prepared": n, "equal_output": eq["output"], "equal_result": eq["result"], "equal_law_ids": eq["law_ids"], "audit": dict(audit)})
# per-branch verification inside each arm: prepared==executed, handler_calls==1
for (r, m) in cells:
    bl = [b for (rr, mm, q), bs in branches.items() if rr == r and mm == m for b in bs]
    result["boundary"][f"{r}-{m}"]["joined_branches"] = len(bl); result["boundary"][f"{r}-{m}"]["valid_branches"] = sum(b["valid"] for b in bl)
    result["boundary"][f"{r}-{m}"]["issues"] = dict(__import__("collections").Counter(i for b in bl for i in b["issues"]))
# LLM calls per stage (public adapter: role/model in request_finished calls)
for (r, m) in cells:
    agg = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "api_ms": 0.0})
    for q, calls in ((q, v) for (rr, mm, q), v in reqcalls.items() if rr == r and mm == m):
        for c in calls:
            k = f"{c.get('model')}/{c.get('role')}"; a = agg[k]; a["calls"] += 1; a["prompt"] += c.get("prompt_tokens") or 0; a["completion"] += c.get("completion_tokens") or 0; a["api_ms"] += c.get("api_ms") or 0
    result["calls"][f"{r}-{m}"] = {k: {"calls": v["calls"], "prompt_tokens": v["prompt"], "completion_tokens": v["completion"], "mean_api_ms": v["api_ms"]/v["calls"] if v["calls"] else None} for k, v in agg.items()}
# engine counters per cell
KEYS = {"requests": "vllm:e2e_request_latency_seconds_count", "queue_s": "vllm:request_queue_time_seconds_sum", "prompt": "vllm:prompt_tokens_total", "generation": "vllm:generation_tokens_total", "preempt": "vllm:num_preemptions_total"}
for (r, m) in cells:
    p = ROOT/"main"/f"round-{r}"/f"{m}-metrics.jsonl"
    if not p.exists(): continue
    first, last = {}, {}
    for line in p.open():
        d = json.loads(line)
        if d.get("metrics"): first.setdefault(d["resource"], d); last[d["resource"]] = d
    result["engine"].append({"round": r, "arm": m, **{res: {k: last[res]["metrics"].get(v, 0)-first[res]["metrics"].get(v, 0) for k, v in KEYS.items()} for res in first}})
OUT.write_text(json.dumps(result, indent=1, default=float)+"\n")
for c in result["cells"]: print(f"cell r{c['round']} {c['arm']:9s} valid {c['valid']}/{c['n']} mean {c['mean_s']:.2f} p50 {c['p50_s']:.2f} p95 {c['p95_s']:.1f} wall {c['wall_s']:.0f}s {c['valid_per_s']:.3f}/s")
for x in result["latency"]: print(f"{x['pair']:36s} n={x['pairs']} {x['mean_a']:.2f} vs {x['mean_b']:.2f} saving {x['saving']:.2f} ({x['pct']:.1f}%) CI [{x['ci'][0]:.2f},{x['ci'][1]:.2f}] median {x['median']:.2f} {x['pos']}/{x['neg']} p={x['sign_p']:.3f} | gain lower {x['mean_gain_lower']*100:.1f}% p95 ratio upper {x['p95_ratio_upper']:.3f} | recall {x['nonempty_recall']*100:.1f} exact {x['exact_set']*100:.1f}")
for x in result["endpoints"]: print(f"endpoint {x['pair']:36s} n={x['pairs']} E {x['E_a']:.2f} saving {x['E_saving']:.3f} [{x['E_ci'][0]:.3f},{x['E_ci'][1]:.3f}] S {x['S_a']:.2f} saving {x['S_saving']:.3f} [{x['S_ci'][0]:.3f},{x['S_ci'][1]:.3f}]")
for k, v in result["boundary"].items(): print(f"boundary {k}: {v}")
for x in result["hashes"]: print(f"hashes {x['pair']}: prepared-equal {x['equal_prepared']} output {x['equal_output']} result {x['equal_result']} law_ids {x['equal_law_ids']} audit {x['audit']}")
for e in result["engine"]: print("engine", {k: (v if not isinstance(v, dict) else {kk: round(vv) for kk, vv in v.items()}) for k, v in e.items()})
for k, v in result["calls"].items(): print("calls", k, {kk: (vv["calls"], vv["prompt_tokens"], vv["completion_tokens"], round(vv["mean_api_ms"] or 0)) for kk, vv in v.items()})

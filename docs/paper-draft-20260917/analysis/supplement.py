"""Supplementary analyses of the cnu-review-20260916 three-condition trial.

Reads the copied experiment directory and writes supplement.json plus printed tables.
No new requests are issued; everything is derived from saved client records,
application trace logs, and per-batch vLLM counter samples.
"""
import json, re, math, sys, statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2])
METHODS = ["original", "direct", "capability"]
LABEL = {"original": "Original", "direct": "Unsigned", "capability": "Signed"}
ROUNDS = [1, 2, 3]
ORDER = {1: ["original", "direct", "capability"], 2: ["direct", "capability", "original"], 3: ["capability", "original", "direct"]}
TYPE = {"law_lookup": "Find law", "single_country_content": "Explain content", "multi_country_comparison": "Compare countries", "country_synthesis": "Analyze a case"}
RNG = np.random.RandomState(20260916)
B = 10000

def parse_ts(s):
    # '2026-09-16T22:05:16.900409113+09:00' -> epoch seconds
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\.(\d+)([+-]\d\d:\d\d)", s)
    base = datetime.fromisoformat(m.group(1) + m.group(3))
    return base.timestamp() + float("0." + m.group(2))

def client_ts(s):
    return datetime.fromisoformat(s).timestamp()

def valid(r):
    resp = r.get("response")
    return r.get("status") == "ok" and isinstance(resp, dict) and isinstance(resp.get("laws"), list) and isinstance(resp.get("comment"), str)

def uniq(ids):
    out = []; seen = set()
    for v in ids:
        v = str(v or "").strip()
        if v and v not in seen: out.append(v); seen.add(v)
    return out

# ---------- client records ----------
cells = {}
for r in ROUNDS:
    for m in METHODS:
        p = ROOT / f"main/round-{r}/responses/review-trial/{m}/client_requests.jsonl"
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        assert len(rows) == 200 and len({x["question_id"] for x in rows}) == 200
        cells[r, m] = {x["question_id"]: x for x in rows}
qtype = {q: TYPE[x["category"]] for q, x in cells[1, "original"].items()}

# ---------- application traces ----------
def load_trace(method):
    events = []
    for line in (ROOT / f"{method}-application.log").open(errors="replace"):
        i = line.find(' {"event"')
        if i < 0: continue
        try: d = json.loads(line[i + 1:])
        except json.JSONDecodeError: continue
        d["_t"] = parse_ts(line[:len("2026-09-16T22:05:16.900409113+09:00")])
        events.append(d)
    return events
traces = {m: load_trace(m) for m in METHODS}
# batch time windows per (round, method) from client records; app events are assigned by timestamp
windows = {}
for r in ROUNDS:
    for m in METHODS:
        recs = cells[r, m].values()
        windows[r, m] = (min(client_ts(x["started_at"]) for x in recs) - 2.0, max(client_ts(x["completed_at"]) for x in recs) + 2.0)
def round_of_time(m, t):
    for r in ROUNDS:
        lo, hi = windows[r, m]
        if lo <= t <= hi: return r
    return None
def qid_of_case(case_id):
    return re.sub(r"-r\d+-c\d+$", "", case_id or "")
case_of_request = {m: {} for m in METHODS}
for m in METHODS:
    for e in traces[m]:
        if e.get("event") == "llm_finished" and e.get("application_request_id") and e.get("case_id"):
            r = round_of_time(m, e["_t"])
            if r is not None: case_of_request[m][r, e["application_request_id"]] = e["case_id"]

branch = defaultdict(lambda: defaultdict(list))  # (r,m,q) -> event -> list
wrapped = defaultdict(lambda: defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "api_ms": 0.0}))
wrapped_role = defaultdict(lambda: defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "errors": 0}))
for m in METHODS:
    for e in traces[m]:
        ev = e.get("event"); r = round_of_time(m, e["_t"])
        if r is None: continue
        if ev in ("review_prepared", "review_handler", "review_boundary"):
            cid = case_of_request[m].get((r, e.get("request_id")))
            if not cid: continue
            q = qid_of_case(cid)
            if q not in cells[r, m]: continue
            branch[r, m, q][ev].append(e)
        elif ev == "llm_finished":
            q = qid_of_case(e.get("case_id"))
            if q not in cells[r, m]: continue
            w = wrapped[r, m][e.get("resource")]
            w["calls"] += 1; w["prompt"] += e.get("prompt_tokens") or 0; w["completion"] += e.get("completion_tokens") or 0; w["api_ms"] += e.get("api_ms") or 0
            wr = wrapped_role[r, m][e.get("model")]
            wr["calls"] += 1; wr["prompt"] += e.get("prompt_tokens") or 0; wr["completion"] += e.get("completion_tokens") or 0; wr["errors"] += (e.get("status") != "ok")

# ---------- helpers ----------
def boot_ci(x, stat=np.mean):
    x = np.asarray(x, float); n = len(x)
    idx = RNG.randint(0, n, size=(B, n))
    vals = stat(x[idx], axis=1) if stat is np.mean else np.array([stat(x[i]) for i in idx])
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
def sign_p(pos, neg):
    n = pos + neg; k = min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n)
def trimmed_mean(x, prop=0.1):
    x = np.sort(np.asarray(x, float)); k = int(len(x) * prop)
    return float(x[k:len(x) - k].mean())

def paired(r, a, b, qs=None):
    A, Bc = cells[r, a], cells[r, b]
    out = []
    for q in (qs or A):
        if q in Bc and valid(A[q]) and valid(Bc[q]):
            out.append((q, A[q]["duration_ms"] / 1000, Bc[q]["duration_ms"] / 1000))
    return out

def summarize_pairs(pairs, label):
    d = np.array([x - y for _, x, y in pairs]); lr = np.array([math.log(y / x) for _, x, y in pairs])
    pos = int((d > 0).sum()); neg = int((d < 0).sum())
    ci = boot_ci(d)
    idx = RNG.randint(0, len(lr), size=(B, len(lr)))
    gm = np.exp(lr[idx].mean(axis=1)); gm_ci = (float(np.percentile(gm, 2.5)), float(np.percentile(gm, 97.5)))
    tm_vals = [trimmed_mean(d[i]) for i in idx[:4000]]
    return {"label": label, "n": len(pairs), "mean_a": float(np.mean([x for _, x, _ in pairs])), "mean_b": float(np.mean([y for _, _, y in pairs])),
            "mean_saving": float(d.mean()), "ci": ci, "median": float(np.median(d)), "pos": pos, "neg": neg, "sign_p": sign_p(pos, neg),
            "sd": float(d.std(ddof=1)), "geo_ratio": float(math.exp(lr.mean())), "geo_ci": gm_ci, "geo_change_pct": float((1 - math.exp(lr.mean())) * 100),
            "trimmed10": trimmed_mean(d), "trimmed10_ci": (float(np.percentile(tm_vals, 2.5)), float(np.percentile(tm_vals, 97.5)))}

results = {}
# ---------- A. latency: candidate vs original, and original vs original null control ----------
lat = []
for r in ROUNDS:
    for cand in ["direct", "capability"]:
        lat.append(summarize_pairs(paired(r, "original", cand), f"round {r}: original vs {LABEL[cand]}"))
null = []
for a, b in [(1, 2), (1, 3), (2, 3)]:
    A, Bc = cells[a, "original"], cells[b, "original"]
    pairs = [(q, A[q]["duration_ms"] / 1000, Bc[q]["duration_ms"] / 1000) for q in A if valid(A[q]) and valid(Bc[q])]
    null.append(summarize_pairs(pairs, f"original round {a} vs original round {b}"))
# candidate-vs-candidate (same mechanism) control
same = []
for r in ROUNDS:
    same.append(summarize_pairs(paired(r, "direct", "capability"), f"round {r}: Unsigned vs Signed"))
results["latency"] = lat; results["null_control"] = null; results["same_mechanism"] = same

# ---------- B. position-balanced estimate with question bootstrap ----------
common = [q for q in cells[1, "original"] if all(valid(cells[r, m][q]) for r in ROUNDS for m in METHODS)]
M = {m: np.array([[cells[r, m][q]["duration_ms"] / 1000 for r in ROUNDS] for q in common]) for m in METHODS}  # q x r
def balanced(idx):
    means = {m: M[m][idx].mean() for m in METHODS}
    return means
means = balanced(np.arange(len(common)))
pos_mean = {}
for p in range(3):
    pos_mean[p + 1] = float(np.mean([M[ORDER[r][p]][:, r - 1].mean() for r in ROUNDS]))
bal = {"n_common": len(common), "method_means": {LABEL[m]: float(v) for m, v in means.items()}, "position_means": pos_mean, "round_means": {r: float(np.mean([M[m][:, r - 1].mean() for m in METHODS])) for r in ROUNDS}}
idx = RNG.randint(0, len(common), size=(B, len(common)))
for cand in ["direct", "capability"]:
    diffs = M["original"][idx].mean(axis=(1, 2)) - M[cand][idx].mean(axis=(1, 2))
    # per-question balanced difference (average over rounds) -> sign test and median
    dq = M["original"].mean(axis=1) - M[cand].mean(axis=1)
    lr = np.log(M[cand] / M["original"]).mean()
    lr_b = np.log(M[cand][idx] / M["original"][idx]).mean(axis=(1, 2))
    bal[LABEL[cand]] = {"saving": float(means["original"] - means[cand]), "ci": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
                        "pct": float((means["original"] - means[cand]) / means["original"] * 100),
                        "per_question_median": float(np.median(dq)), "pos": int((dq > 0).sum()), "neg": int((dq < 0).sum()), "sign_p": sign_p(int((dq > 0).sum()), int((dq < 0).sum())),
                        "geo_change_pct": float((1 - math.exp(lr)) * 100), "geo_ci_pct": (float((1 - math.exp(np.percentile(lr_b, 97.5))) * 100), float((1 - math.exp(np.percentile(lr_b, 2.5))) * 100))}
# two-way additive model residual SD (question + cell effects) for reference
results["balanced"] = bal

# ---------- C. exact-set agreement gap ----------
def same_set(a, b): return set(uniq(a.get("law_ids", []))) == set(uniq(b.get("law_ids", [])))
setagree = {}
orig_pairs = {(1, 2), (1, 3), (2, 3)}
oo = {}
for a, b in orig_pairs:
    oo[a, b] = {q: same_set(cells[a, "original"][q], cells[b, "original"][q]) for q in common}
for cand in ["direct", "capability"]:
    co = {r: {q: same_set(cells[r, "original"][q], cells[r, cand][q]) for q in common} for r in ROUNDS}
    rate_c = np.array([[co[r][q] for r in ROUNDS] for q in common], float)  # q x 3
    rate_o = np.array([[oo[k][q] for k in sorted(oo)] for q in common], float)
    gap = rate_o.mean() - rate_c.mean()
    gaps = rate_o[idx].mean(axis=(1, 2)) - rate_c[idx].mean(axis=(1, 2))
    setagree[LABEL[cand]] = {"cand_vs_original_rate": float(rate_c.mean()), "original_repeat_rate": float(rate_o.mean()), "gap_pp": float(gap * 100),
                             "gap_ci_pp": (float(np.percentile(gaps, 2.5) * 100), float(np.percentile(gaps, 97.5) * 100)),
                             "per_round_cand": [float(rate_c[:, i].mean()) for i in range(3)]}
# unsigned vs signed
us = np.array([[same_set(cells[r, "direct"][q], cells[r, "capability"][q]) for r in ROUNDS] for q in common], float)
setagree["Unsigned vs Signed rate"] = float(us.mean())
# nonempty recall gap with CI (macro over nonempty references in the common cohort)
def recall(a, b):
    g = set(uniq(a.get("law_ids", []))); p = set(uniq(b.get("law_ids", [])))
    return None if not g else len(g & p) / len(g)
for cand in ["direct", "capability"]:
    rc = np.array([[recall(cells[r, "original"][q], cells[r, cand][q]) for r in ROUNDS] for q in common], float)
    ro = np.array([[recall(cells[a, "original"][q], cells[b, "original"][q]) for (a, b) in sorted(oo)] for q in common], float)
    mask = ~np.isnan(rc).any(axis=1) & ~np.isnan(ro).any(axis=1)
    rc, ro = rc[mask], ro[mask]
    i2 = RNG.randint(0, len(rc), size=(B, len(rc)))
    g = ro[i2].mean(axis=(1, 2)) - rc[i2].mean(axis=(1, 2))
    setagree[LABEL[cand]]["recall_cand"] = float(rc.mean()); setagree[LABEL[cand]]["recall_orig_repeat"] = float(ro.mean())
    setagree[LABEL[cand]]["recall_gap_pp"] = float((ro.mean() - rc.mean()) * 100); setagree[LABEL[cand]]["recall_gap_ci_pp"] = (float(np.percentile(g, 2.5) * 100), float(np.percentile(g, 97.5) * 100)); setagree[LABEL[cand]]["n_nonempty_common"] = int(mask.sum())
results["set_agreement"] = setagree

# ---------- D. engine counters per batch vs wrapped calls ----------
KEYS = {"requests": "vllm:e2e_request_latency_seconds_count", "prompt": "vllm:prompt_tokens_total", "generation": "vllm:generation_tokens_total",
        "e2e_s": "vllm:e2e_request_latency_seconds_sum", "queue_s": "vllm:request_queue_time_seconds_sum", "preempt": "vllm:num_preemptions_total",
        "prefill_s": "vllm:request_prefill_time_seconds_sum", "decode_s": "vllm:request_decode_time_seconds_sum"}
RES = {"receiver-1": "orchestrator (Gemma-4-31B-it)", "receiver-2": "tool/answer (GPT-OSS-20B)", "receiver-3": "comparison (gemma4-e4b)", "receiver-4": "safety (Kanana-8b)"}
counters = {}
for r in ROUNDS:
    for m in METHODS:
        first = {}; last = {}
        for line in (ROOT / f"main/round-{r}/{m}-metrics.jsonl").open():
            d = json.loads(line); first.setdefault(d["resource"], d); last[d["resource"]] = d
        counters[r, m] = {res: {k: (last[res]["metrics"].get(v, 0) - first[res]["metrics"].get(v, 0)) for k, v in KEYS.items()} | {"span_s": last[res]["time"] - first[res]["time"]} for res in first}
engine = []
for r in ROUNDS:
    for m in METHODS:
        row = {"round": r, "method": LABEL[m], "position": ORDER[r].index(m) + 1}
        for res in ["receiver-1", "receiver-2", "receiver-3", "receiver-4"]:
            c = counters[r, m].get(res, {}); w = wrapped[r, m].get(res, {"calls": 0, "prompt": 0, "completion": 0})
            row[res] = {"counter_requests": int(c.get("requests", 0)), "wrapped_calls": w["calls"], "native_requests": int(c.get("requests", 0)) - w["calls"],
                        "counter_prompt": int(c.get("prompt", 0)), "wrapped_prompt": w["prompt"], "native_prompt": int(c.get("prompt", 0)) - w["prompt"],
                        "counter_generation": int(c.get("generation", 0)), "wrapped_completion": w["completion"], "native_generation": int(c.get("generation", 0)) - w["completion"],
                        "e2e_s": float(c.get("e2e_s", 0)), "queue_s": float(c.get("queue_s", 0)), "preempt": int(c.get("preempt", 0)), "span_s": float(c.get("span_s", 0))}
        engine.append(row)
results["engine"] = engine
# worker-engine summary by method (mean over rounds) and removed-call estimate
w2 = {m: [next(e for e in engine if e["round"] == r and e["method"] == LABEL[m])["receiver-2"] for r in ROUNDS] for m in METHODS}
summary_w = {LABEL[m]: {k: float(np.mean([x[k] for x in w2[m]])) for k in ["counter_requests", "wrapped_calls", "native_requests", "counter_prompt", "native_prompt", "counter_generation", "native_generation", "e2e_s", "queue_s"]} for m in METHODS}
removed = {}
for cand in ["direct", "capability"]:
    per_round = [{k: w2["original"][i][k] - w2[cand][i][k] for k in ["counter_requests", "counter_prompt", "counter_generation", "native_requests", "native_prompt", "native_generation", "e2e_s"]} for i in range(3)]
    removed[LABEL[cand]] = {"per_round": per_round, "mean": {k: float(np.mean([p[k] for p in per_round])) for k in per_round[0]}}
results["worker_engine"] = {"by_method": summary_w, "original_minus_candidate": removed}
# outside-traffic check: safety engine requests per batch should equal 200; orchestrator counter vs wrapped
results["wrapped_role"] = {f"{r}-{LABEL[m]}": {k: dict(v) for k, v in wrapped_role[r, m].items()} for r in ROUNDS for m in METHODS}
results["traffic_check"] = [{"round": e["round"], "method": e["method"], "safety_requests": e["receiver-4"]["counter_requests"], "orchestrator_counter": e["receiver-1"]["counter_requests"], "orchestrator_wrapped": e["receiver-1"]["wrapped_calls"], "comparison_counter": e["receiver-3"]["counter_requests"], "comparison_wrapped": e["receiver-3"]["wrapped_calls"]} for e in engine]

# ---------- E. branch checks and intermediate endpoints ----------
bcheck = {}
for m in METHODS:
    n_b = sum(len(branch[r, m, q]["review_boundary"]) for r in ROUNDS for q in cells[r, m])
    eq = sum(1 for r in ROUNDS for q in cells[r, m] for e in branch[r, m, q]["review_handler"] if e.get("arguments_equal"))
    nh = [e["non_handler_ms"] for r in ROUNDS for q in cells[r, m] for e in branch[r, m, q]["review_boundary"] if e.get("status") == "ok"]
    bcheck[LABEL[m]] = {"branches": n_b, "arguments_equal": eq, "non_handler_ms_mean": float(np.mean(nh)), "handler_ms_mean": float(np.mean([e["handler_ms"] for r in ROUNDS for q in cells[r, m] for e in branch[r, m, q]["review_boundary"] if e.get("status") == "ok"]))}
results["branch_check"] = bcheck
def endpoints(r, m, q):
    rec = cells[r, m][q]; b = branch[r, m, q]
    if not b["review_boundary"] or not b["review_prepared"] or not valid(rec): return None
    t0 = client_ts(rec["started_at"]); tl = max(e["_t"] for e in b["review_boundary"]); tp = min(e["_t"] for e in b["review_prepared"])
    return {"E": tl - t0, "S": tl - tp, "T": rec["duration_ms"] / 1000}
endpoint = []
for r in ROUNDS:
    for cand in ["direct", "capability"]:
        rows = []
        for q in cells[r, "original"]:
            a = endpoints(r, "original", q); b = endpoints(r, cand, q)
            if a and b: rows.append((q, a, b))
        dE = np.array([a["E"] - b["E"] for _, a, b in rows]); dS = np.array([a["S"] - b["S"] for _, a, b in rows]); dT = np.array([a["T"] - b["T"] for _, a, b in rows])
        by_type = {}
        for t in TYPE.values():
            sel = np.array([qtype[q] == t for q, _, _ in rows])
            if sel.sum(): by_type[t] = {"n": int(sel.sum()), "S_saving": float(dS[sel].mean()), "E_saving": float(dE[sel].mean()), "T_saving": float(dT[sel].mean())}
        endpoint.append({"round": r, "candidate": LABEL[cand], "pairs": len(rows), "E_orig": float(np.mean([a["E"] for _, a, _ in rows])), "E_cand": float(np.mean([b["E"] for _, _, b in rows])),
                         "E_saving": float(dE.mean()), "E_ci": boot_ci(dE), "S_saving": float(dS.mean()), "S_ci": boot_ci(dS), "S_sd": float(dS.std(ddof=1)), "E_sd": float(dE.std(ddof=1)), "A_saving": float((dT - dE).mean()), "A_sd": float((dT - dE).std(ddof=1)), "T_saving": float(dT.mean()), "by_type": by_type})
# null control for endpoints: original round a vs b
endpoint_null = []
for a, b in [(1, 2), (1, 3), (2, 3)]:
    rows = []
    for q in cells[a, "original"]:
        x = endpoints(a, "original", q); y = endpoints(b, "original", q)
        if x and y: rows.append((q, x, y))
    dS = np.array([x["S"] - y["S"] for _, x, y in rows]); dE = np.array([x["E"] - y["E"] for _, x, y in rows])
    endpoint_null.append({"rounds": [a, b], "pairs": len(rows), "S_saving": float(dS.mean()), "S_ci": boot_ci(dS), "E_saving": float(dE.mean()), "E_ci": boot_ci(dE)})
results["endpoints"] = endpoint; results["endpoint_null"] = endpoint_null

OUT.write_text(json.dumps(results, indent=1, default=float))

# ---------- print ----------
def fmt(x, d=3): return f"{x:.{d}f}"
print("\n== A. Completed-pair latency (seconds); saving = original - candidate ==")
print(f"{'comparison':42s} {'n':>4} {'mean_a':>8} {'mean_b':>8} {'saving':>7} {'95% CI':>18} {'median':>7} {'pos/neg':>8} {'sign p':>8} {'SD':>6} {'geo %':>7} {'geo CI %':>16} {'trim10':>7} {'trim CI':>16}")
for s in lat + null + same:
    print(f"{s['label']:42s} {s['n']:4d} {fmt(s['mean_a']):>8} {fmt(s['mean_b']):>8} {fmt(s['mean_saving']):>7} [{fmt(s['ci'][0])},{fmt(s['ci'][1])}] {fmt(s['median']):>7} {s['pos']:>3}/{s['neg']:<4} {s['sign_p']:8.4f} {fmt(s['sd'],1):>6} {fmt(s['geo_change_pct'],2):>7} [{fmt((1-s['geo_ci'][1])*100,2)},{fmt((1-s['geo_ci'][0])*100,2)}] {fmt(s['trimmed10']):>7} [{fmt(s['trimmed10_ci'][0])},{fmt(s['trimmed10_ci'][1])}]")
print("\n== B. Position-balanced (Latin square) on common questions ==")
print(json.dumps(bal, indent=1))
print("\n== C. Exact-set / recall gap vs original-repeat (common cohort) ==")
print(json.dumps(setagree, indent=1))
print("\n== D. Worker engine (receiver-2) per method, mean over rounds ==")
print(json.dumps(results["worker_engine"], indent=1))
print("\n== D2. traffic check ==")
for t in results["traffic_check"]: print(t)
print("\n== E. Branch check ==")
print(json.dumps(bcheck, indent=1))
print("\n== E2. Endpoints ==")
for e in endpoint: print({k: v for k, v in e.items() if k != "by_type"}); print("   by type:", e["by_type"])
print("null:", endpoint_null)

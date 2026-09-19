"""Extended validation analysis: original vs direct (checked dispatch) vs overlap (direct + verified overlap of the
preparation call), 200 questions x 4 position-rotated rounds at concurrency 4. Reads client records, pod logs
(CNU_CAPABILITY_V1 / CNU_APP_ADAPTER_V1) and engine counters. Never prints questions or answers."""
import csv, json, math, re, statistics as st, sys
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); SUB = sys.argv[3] if len(sys.argv) > 3 else "main"
RNG = np.random.RandomState(20260919); B = 10000
ARMS = ["original", "direct", "overlap"]


def ts(s):
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\.(\d+)(Z|[+-]\d\d:\d\d)", s)
    return datetime.fromisoformat(m.group(1) + ("+00:00" if m.group(3) == "Z" else m.group(3))).timestamp() + float("0." + m.group(2))


def cts(s): return datetime.fromisoformat(s).timestamp()


def valid(r):
    resp = r.get("response")
    return r.get("status") == "ok" and isinstance(resp, dict) and isinstance(resp.get("laws"), list) and isinstance(resp.get("comment"), str)


def uniq(ids):
    out, seen = [], set()
    for v in ids:
        v = str(v or "").strip()
        if v and v not in seen: out.append(v); seen.add(v)
    return out


def sign_p(pos, neg):
    n, k = pos + neg, min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else None


def p95(x): x = sorted(x); return x[int(0.95 * (len(x) - 1))] if x else None


def summary(x):
    x = [v for v in x if v is not None]
    return {"n": len(x), "mean": st.mean(x), "p50": st.median(x), "p95": p95(x)} if x else {"n": 0}


def fmt(v, spec=".3f"): return "-" if v is None else format(v, spec)


# ---------- cells ----------
status = json.loads((ROOT / SUB / "status.json").read_text())
cells, order = {}, defaultdict(list)
for c in status["completed"]:
    p = ROOT / SUB / f"round-{c['round']}" / "responses" / "review-trial" / c["method"] / "client_requests.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    cells[c["round"], c["method"]] = {x["question_id"]: x for x in rows}; order[c["round"]].append(c["method"])
rounds = sorted({r for r, _ in cells})
print("cells:", [(k, len(v), sum(valid(x) for x in v.values())) for k, v in cells.items()])
windows = {k: (min(cts(x["started_at"]) for x in v.values()) - 2, max(cts(x["completed_at"]) for x in v.values()) + 2) for k, v in cells.items()}


def load(arm):
    ev = []
    for line in (ROOT / f"{arm}-application.log").open(errors="replace"):
        i = line.find(' {"event"')
        if i < 0: continue
        try:
            e = json.loads(line[i + 1:]); e["_t"] = ts(line[:40]); ev.append(e)
        except Exception:
            pass
    return ev


arms = [a for a in ARMS if any(m == a for _, m in cells)]; traces = {a: load(a) for a in arms}


def cell_of(arm, t):
    for (r, m), (lo, hi) in windows.items():
        if m == arm and lo <= t <= hi: return r


def qid(case): return re.sub(r"-r\d+-c\d+$", "", case or "")


result = {"design": {"concurrency": 4, "rounds": rounds, "order": {str(r): order[r] for r in rounds}},
          "cells": [], "latency": [], "overlap": {}, "decisions": [], "boundary": {}, "calls": {}, "engine": [], "mapping": {}}

# ---------- request-id mapping and per-request trace rows ----------
casemap = {}; reqcalls = defaultdict(list); overlap_events = defaultdict(list); decision = {}; boundaries = defaultdict(list)
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
    for e in traces[arm]:
        r = cell_of(arm, e["_t"])
        if r is None: continue
        kind = e.get("event")
        if kind == "call_overlap":
            q = casemap.get((r, arm, e.get("request_key"))); overlap_events[r, arm, q].append(e)
        elif kind in ("classify_result", "preparation_result"):
            q = casemap.get((r, arm, e.get("request_id")))
            if q is not None: decision.setdefault((r, arm, q), {})[kind] = e.get("output_hash")
        elif kind == "review_boundary":
            q = casemap.get((r, arm, e.get("request_id")))
            if q is not None: boundaries[r, arm, q].append(e)
for (r, m), v in cells.items():
    result["mapping"][f"{r}-{m}"] = {"questions": len(v), "with_calls": sum(1 for q in v if (r, m, q) in reqcalls),
                                     "with_decisions": sum(1 for q in v if (r, m, q) in decision), "with_boundary": sum(1 for q in v if (r, m, q) in boundaries)}

# ---------- cell stats ----------
for (r, m), v in sorted(cells.items()):
    recs = list(v.values()); lat = [x["duration_ms"] / 1000 for x in recs]; ok = [x for x in recs if valid(x)]
    t0 = min(cts(x["started_at"]) for x in recs); t1 = max(cts(x["completed_at"]) for x in recs)
    result["cells"].append({"round": r, "arm": m, "position": order[r].index(m) + 1, "n": len(recs), "valid": len(ok), "failures": len(recs) - len(ok),
                            "mean_s": st.mean(lat), "p50_s": st.median(lat), "p95_s": p95(lat), "wall_s": t1 - t0, "valid_per_s": len(ok) / (t1 - t0)})


# ---------- paired latency ----------
def pair_rows(a, b):
    A, Bc = cells[a], cells[b]
    return [(q, A[q], Bc[q]) for q in A if q in Bc and valid(A[q]) and valid(Bc[q])]


def paired(pairs, label, kind):
    if not pairs: return None
    qs = [q for q, _, _ in pairs]
    pa = np.array([x["duration_ms"] / 1000 for _, x, _ in pairs]); pb = np.array([y["duration_ms"] / 1000 for _, _, y in pairs]); d = pa - pb
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    groups = defaultdict(list)
    for i, q in enumerate(qs): groups[q].append(i)
    keys = list(groups); idx = RNG.randint(0, len(keys), size=(4000, len(keys)))
    boots = []
    for row in idx:
        sel = np.concatenate([groups[keys[k]] for k in row])
        boots.append((d[sel].mean(), 1 - pb[sel].mean() / pa[sel].mean(), np.percentile(pb[sel], 95) / np.percentile(pa[sel], 95)))
    boots = np.array(boots)
    rec, same = [], []
    for q, x, y in pairs:
        g, p = set(uniq(x["law_ids"])), set(uniq(y["law_ids"]))
        if g: rec.append(len(g & p) / len(g))
        same.append(g == p)
    row = {"pair": label, "kind": kind, "pairs": len(pairs), "questions": len(keys), "mean_a": float(pa.mean()), "mean_b": float(pb.mean()),
           "saving": float(d.mean()), "ci": [float(np.percentile(boots[:, 0], 2.5)), float(np.percentile(boots[:, 0], 97.5))], "pct": float(d.mean() / pa.mean() * 100),
           "median": float(np.median(d)), "pos": pos, "neg": neg, "sign_p": sign_p(pos, neg), "sd": float(d.std(ddof=1)) if len(d) > 1 else None,
           "mean_gain_lower": float(np.percentile(boots[:, 1], 2.5)), "mean_gain_upper": float(np.percentile(boots[:, 1], 97.5)),
           "p95_ratio_point": float(np.percentile(pb, 95) / np.percentile(pa, 95)), "p95_ratio_upper": float(np.percentile(boots[:, 2], 97.5)),
           "nonempty_recall": float(np.mean(rec)) if rec else None, "n_nonempty": len(rec), "exact_set": float(np.mean(same))}
    result["latency"].append(row); return row


contrasts = [("original", "direct"), ("direct", "overlap"), ("original", "overlap")]
for a, b in contrasts:
    for r in rounds:
        if (r, a) in cells and (r, b) in cells: paired(pair_rows((r, a), (r, b)), f"round {r}: {a} vs {b}", "within_round")
    pooled = []
    for r in rounds:
        if (r, a) in cells and (r, b) in cells: pooled += pair_rows((r, a), (r, b))
    paired(pooled, f"pooled: {a} vs {b}", "pooled")
for a in arms:
    pooled = []
    for r1, r2 in [(1, 2), (3, 4), (1, 3), (2, 4)]:
        if (r1, a) in cells and (r2, a) in cells: pooled += pair_rows((r1, a), (r2, a))
    paired(pooled, f"null: {a} across rounds", "null")

# ---------- overlap events (overlap arm) ----------
for (r, m), v in sorted(cells.items()):
    evs = [e for (rr, mm, q), es in overlap_events.items() if rr == r and mm == m for e in es]
    if not evs: continue
    prepared = [e for e in evs if e.get("prepared") is True]; refused = Counter(e.get("reason") for e in evs if e.get("prepared") is False)
    consumed = [e for e in evs if "reused" in e]; reused = [e for e in consumed if e["reused"]]
    dropped = [e for e in evs if e.get("discarded")]
    per_q = defaultdict(lambda: {"reused": 0, "fallback": 0})
    for (rr, mm, q), es in overlap_events.items():
        if rr == r and mm == m and q in v:
            for e in es:
                if "reused" in e: per_q[q]["reused" if e["reused"] else "fallback"] += 1
    result["overlap"][f"{r}-{m}"] = {"prepared": len(prepared), "refused": dict(refused), "consumed": len(consumed), "reused": len(reused),
                                     "consume_reasons": dict(Counter(e["reason"] for e in consumed)), "discarded": len(dropped),
                                     "discarded_reasons": dict(Counter(e["reason"] for e in dropped)),
                                     "wasted_completed": sum(1 for e in dropped if e.get("completed_before_discard")),
                                     "questions_reused": sum(1 for q in per_q.values() if q["reused"]), "questions_fallback_only": sum(1 for q in per_q.values() if q["fallback"] and not q["reused"]),
                                     "head_start_ms": summary([e.get("head_start_ms") for e in reused]), "call_ms": summary([e.get("call_ms") for e in reused]),
                                     "residual_wait_ms": summary([e.get("residual_wait_ms") for e in reused])}

# ---------- decision agreement (classify / preparation output hashes) ----------
def decision_pair(a, b, label):
    n = {"pairs": 0, "both_classify": 0, "classify_equal": 0, "both_preparation": 0, "preparation_equal": 0}
    for q in cells[a]:
        if q not in cells[b] or not (valid(cells[a][q]) and valid(cells[b][q])): continue
        n["pairs"] += 1
        da, db = decision.get((a[0], a[1], q), {}), decision.get((b[0], b[1], q), {})
        for kind in ("classify", "preparation"):
            x, y = da.get(kind + "_result"), db.get(kind + "_result")
            if x and y:
                n["both_" + kind] += 1; n[kind + "_equal"] += x == y
    result["decisions"].append({"pair": label, **n})


for a, b in contrasts:
    for r in rounds:
        if (r, a) in cells and (r, b) in cells: decision_pair((r, a), (r, b), f"round {r}: {a} vs {b}")
for a in arms:
    for r1, r2 in [(1, 2), (3, 4)]:
        if (r1, a) in cells and (r2, a) in cells: decision_pair((r1, a), (r2, a), f"null: {a} r{r1} vs r{r2}")

# ---------- dispatch boundary per cell ----------
for (r, m), v in sorted(cells.items()):
    bl = [e for (rr, mm, q), es in boundaries.items() if rr == r and mm == m and q in v for e in es]
    ok = [e for e in bl if e.get("status") == "ok"]
    result["boundary"][f"{r}-{m}"] = {"branches": len(bl), "effective_method": dict(Counter(e.get("effective_method") for e in bl)),
                                      "non_handler_ms": summary([e.get("non_handler_ms") for e in ok]), "handler_ms": summary([e.get("handler_ms") for e in ok]),
                                      "handler_calls": dict(Counter(e.get("handler_calls") for e in ok))}

# ---------- LLM calls per cell ----------
for (r, m), v in cells.items():
    agg = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "api_ms": 0.0}); per_request = []; master = []
    for q in v:
        calls = reqcalls.get((r, m, q), [])
        if valid(v[q]): per_request.append(len(calls)); master.append(sum(1 for c in calls if c.get("role") == "master"))
        for c in calls:
            k = f"{c.get('model')}/{c.get('role')}"; a = agg[k]; a["calls"] += 1; a["prompt"] += c.get("prompt_tokens") or 0; a["completion"] += c.get("completion_tokens") or 0; a["api_ms"] += c.get("api_ms") or 0
    result["calls"][f"{r}-{m}"] = {"calls_per_valid_request": summary(per_request), "master_calls_per_valid_request": summary(master),
                                   "by_role": {k: {"calls": x["calls"], "prompt_tokens": x["prompt"], "completion_tokens": x["completion"], "mean_api_ms": x["api_ms"] / x["calls"] if x["calls"] else None} for k, x in agg.items()}}

# ---------- engine counters per cell ----------
KEYS = {"requests": "vllm:e2e_request_latency_seconds_count", "queue_s": "vllm:request_queue_time_seconds_sum", "prompt": "vllm:prompt_tokens_total", "generation": "vllm:generation_tokens_total", "preempt": "vllm:num_preemptions_total"}
for (r, m) in cells:
    p = ROOT / SUB / f"round-{r}" / f"{m}-metrics.jsonl"
    if not p.exists(): continue
    first, last, run = {}, {}, defaultdict(list)
    for line in p.open():
        d = json.loads(line)
        if d.get("metrics"):
            first.setdefault(d["resource"], d); last[d["resource"]] = d; run[d["resource"]].append(d["metrics"].get("vllm:num_requests_running", 0))
    result["engine"].append({"round": r, "arm": m, **{res: {**{k: last[res]["metrics"].get(v, 0) - first[res]["metrics"].get(v, 0) for k, v in KEYS.items()}, "running_mean": st.mean(run[res]) if run[res] else None} for res in first}})

# ---------- per-request table ----------
with OUT.with_suffix(".requests.csv").open("w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["question_id", "round", "arm", "position", "status", "valid", "duration_ms", "law_ids", "master_calls", "overlap_reused", "overlap_head_start_ms"])
    for (r, m), v in sorted(cells.items()):
        for q, x in sorted(v.items()):
            calls = reqcalls.get((r, m, q), []); evs = [e for e in overlap_events.get((r, m, q), []) if "reused" in e]
            reused = next((e for e in evs if e["reused"]), None)
            w.writerow([q, r, m, order[r].index(m) + 1, x.get("status"), int(valid(x)), x.get("duration_ms"), len(uniq(x.get("law_ids") or [])),
                        sum(1 for c in calls if c.get("role") == "master"), None if not evs else int(reused is not None), None if reused is None else round(reused.get("head_start_ms") or 0, 1)])

OUT.write_text(json.dumps(result, indent=1, default=float) + "\n")
for c in result["cells"]: print(f"cell r{c['round']} pos{c['position']} {c['arm']:9s} valid {c['valid']}/{c['n']} mean {c['mean_s']:.2f} p50 {c['p50_s']:.2f} p95 {c['p95_s']:.1f} wall {c['wall_s']:.0f}s")
for k, v in result["mapping"].items(): print("mapping", k, v)
for x in result["latency"]: print(f"{x['pair']:34s} n={x['pairs']} {x['mean_a']:.2f} vs {x['mean_b']:.2f} saving {x['saving']:.3f}s ({x['pct']:.1f}%) CI [{fmt(x['ci'][0])},{fmt(x['ci'][1])}] median {x['median']:.3f} {x['pos']}/{x['neg']} p={fmt(x['sign_p'])} gain95 [{x['mean_gain_lower']*100:.1f}%,{x['mean_gain_upper']*100:.1f}%] p95r {x['p95_ratio_point']:.3f} (upper {x['p95_ratio_upper']:.3f}) recall {fmt(None if x['nonempty_recall'] is None else x['nonempty_recall']*100, '.1f')} exact {x['exact_set']*100:.1f}")
for k, v in result["overlap"].items():
    f = lambda s: f"{s['mean']:.0f}/{s['p50']:.0f}/{s['p95']:.0f}" if s.get("n") else "-"
    print(f"overlap {k}: prepared {v['prepared']} refused {v['refused']} reused {v['reused']}/{v['consumed']} reasons {v['consume_reasons']} discarded {v['discarded']} {v['discarded_reasons']} wasted_completed {v['wasted_completed']} | head_start {f(v['head_start_ms'])} call {f(v['call_ms'])} residual {f(v['residual_wait_ms'])}")
for x in result["decisions"]: print("decisions", x)
for k, v in result["boundary"].items(): print(f"boundary {k}: {v['branches']} branches {v['effective_method']} non_handler_ms {fmt(v['non_handler_ms'].get('mean'), '.2f')} handler_calls {v['handler_calls']}")
for k, v in result["calls"].items(): print("calls", k, "per request", round(v["calls_per_valid_request"].get("mean", 0), 2), "master", round(v["master_calls_per_valid_request"].get("mean", 0), 2), {kk: (vv["calls"], round(vv["mean_api_ms"] or 0)) for kk, vv in v["by_role"].items()})
for e in result["engine"]: print("engine", {k: (v if not isinstance(v, dict) else {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()}) for k, v in e.items()})

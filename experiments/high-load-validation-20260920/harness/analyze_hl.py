"""High-load validation analysis: original vs direct (checked dispatch), 200 questions per cell, several
concurrency levels, rotated rounds per level. Reads client records (<root>/main/level-<C>/round-<r>/...),
pod logs (CNU_CAPABILITY_V1 / CNU_APP_ADAPTER_V1) and engine counters; evaluates the release gate on the
measured loads. Never prints questions or answers."""
import csv, json, math, re, statistics as st, sys
from collections import defaultdict, Counter
from datetime import datetime
from pathlib import Path
import numpy as np

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); SUB = sys.argv[3] if len(sys.argv) > 3 else "main"
PKG = sys.argv[4] if len(sys.argv) > 4 else None
RNG = np.random.RandomState(20260920); B = 4000
ARMS = ["original", "direct"]


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
    p = ROOT / SUB / f"level-{c['level']:03d}" / f"round-{c['round']}" / "responses" / "sweep-trial" / c["method"] / "client_requests.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    cells[c["level"], c["round"], c["method"]] = {x["question_id"]: x for x in rows}; order[c["level"], c["round"]].append(c["method"])
levels = sorted({l for l, _, _ in cells})
print("cells:", [(k, len(v), sum(valid(x) for x in v.values())) for k, v in cells.items()])
windows = {k: (min(cts(x["started_at"]) for x in v.values()) - 2, max(cts(x["completed_at"]) for x in v.values()) + 2) for k, v in cells.items()}


def load(arm):
    ev = []
    p = ROOT / f"{arm}-application.log"
    if not p.exists(): return ev
    for line in p.open(errors="replace"):
        i = line.find(' {"event"')
        if i < 0: continue
        try:
            e = json.loads(line[i + 1:]); e["_t"] = ts(line[:40]); ev.append(e)
        except Exception:
            pass
    return ev


arms = [a for a in ARMS if any(m == a for _, _, m in cells)]; traces = {a: load(a) for a in arms}


def cell_of(arm, t):
    for (l, r, m), (lo, hi) in windows.items():
        if m == arm and lo <= t <= hi: return (l, r)


def qid(case): return re.sub(r"-r\d+-c\d+$", "", case or "")


result = {"design": {"levels": levels, "rounds": sorted({r for _, r, _ in cells}), "order": {f"{l}-{r}": v for (l, r), v in order.items()}},
          "cells": [], "latency": [], "boundary": {}, "calls": {}, "engine": [], "gate": {}, "mapping": {}}

# ---------- trace mapping ----------
casemap = {}; reqcalls = defaultdict(list); boundaries = defaultdict(list)
for arm in arms:
    for e in traces[arm]:
        c = cell_of(arm, e["_t"])
        if c is None: continue
        if e.get("event") == "request_finished":
            q = qid(e.get("case_id"))
            if q in cells.get((c[0], c[1], arm), {}):
                for x in e.get("calls", []):
                    if x.get("application_request_id"): casemap[c, arm, x["application_request_id"]] = q
                    reqcalls[c, arm, q].append(x)
    for e in traces[arm]:
        c = cell_of(arm, e["_t"])
        if c is None or e.get("event") != "review_boundary": continue
        q = casemap.get((c, arm, e.get("request_id")))
        if q is not None: boundaries[c, arm, q].append(e)
for (l, r, m), v in cells.items():
    result["mapping"][f"{l}-{r}-{m}"] = {"questions": len(v), "with_calls": sum(1 for q in v if ((l, r), m, q) in reqcalls), "with_boundary": sum(1 for q in v if ((l, r), m, q) in boundaries)}

# ---------- cell stats ----------
for (l, r, m), v in sorted(cells.items()):
    recs = list(v.values()); lat = [x["duration_ms"] / 1000 for x in recs]; ok = [x for x in recs if valid(x)]
    t0 = min(cts(x["started_at"]) for x in recs); t1 = max(cts(x["completed_at"]) for x in recs)
    result["cells"].append({"level": l, "round": r, "arm": m, "position": order[l, r].index(m) + 1, "n": len(recs), "valid": len(ok), "failures": len(recs) - len(ok),
                            "mean_s": st.mean(lat), "p50_s": st.median(lat), "p95_s": p95(lat), "wall_s": t1 - t0, "valid_per_s": len(ok) / (t1 - t0),
                            "slo60": sum(1 for x in ok if x["duration_ms"] <= 60000) / len(recs)})


# ---------- paired latency ----------
def pair_rows(a, b):
    A, Bc = cells[a], cells[b]
    return [(q, A[q], Bc[q]) for q in A if q in Bc and valid(A[q]) and valid(Bc[q])]


def paired(pairs, label, kind, level):
    if not pairs: return None
    qs = [q for q, _, _ in pairs]
    pa = np.array([x["duration_ms"] / 1000 for _, x, _ in pairs]); pb = np.array([y["duration_ms"] / 1000 for _, _, y in pairs]); d = pa - pb
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    groups = defaultdict(list)
    for i, q in enumerate(qs): groups[q].append(i)
    keys = list(groups); idx = RNG.randint(0, len(keys), size=(B, len(keys)))
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
    row = {"pair": label, "kind": kind, "level": level, "pairs": len(pairs), "questions": len(keys), "mean_a": float(pa.mean()), "mean_b": float(pb.mean()),
           "saving": float(d.mean()), "ci": [float(np.percentile(boots[:, 0], 2.5)), float(np.percentile(boots[:, 0], 97.5))], "pct": float(d.mean() / pa.mean() * 100),
           "median": float(np.median(d)), "pos": pos, "neg": neg, "sign_p": sign_p(pos, neg), "sd": float(d.std(ddof=1)) if len(d) > 1 else None,
           "mean_gain_lower": float(np.percentile(boots[:, 1], 2.5)), "mean_gain_upper": float(np.percentile(boots[:, 1], 97.5)),
           "p95_ratio_point": float(np.percentile(pb, 95) / np.percentile(pa, 95)), "p95_ratio_upper": float(np.percentile(boots[:, 2], 97.5)),
           "nonempty_recall": float(np.mean(rec)) if rec else None, "n_nonempty": len(rec), "exact_set": float(np.mean(same))}
    result["latency"].append(row); return row


pooled_rows = {}
for l in levels:
    rounds = sorted({r for ll, r, _ in cells if ll == l})
    pooled = []
    for r in rounds:
        if (l, r, "original") in cells and (l, r, "direct") in cells:
            paired(pair_rows((l, r, "original"), (l, r, "direct")), f"C{l} round {r}: original vs direct", "within_round", l)
            pooled += pair_rows((l, r, "original"), (l, r, "direct"))
    pooled_rows[l] = paired(pooled, f"C{l} pooled: original vs direct", "pooled", l)
    for a in arms:
        null = []
        for r1, r2 in [(1, 2), (3, 4), (1, 3), (2, 4)]:
            if (l, r1, a) in cells and (l, r2, a) in cells: null += pair_rows((l, r1, a), (l, r2, a))
        paired(null, f"C{l} null: {a} across rounds", "null", l)

# ---------- dispatch boundary and calls per cell ----------
for (l, r, m), v in sorted(cells.items()):
    bl = [e for ((ll, rr), mm, q), es in boundaries.items() if (ll, rr, mm) == (l, r, m) and q in v for e in es]
    ok = [e for e in bl if e.get("status") == "ok"]
    result["boundary"][f"{l}-{r}-{m}"] = {"branches": len(bl), "effective_method": dict(Counter(e.get("effective_method") for e in bl)),
                                          "non_handler_ms": summary([e.get("non_handler_ms") for e in ok]), "handler_ms": summary([e.get("handler_ms") for e in ok]),
                                          "handler_calls": dict(Counter(e.get("handler_calls") for e in ok))}
    agg = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "api_ms": 0.0, "wait_ms": 0.0}); per_request = []
    for q in v:
        calls = reqcalls.get(((l, r), m, q), [])
        if valid(v[q]): per_request.append(len(calls))
        for c in calls:
            k = f"{c.get('model')}/{c.get('role')}"; a = agg[k]; a["calls"] += 1; a["prompt"] += c.get("prompt_tokens") or 0; a["completion"] += c.get("completion_tokens") or 0; a["api_ms"] += c.get("api_ms") or 0; a["wait_ms"] += c.get("wait_ms") or 0
    result["calls"][f"{l}-{r}-{m}"] = {"calls_per_valid_request": summary(per_request), "by_role": {k: {"calls": x["calls"], "prompt_tokens": x["prompt"], "completion_tokens": x["completion"], "mean_api_ms": x["api_ms"] / x["calls"] if x["calls"] else None, "mean_wait_ms": x["wait_ms"] / x["calls"] if x["calls"] else None} for k, x in agg.items()}}

# ---------- engine counters per cell ----------
KEYS = {"requests": "vllm:e2e_request_latency_seconds_count", "queue_s": "vllm:request_queue_time_seconds_sum", "prompt": "vllm:prompt_tokens_total", "generation": "vllm:generation_tokens_total", "preempt": "vllm:num_preemptions_total"}
for (l, r, m) in cells:
    p = ROOT / SUB / f"level-{l:03d}" / f"round-{r}" / f"{m}-metrics.jsonl"
    if not p.exists(): continue
    first, last, run, kv = {}, {}, defaultdict(list), defaultdict(list)
    for line in p.open():
        d = json.loads(line)
        if d.get("metrics"):
            first.setdefault(d["resource"], d); last[d["resource"]] = d
            run[d["resource"]].append(d["metrics"].get("vllm:num_requests_running", 0)); kv[d["resource"]].append(d["metrics"].get("vllm:gpu_cache_usage_perc", d["metrics"].get("vllm:kv_cache_usage_perc", 0)))
    result["engine"].append({"level": l, "round": r, "arm": m, **{res: {**{k: last[res]["metrics"].get(v, 0) - first[res]["metrics"].get(v, 0) for k, v in KEYS.items()}, "running_mean": st.mean(run[res]) if run[res] else None, "kv_max": max(kv[res]) if kv[res] else None} for res in first}})

# ---------- release gate on the measured loads ----------
try:
    if PKG: sys.path.insert(0, PKG)
    from cnu_rag_optimization.performance_gate import LoadEvidence, PerformanceThresholds, evaluate_release_gate
    loads = []
    for l, row in pooled_rows.items():
        if row is None: continue
        cell_rows = [c for c in result["cells"] if c["level"] == l]
        loads.append(LoadEvidence(concurrency=l, unique_questions=200, rounds=len({c["round"] for c in cell_rows}), paired_complete=True,
                                  original_failures=sum(c["failures"] for c in cell_rows if c["arm"] == "original"),
                                  candidate_failures=sum(c["failures"] for c in cell_rows if c["arm"] == "direct"),
                                  mean_gain_lower=row["mean_gain_lower"], p95_ratio_upper=row["p95_ratio_upper"], scope="end_to_end",
                                  question_and_round_resampling=False))
    thresholds = PerformanceThresholds()
    gate = evaluate_release_gate(loads, {}, thresholds=thresholds, thresholds_predeclared=True, execution_equivalence_verified=False)
    result["gate"] = {"thresholds": {"loads": thresholds.loads, "primary_load": thresholds.primary_load, "minimum_primary_gain": thresholds.minimum_primary_gain,
                                     "maximum_secondary_regression": thresholds.maximum_secondary_regression, "maximum_p95_ratio": thresholds.maximum_p95_ratio, "minimum_rounds": thresholds.minimum_rounds},
                      "evidence": [{"concurrency": e.concurrency, "rounds": e.rounds, "mean_gain_lower": e.mean_gain_lower, "p95_ratio_upper": e.p95_ratio_upper,
                                    "original_failures": e.original_failures, "candidate_failures": e.candidate_failures} for e in loads],
                      "result": gate}
except Exception as exc:
    result["gate"] = {"error": repr(exc)}

# ---------- per-request table ----------
with OUT.with_suffix(".requests.csv").open("w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["question_id", "level", "round", "arm", "position", "status", "valid", "duration_ms", "law_ids", "search_branches", "non_handler_ms_sum"])
    for (l, r, m), v in sorted(cells.items()):
        for q, x in sorted(v.items()):
            bl = boundaries.get(((l, r), m, q), [])
            w.writerow([q, l, r, m, order[l, r].index(m) + 1, x.get("status"), int(valid(x)), x.get("duration_ms"), len(uniq(x.get("law_ids") or [])), len(bl),
                        round(sum(e.get("non_handler_ms") or 0 for e in bl), 1) if bl else None])

OUT.write_text(json.dumps(result, indent=1, default=float) + "\n")
for c in result["cells"]: print(f"cell C{c['level']:<3d} r{c['round']} pos{c['position']} {c['arm']:9s} valid {c['valid']}/{c['n']} mean {c['mean_s']:.2f} p50 {c['p50_s']:.2f} p95 {c['p95_s']:.1f} wall {c['wall_s']:.0f}s {c['valid_per_s']:.3f}/s slo60 {c['slo60']*100:.0f}%")
for k, v in result["mapping"].items():
    if v["with_calls"] != v["questions"]: print("mapping", k, v)
for x in result["latency"]: print(f"{x['pair']:40s} n={x['pairs']} {x['mean_a']:.2f} vs {x['mean_b']:.2f} saving {x['saving']:.3f}s ({x['pct']:.1f}%) CI [{fmt(x['ci'][0])},{fmt(x['ci'][1])}] median {x['median']:.3f} {x['pos']}/{x['neg']} p={fmt(x['sign_p'])} gain95 [{x['mean_gain_lower']*100:.1f}%,{x['mean_gain_upper']*100:.1f}%] p95r {x['p95_ratio_point']:.3f} (upper {x['p95_ratio_upper']:.3f}) recall {fmt(None if x['nonempty_recall'] is None else x['nonempty_recall']*100, '.1f')} exact {x['exact_set']*100:.1f}")
for k, v in result["boundary"].items(): print(f"boundary {k}: {v['branches']} branches {v['effective_method']} non_handler_ms mean {fmt(v['non_handler_ms'].get('mean'), '.1f')} p95 {fmt(v['non_handler_ms'].get('p95'), '.1f')} handler_calls {v['handler_calls']}")
for k, v in result["calls"].items(): print("calls", k, "per request", round(v["calls_per_valid_request"].get("mean", 0), 2), {kk: (vv["calls"], round(vv["mean_api_ms"] or 0), round(vv["mean_wait_ms"] or 0)) for kk, vv in v["by_role"].items()})
for e in result["engine"]: print("engine", {k: (v if not isinstance(v, dict) else {kk: (round(vv, 2) if isinstance(vv, float) else vv) for kk, vv in v.items()}) for k, v in e.items()})
print("gate", json.dumps(result["gate"], default=float)[:1500])

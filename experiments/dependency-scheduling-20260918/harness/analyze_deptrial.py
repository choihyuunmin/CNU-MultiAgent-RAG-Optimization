"""Dependency-scheduling trial analysis: barrier (original join) vs ready (start-when-ready) at concurrency 4,
two position-balanced rounds. Reads the client records, the pod logs (CNU_PROGRAM_HARNESS_V1 / CNU_APP_ADAPTER_V1)
and the engine counters. Never prints questions or answers."""
import json, math, re, statistics as st, sys
from collections import defaultdict, Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(sys.argv[1]); OUT = Path(sys.argv[2]); RNG = np.random.RandomState(20260918); B = 10000
SUB = sys.argv[3] if len(sys.argv) > 3 else "main"


def ts(s):
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\.(\d+)(Z|[+-]\d\d:\d\d)", s)
    base = datetime.fromisoformat(m.group(1) + ("+00:00" if m.group(3) == "Z" else m.group(3))).timestamp()
    return base + float("0." + m.group(2))


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


def ci(x):
    x = np.asarray(x, float)
    if len(x) == 0: return [None, None]
    idx = RNG.randint(0, len(x), size=(B, len(x))); v = x[idx].mean(axis=1)
    return [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]


def p95(x): x = sorted(x); return x[int(0.95 * (len(x) - 1))] if x else None


def summary(x):
    x = [v for v in x if v is not None]
    return {"n": len(x), "mean": st.mean(x) if x else None, "p50": st.median(x) if x else None, "p95": p95(x)} if x else {"n": 0}


# ---------- cells ----------
status = json.loads((ROOT / SUB / "status.json").read_text())
cells = {}
for c in status["completed"]:
    p = ROOT / SUB / f"round-{c['round']}" / "responses" / "review-trial" / c["method"] / "client_requests.jsonl"
    rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
    cells[c["round"], c["method"]] = {x["question_id"]: x for x in rows}
print("cells:", [(k, len(v), sum(valid(x) for x in v.values())) for k, v in cells.items()])
windows = {k: (min(cts(x["started_at"]) for x in v.values()) - 2, max(cts(x["completed_at"]) for x in v.values()) + 2) for k, v in cells.items()}


def load(arm):
    ev = []
    for line in (ROOT / f"{arm}-application.log").open(errors="replace"):
        i = line.find(' {"event"')
        if i < 0: continue
        try:
            e = json.loads(line[i + 1:]); e["_t"] = ts(line[:40]); e["_kind"] = line[:i].split()[-1]; ev.append(e)
        except Exception:
            pass
    return ev


arms = sorted({m for _, m in cells}); traces = {a: load(a) for a in arms}


def cell_of(arm, t):
    for (r, m), (lo, hi) in windows.items():
        if m == arm and lo <= t <= hi: return r


def qid(case): return re.sub(r"-r\d+-c\d+$", "", case or "")


result = {"design": {"concurrency": 4, "rounds": sorted({r for r, _ in cells}), "order": {str(r): [c["method"] for c in status["completed"] if c["round"] == r] for r in {r for r, _ in cells}}},
          "cells": [], "latency": [], "nodes": {}, "hashes": [], "calls": {}, "engine": [], "mapping": {}}

# ---------- request-id mapping and per-request trace rows ----------
casemap = {}; reqcalls = defaultdict(list); workflows = defaultdict(list); reads = {}; handoffs = {}
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
            casemap[r, arm, e["application_request_id"]] = qid(e.get("case_id"))
    for e in traces[arm]:
        r = cell_of(arm, e["_t"])
        if r is None: continue
        q = casemap.get((r, arm, e.get("request_id")))
        if e.get("event") == "workflow_closed":
            workflows[r, arm, q].append(e)
        elif e.get("event") == "program_read_result":
            reads[r, arm, e.get("workflow_id")] = e
        elif e.get("event") == "program_search_result":
            handoffs[r, arm, e.get("workflow_id")] = e
for (r, m), v in cells.items():
    wf = [e for (rr, mm, q), es in workflows.items() if rr == r and mm == m for e in es]
    mapped = sum(1 for (rr, mm, q), es in workflows.items() if rr == r and mm == m and q in v for _ in es)
    result["mapping"][f"{r}-{m}"] = {"workflow_rows": len(wf), "mapped_to_question": mapped, "questions_with_workflow": len({q for (rr, mm, q) in workflows if rr == r and mm == m and q in v}),
                                     "read_rows": sum(1 for (rr, mm, _) in reads if rr == r and mm == m), "handoff_rows": sum(1 for (rr, mm, _) in handoffs if rr == r and mm == m)}

# ---------- cell stats ----------
for (r, m), v in sorted(cells.items()):
    recs = list(v.values()); lat = [x["duration_ms"] / 1000 for x in recs]; ok = [x for x in recs if valid(x)]
    t0 = min(cts(x["started_at"]) for x in recs); t1 = max(cts(x["completed_at"]) for x in recs)
    result["cells"].append({"round": r, "arm": m, "position": result["design"]["order"][str(r)].index(m) + 1, "n": len(recs), "valid": len(ok), "failures": len(recs) - len(ok),
                            "mean_s": st.mean(lat), "p50_s": st.median(lat), "p95_s": p95(lat), "wall_s": t1 - t0, "valid_per_s": len(ok) / (t1 - t0)})


# ---------- paired latency ----------
def pair_rows(a, b):
    A, Bc = cells[a], cells[b]
    return [(q, A[q], Bc[q]) for q in A if q in Bc and valid(A[q]) and valid(Bc[q])]


def paired(pairs, label, kind):
    qs = [q for q, _, _ in pairs]
    pa = np.array([x["duration_ms"] / 1000 for _, x, _ in pairs]); pb = np.array([y["duration_ms"] / 1000 for _, _, y in pairs]); d = pa - pb
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    # question-cluster bootstrap (a question may appear in several rounds)
    groups = defaultdict(list)
    for i, q in enumerate(qs): groups[q].append(i)
    keys = list(groups); idx = RNG.randint(0, len(keys), size=(B, len(keys)))
    boots = []
    for row in idx[:4000]:
        sel = np.concatenate([groups[keys[k]] for k in row]); boots.append((d[sel].mean(), 1 - pb[sel].mean() / pa[sel].mean(), np.percentile(pb[sel], 95) / np.percentile(pa[sel], 95)))
    boots = np.array(boots)
    rec, same = [], []
    for q, x, y in pairs:
        g, p = set(uniq(x["law_ids"])), set(uniq(y["law_ids"]))
        if g: rec.append(len(g & p) / len(g))
        same.append(g == p)
    row = {"pair": label, "kind": kind, "pairs": len(pairs), "questions": len(keys), "mean_a": float(pa.mean()), "mean_b": float(pb.mean()), "saving": float(d.mean()),
           "ci": [float(np.percentile(boots[:, 0], 2.5)), float(np.percentile(boots[:, 0], 97.5))], "pct": float(d.mean() / pa.mean() * 100),
           "median": float(np.median(d)), "pos": pos, "neg": neg, "sign_p": sign_p(pos, neg), "sd": float(d.std(ddof=1)) if len(d) > 1 else None,
           "mean_gain_lower": float(np.percentile(boots[:, 1], 2.5)), "mean_gain_upper": float(np.percentile(boots[:, 1], 97.5)),
           "p95_ratio_point": float(np.percentile(pb, 95) / np.percentile(pa, 95)), "p95_ratio_upper": float(np.percentile(boots[:, 2], 97.5)),
           "nonempty_recall": float(np.mean(rec)) if rec else None, "n_nonempty": len(rec), "exact_set": float(np.mean(same))}
    result["latency"].append(row); return row


rounds = sorted({r for r, _ in cells})
for r in rounds:
    if (r, "barrier") in cells and (r, "ready") in cells:
        paired(pair_rows((r, "barrier"), (r, "ready")), f"round {r}: barrier vs ready", "within_round")
if all((r, a) in cells for r in rounds for a in ("barrier", "ready")) and len(rounds) >= 2:
    pooled = []
    for r in rounds: pooled += pair_rows((r, "barrier"), (r, "ready"))
    paired(pooled, "pooled rounds: barrier vs ready", "pooled_position_balanced")
    paired(pair_rows((1, "barrier"), (2, "ready")), "position 1: barrier r1 vs ready r2", "same_position")
    paired(pair_rows((1, "ready"), (2, "barrier")), "position 2: ready r1 vs barrier r2", "same_position")
    paired(pair_rows((1, "barrier"), (2, "barrier")), "null: barrier r1 vs barrier r2", "null")
    paired(pair_rows((1, "ready"), (2, "ready")), "null: ready r1 vs ready r2", "null")

# ---------- program-harness node timing ----------
for (r, m), v in sorted(cells.items()):
    rows = [e for (rr, mm, q), es in workflows.items() if rr == r and mm == m and q in v and valid(v[q]) for e in es]
    nodes = [n for e in rows for n in e["nodes"] if n["name"] == "ontology_search"]
    ok = [n for n in nodes if n["status"] == "ok"]
    cell = {"workflows": len(rows), "node_status": dict(Counter(n["status"] for n in nodes)), "cancelled_workflows": sum(bool(e.get("cancelled")) for e in rows),
            "search_node_elapsed_ms": summary([e["elapsed_ms"] for e in rows]),
            "ready_wait_ms": summary([n.get("ready_wait_ms") for n in ok]), "run_ms": summary([n.get("run_ms") for n in ok]),
            "join_wait_ms": summary([n.get("join_wait_ms") for n in ok]), "overlap_before_join_ms": summary([n.get("overlap_before_join_ms") for n in ok]),
            "dependency_ready_ms": summary([n.get("dependency_ready_ms") for n in ok]), "joined_ms": summary([n.get("joined_ms") for n in ok])}
    if m == "barrier":
        # opportunity: the read could have overlapped min(run, barrier wait) if started when ready
        cell["potential_hidden_ms"] = summary([min(n["run_ms"], n["ready_wait_ms"]) for n in ok if n.get("run_ms") is not None and n.get("ready_wait_ms") is not None])
    else:
        cell["hidden_ms"] = summary([max(0.0, n["run_ms"] - (n.get("join_wait_ms") or 0.0)) for n in ok if n.get("run_ms") is not None])
    result["nodes"][f"{r}-{m}"] = cell

# ---------- hash equality between cells ----------
def wf_seq(r, m, q):
    """Workflows of one request in time order (a request may run the search node more than once)."""
    return sorted(workflows.get((r, m, q), []), key=lambda e: e["_t"])


def hash_pair(a, b, label):
    n = {"pairs": 0, "both_traced": 0, "workflow_count_equal": 0, "read_input_equal": 0, "read_output_equal_given_input": 0,
         "handoff_equal": 0, "read_trace_error": 0, "handoff_trace_error": 0, "no_workflow": 0}
    for q in cells[a]:
        if q not in cells[b] or not (valid(cells[a][q]) and valid(cells[b][q])): continue
        n["pairs"] += 1
        wa, wb = wf_seq(a[0], a[1], q), wf_seq(b[0], b[1], q)
        if not wa or not wb: n["no_workflow"] += 1; continue
        n["both_traced"] += 1
        if len(wa) != len(wb): continue
        n["workflow_count_equal"] += 1
        ra = [reads.get((a[0], a[1], e["workflow_id"])) for e in wa]; rb = [reads.get((b[0], b[1], e["workflow_id"])) for e in wb]
        ha = [handoffs.get((a[0], a[1], e["workflow_id"])) for e in wa]; hb = [handoffs.get((b[0], b[1], e["workflow_id"])) for e in wb]
        n["read_trace_error"] += sum(1 for x in ra + rb if x and x.get("status") != "ok")
        n["handoff_trace_error"] += sum(1 for x in ha + hb if x and x.get("status") != "ok")
        # reads: compare only workflows where both arms recorded a read (an early-return path records none)
        reads_ok = all((x is None) == (y is None) and (x is None or (x.get("status") == "ok" and y.get("status") == "ok")) for x, y in zip(ra, rb))
        if reads_ok and any(x is not None for x in ra):
            if all(x["input_hash"] == y["input_hash"] for x, y in zip(ra, rb) if x is not None):
                n["read_input_equal"] += 1
                n["read_output_equal_given_input"] += all(x["output_hash"] == y["output_hash"] for x, y in zip(ra, rb) if x is not None)
        if all(x and y and x.get("status") == "ok" and y.get("status") == "ok" for x, y in zip(ha, hb)):
            n["handoff_equal"] += all(x["output_hash"] == y["output_hash"] for x, y in zip(ha, hb))
    result["hashes"].append({"pair": label, **n})


for r in rounds:
    if (r, "barrier") in cells and (r, "ready") in cells: hash_pair((r, "barrier"), (r, "ready"), f"round {r}: barrier vs ready")
if len(rounds) >= 2:
    for a in ("barrier", "ready"):
        if (1, a) in cells and (2, a) in cells: hash_pair((1, a), (2, a), f"null: {a} r1 vs {a} r2")

# ---------- LLM calls per cell ----------
for (r, m), v in cells.items():
    agg = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "api_ms": 0.0}); per_request = []
    for q in v:
        calls = reqcalls.get((r, m, q), [])
        if valid(v[q]): per_request.append(len(calls))
        for c in calls:
            k = f"{c.get('model')}/{c.get('role')}"; a = agg[k]; a["calls"] += 1; a["prompt"] += c.get("prompt_tokens") or 0; a["completion"] += c.get("completion_tokens") or 0; a["api_ms"] += c.get("api_ms") or 0
    result["calls"][f"{r}-{m}"] = {"calls_per_valid_request": summary(per_request), "by_role": {k: {"calls": x["calls"], "prompt_tokens": x["prompt"], "completion_tokens": x["completion"], "mean_api_ms": x["api_ms"] / x["calls"] if x["calls"] else None} for k, x in agg.items()}}

# ---------- engine counters per cell ----------
KEYS = {"requests": "vllm:e2e_request_latency_seconds_count", "queue_s": "vllm:request_queue_time_seconds_sum", "prompt": "vllm:prompt_tokens_total", "generation": "vllm:generation_tokens_total", "preempt": "vllm:num_preemptions_total"}
for (r, m) in cells:
    p = ROOT / SUB / f"round-{r}" / f"{m}-metrics.jsonl"
    if not p.exists(): continue
    first, last = {}, {}
    for line in p.open():
        d = json.loads(line)
        if d.get("metrics"): first.setdefault(d["resource"], d); last[d["resource"]] = d
    result["engine"].append({"round": r, "arm": m, **{res: {k: last[res]["metrics"].get(v, 0) - first[res]["metrics"].get(v, 0) for k, v in KEYS.items()} for res in first}})

# ---------- per-request table (ids and timings only; no question or answer text) ----------
import csv
with OUT.with_suffix(".requests.csv").open("w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["question_id", "round", "arm", "position", "status", "valid", "duration_ms", "law_ids", "search_workflows",
                                    "search_node_elapsed_ms", "dependency_ready_ms", "ready_wait_ms", "run_ms", "join_wait_ms", "overlap_before_join_ms", "read_status"])
    for (r, m), v in sorted(cells.items()):
        pos = result["design"]["order"][str(r)].index(m) + 1
        for q, x in sorted(v.items()):
            wfs = sorted(workflows.get((r, m, q), []), key=lambda e: e["_t"]); first = wfs[0] if wfs else None
            node = next((n for n in first["nodes"] if n["name"] == "ontology_search"), None) if first else None
            f = lambda k: (None if node is None or node.get(k) is None else round(node[k], 3))
            w.writerow([q, r, m, pos, x.get("status"), int(valid(x)), x.get("duration_ms"), len(uniq(x.get("law_ids") or [])), len(wfs),
                        None if first is None else round(first["elapsed_ms"], 3), f("dependency_ready_ms"), f("ready_wait_ms"), f("run_ms"), f("join_wait_ms"), f("overlap_before_join_ms"),
                        node["status"] if node else None])
OUT.write_text(json.dumps(result, indent=1, default=float) + "\n")
for c in result["cells"]: print(f"cell r{c['round']} pos{c['position']} {c['arm']:8s} valid {c['valid']}/{c['n']} mean {c['mean_s']:.2f} p50 {c['p50_s']:.2f} p95 {c['p95_s']:.1f} wall {c['wall_s']:.0f}s {c['valid_per_s']:.3f}/s")
for k, v in result["mapping"].items(): print("mapping", k, v)
def fmt(v, spec=".3f"): return "-" if v is None else format(v, spec)
for x in result["latency"]: print(f"{x['pair']:40s} n={x['pairs']} {x['mean_a']:.2f} vs {x['mean_b']:.2f} saving {x['saving']:.3f}s ({x['pct']:.1f}%) CI [{fmt(x['ci'][0])},{fmt(x['ci'][1])}] median {x['median']:.3f} {x['pos']}/{x['neg']} p={fmt(x['sign_p'])} | p95 ratio {x['p95_ratio_point']:.3f} | recall {fmt(None if x['nonempty_recall'] is None else x['nonempty_recall']*100, '.1f')} exact {x['exact_set']*100:.1f}")
for k, v in result["nodes"].items():
    f = lambda s: f"{s['mean']:.0f}/{s['p50']:.0f}/{s['p95']:.0f}" if s.get("n") else "-"
    print(f"nodes {k}: wf {v['workflows']} status {v['node_status']} elapsed {f(v['search_node_elapsed_ms'])} ready_wait {f(v['ready_wait_ms'])} run {f(v['run_ms'])} join_wait {f(v['join_wait_ms'])} overlap {f(v['overlap_before_join_ms'])}" + (f" potential_hidden {f(v['potential_hidden_ms'])}" if "potential_hidden_ms" in v else f" hidden {f(v['hidden_ms'])}"))
for x in result["hashes"]: print("hashes", x)
for k, v in result["calls"].items(): print("calls", k, v["calls_per_valid_request"], {kk: (vv["calls"], round(vv["mean_api_ms"] or 0)) for kk, vv in v["by_role"].items()})
for e in result["engine"]: print("engine", {k: (v if not isinstance(v, dict) else {kk: round(vv) for kk, vv in v.items()}) for k, v in e.items()})

import csv
import json
import math
import re
import statistics as st
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(sys.argv[1])
OUT = Path(sys.argv[2])
SUB = sys.argv[3] if len(sys.argv) > 3 else "main"
RNG = np.random.RandomState(20260921)
B = 4000
ARMS = ["original", "direct", "reduce", "combined"]
PAIRS = [("original", "direct"), ("original", "reduce"), ("original", "combined"), ("direct", "combined"), ("reduce", "combined")]


def ts(s):
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\.(\d+)(Z|[+-]\d\d:\d\d)", s)
    return datetime.fromisoformat(m.group(1) + ("+00:00" if m.group(3) == "Z" else m.group(3))).timestamp() + float("0." + m.group(2))


def cts(s):
    return datetime.fromisoformat(s).timestamp()


def valid(r):
    resp = r.get("response")
    return r.get("status") == "ok" and isinstance(resp, dict) and isinstance(resp.get("laws"), list) and isinstance(resp.get("comment"), str)


def uniq(ids):
    out, seen = [], set()
    for v in ids:
        v = str(v or "").strip()
        if v and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def law_key(law):
    item = str(law.get("item_id") or "").strip()
    return item if item else f"{law.get('law_id')}#{law.get('paragraph')}"


def sign_p(pos, neg):
    n, k = pos + neg, min(pos, neg)
    return min(1.0, 2 * sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n) if n else None


def p95(x):
    x = sorted(x)
    return x[int(0.95 * (len(x) - 1))] if x else None


def summary(x):
    x = [v for v in x if v is not None]
    return {"n": len(x), "mean": st.mean(x), "p50": st.median(x), "p95": p95(x)} if x else {"n": 0}


def fmt(v, spec=".3f"):
    return "-" if v is None else format(v, spec)


def cluster_boot(values_by_question, stat):
    """values_by_question: {qid: [per-round values]}; returns (point, lo, hi) of stat over question-cluster resamples."""
    keys = list(values_by_question)
    if not keys:
        return None
    flat = [v for k in keys for v in values_by_question[k]]
    point = stat(flat)
    idx = RNG.randint(0, len(keys), size=(B, len(keys)))
    boots = [stat([v for k in row for v in values_by_question[keys[k]]]) for row in idx]
    return point, float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))


status = json.loads((ROOT / SUB / "status.json").read_text())
cells, order = {}, defaultdict(list)
for c in status["completed"]:
    p = ROOT / SUB / f"level-{c['level']:03d}" / f"round-{c['round']}" / "responses" / "sweep-trial" / c["method"] / "client_requests.jsonl"
    rows = [json.loads(l) for l in p.read_text(errors="replace").splitlines() if l.strip()]
    cells[c["level"], c["round"], c["method"]] = {x["question_id"]: x for x in rows}
    order[c["level"], c["round"]].append(c["method"])
levels = sorted({l for l, _, _ in cells})
print("cells:", [(k, len(v), sum(valid(x) for x in v.values())) for k, v in cells.items()])
windows = {k: (min(cts(x["started_at"]) for x in v.values()) - 2, max(cts(x["completed_at"]) for x in v.values()) + 2) for k, v in cells.items()}


def load(arm):
    ev = []
    p = ROOT / f"{arm}-application.log"
    if not p.exists():
        return ev
    for line in p.open(errors="replace"):
        i = line.find(' {"event"')
        if i < 0:
            continue
        try:
            e = json.loads(line[i + 1:])
            e["_t"] = ts(line[:40])
            ev.append(e)
        except Exception:
            pass
    return ev


arms = [a for a in ARMS if any(m == a for _, _, m in cells)]
traces = {a: load(a) for a in arms}


def cell_of(arm, t):
    for (l, r, m), (lo, hi) in windows.items():
        if m == arm and lo <= t <= hi:
            return (l, r)


def qid(case):
    return re.sub(r"-r\d+-c\d+$", "", case or "")


result = {"design": {"levels": levels, "arms": arms, "order": {f"{l}-{r}": v for (l, r), v in order.items()}},
          "cells": [], "latency": [], "interaction": [], "selection": {}, "agreement": [], "boundary": {}, "calls": {}, "engine": [], "accuracy": {}}
casemap = {}
reqcalls = defaultdict(list)
sel_in = defaultdict(list)
sel_out = defaultdict(list)
boundary = defaultdict(list)
for arm in arms:
    for e in traces[arm]:
        c = cell_of(arm, e["_t"])
        if c is None:
            continue
        if e.get("event") == "request_finished":
            q = qid(e.get("case_id"))
            if q in cells.get((c[0], c[1], arm), {}):
                for x in e.get("calls", []):
                    if x.get("application_request_id"):
                        casemap[c, arm, x["application_request_id"]] = q
                    reqcalls[c, arm, q].append(x)
    for e in traces[arm]:
        c = cell_of(arm, e["_t"])
        if c is None:
            continue
        if e.get("event") == "review_boundary" and e.get("status") == "ok":
            boundary[c, arm].append(e)
        q = casemap.get((c, arm, e.get("request_id")))
        if q is None:
            continue
        if e.get("event") == "selection_input":
            sel_in[c, arm, q].append(e)
        elif e.get("event") == "selection_result":
            sel_out[c, arm, q].append(e)

for (l, r, m), v in sorted(cells.items()):
    recs = list(v.values())
    lat = [x["duration_ms"] / 1000 for x in recs]
    ok = [x for x in recs if valid(x)]
    t0 = min(cts(x["started_at"]) for x in recs)
    t1 = max(cts(x["completed_at"]) for x in recs)
    result["cells"].append({"level": l, "round": r, "arm": m, "position": order[l, r].index(m) + 1, "n": len(recs), "valid": len(ok), "failures": len(recs) - len(ok),
                            "mean_s": st.mean(lat), "p50_s": st.median(lat), "p95_s": p95(lat), "wall_s": t1 - t0, "valid_per_s": len(ok) / (t1 - t0),
                            "law_per_s": sum(1 for x in ok if x.get("law_ids")) / (t1 - t0), "no_law": sum(1 for x in ok if not x.get("law_ids"))})
    ins = [e for (cc, mm, q), es in sel_in.items() if cc == (l, r) and mm == m and q in v for e in es]
    before = [e.get("chars_before", e.get("chars")) for e in ins]
    after = [e.get("chars_after", e.get("chars")) for e in ins]
    result["selection"][f"{l}-{r}-{m}"] = {"selection_calls": len(ins), "applied": sum(1 for e in ins if e.get("applied")), "chars_before": summary(before), "chars_after": summary(after),
                                          "reduction": 1 - (sum(after) / sum(before)) if sum(before) else None,
                                          "dropped_fields": dict(Counter(f for e in ins for f in e.get("dropped_fields", []))), "shortened_fields": sum(e.get("shortened_fields", 0) for e in ins)}
    bs = boundary.get(((l, r), m), [])
    result["boundary"][f"{l}-{r}-{m}"] = {"branches": len(bs), "effective_method": dict(Counter(e.get("effective_method") for e in bs)),
                                         "non_handler_ms": summary([e.get("non_handler_ms") for e in bs]), "handler_ms": summary([e.get("handler_ms") for e in bs]),
                                         "handler_calls": dict(Counter(str(e.get("handler_calls")) for e in bs))}


def pair_rows(a, b):
    A, Bc = cells[a], cells[b]
    return [(q, A[q], Bc[q]) for q in A if q in Bc and valid(A[q]) and valid(Bc[q])]


def paired(pairs, label, kind, level):
    if not pairs:
        return None
    qs = [q for q, _, _ in pairs]
    pa = np.array([x["duration_ms"] / 1000 for _, x, _ in pairs])
    pb = np.array([y["duration_ms"] / 1000 for _, _, y in pairs])
    d = pa - pb
    pos, neg = int((d > 0).sum()), int((d < 0).sum())
    groups = defaultdict(list)
    for i, q in enumerate(qs):
        groups[q].append(i)
    keys = list(groups)
    idx = RNG.randint(0, len(keys), size=(B, len(keys)))
    boots = []
    for row in idx:
        sel = np.concatenate([groups[keys[k]] for k in row])
        boots.append((d[sel].mean(), 1 - pb[sel].mean() / pa[sel].mean(), np.percentile(pb[sel], 95) / np.percentile(pa[sel], 95)))
    boots = np.array(boots)
    rec, same = [], []
    for q, x, y in pairs:
        g, p = set(uniq(x["law_ids"])), set(uniq(y["law_ids"]))
        if g:
            rec.append(len(g & p) / len(g))
        same.append(g == p)
    qmean = defaultdict(list)
    for i, q in enumerate(qs):
        qmean[q].append(d[i])
    qd = [st.mean(v) for v in qmean.values()]
    qpos, qneg = sum(1 for v in qd if v > 0), sum(1 for v in qd if v < 0)
    row = {"pair": label, "kind": kind, "level": level, "pairs": len(pairs), "questions": len(keys), "mean_a": float(pa.mean()), "mean_b": float(pb.mean()), "saving": float(d.mean()),
           "ci": [float(np.percentile(boots[:, 0], 2.5)), float(np.percentile(boots[:, 0], 97.5))], "pct": float(d.mean() / pa.mean() * 100), "median": float(np.median(d)),
           "pos": pos, "neg": neg, "question_pos": qpos, "question_neg": qneg, "question_sign_p": sign_p(qpos, qneg), "sd": float(d.std(ddof=1)) if len(d) > 1 else None,
           "mean_gain_lower": float(np.percentile(boots[:, 1], 2.5)), "mean_gain_upper": float(np.percentile(boots[:, 1], 97.5)),
           "p95_ratio_point": float(np.percentile(pb, 95) / np.percentile(pa, 95)), "p95_ratio_upper": float(np.percentile(boots[:, 2], 97.5)),
           "nonempty_recall": float(np.mean(rec)) if rec else None, "n_nonempty": len(rec), "exact_set": float(np.mean(same))}
    result["latency"].append(row)
    return row


def agreement(a, b, label, level):
    n = {"pairs": 0, "both_selected": 0, "selection_equal": 0, "both_has_flag": 0, "flag_equal": 0}
    for q in cells[a]:
        if q not in cells[b] or not (valid(cells[a][q]) and valid(cells[b][q])):
            continue
        n["pairs"] += 1
        sa, sb = sel_out.get((a[:2], a[2], q), []), sel_out.get((b[:2], b[2], q), [])
        if sa and sb:
            n["both_selected"] += 1
            n["selection_equal"] += sa[0].get("selected_hash") == sb[0].get("selected_hash")
            n["both_has_flag"] += 1
            n["flag_equal"] += bool(sa[0].get("has_relevant_laws")) == bool(sb[0].get("has_relevant_laws"))
    result["agreement"].append({"pair": label, "level": level, **n})


for l in levels:
    rounds = sorted({r for ll, r, _ in cells if ll == l})
    for a, b in PAIRS:
        pooled = []
        for r in rounds:
            if (l, r, a) in cells and (l, r, b) in cells:
                paired(pair_rows((l, r, a), (l, r, b)), f"C{l} round {r}: {a} vs {b}", "within_round", l)
                pooled += pair_rows((l, r, a), (l, r, b))
                agreement((l, r, a), (l, r, b), f"C{l} round {r}: {a} vs {b}", l)
        paired(pooled, f"C{l} pooled: {a} vs {b}", "pooled", l)
    null_pairs = [(1, 2), (3, 4), (1, 3), (2, 4)] if len(rounds) >= 4 else [(1, 2)]
    for a in arms:
        pooled = []
        for r1, r2 in null_pairs:
            if (l, r1, a) in cells and (l, r2, a) in cells:
                pooled += pair_rows((l, r1, a), (l, r2, a))
                agreement((l, r1, a), (l, r2, a), f"C{l} null: {a} r{r1} vs r{r2}", l)
        paired(pooled, f"C{l} null: {a} across rounds", "null", l)
    # 2x2 factorial: reduction effect on the original path vs on the direct path (question-cluster bootstrap)
    inter = defaultdict(list)
    add = defaultdict(list)
    for r in rounds:
        if all((l, r, a) in cells for a in ARMS):
            for q in cells[l, r, "original"]:
                recs = [cells[l, r, a].get(q) for a in ARMS]
                if all(x is not None and valid(x) for x in recs):
                    o, d, u, c = [x["duration_ms"] / 1000 for x in recs]
                    inter[q].append((d - c) - (o - u))
                    add[q].append((o - c) - ((o - d) + (o - u)))
    if inter:
        pt, lo, hi = cluster_boot(inter, lambda v: float(np.mean(v)))
        pa_, la_, ha_ = cluster_boot(add, lambda v: float(np.mean(v)))
        result["interaction"].append({"level": l, "pairs": sum(len(v) for v in inter.values()), "questions": len(inter),
                                      "reduction_saving_direct_minus_original": pt, "ci": [lo, hi],
                                      "combined_saving_minus_sum_of_single_savings": pa_, "ci_additive": [la_, ha_]})

for (l, r, m), v in cells.items():
    agg = defaultdict(lambda: {"calls": 0, "prompt": 0, "completion": 0, "api_ms": 0.0})
    master_prompt = []
    for q in v:
        for c in reqcalls.get(((l, r), m, q), []):
            k = f"{c.get('model')}/{c.get('role')}"
            a = agg[k]
            a["calls"] += 1
            a["prompt"] += c.get("prompt_tokens") or 0
            a["completion"] += c.get("completion_tokens") or 0
            a["api_ms"] += c.get("api_ms") or 0
            if c.get("role") == "master":
                master_prompt.append(c.get("prompt_tokens") or 0)
    result["calls"][f"{l}-{r}-{m}"] = {"master_prompt_tokens_per_request": sum(master_prompt) / max(1, len(v)), "master_prompt_p95": p95(master_prompt),
                                       "by_role": {k: {"calls": x["calls"], "prompt_tokens": x["prompt"], "completion_tokens": x["completion"], "mean_api_ms": x["api_ms"] / x["calls"] if x["calls"] else None} for k, x in agg.items()}}

KEYS = {"requests": "vllm:e2e_request_latency_seconds_count", "queue_s": "vllm:request_queue_time_seconds_sum", "prefill_s": "vllm:request_prefill_time_seconds_sum",
        "decode_s": "vllm:request_decode_time_seconds_sum", "prompt": "vllm:prompt_tokens_total", "generation": "vllm:generation_tokens_total", "preempt": "vllm:num_preemptions_total"}
for (l, r, m) in cells:
    p = ROOT / SUB / f"level-{l:03d}" / f"round-{r}" / f"{m}-metrics.jsonl"
    if not p.exists():
        continue
    first, last, run, kv = {}, {}, defaultdict(list), defaultdict(list)
    for line in p.open():
        d = json.loads(line)
        if d.get("metrics"):
            first.setdefault(d["resource"], d)
            last[d["resource"]] = d
            run[d["resource"]].append(d["metrics"].get("vllm:num_requests_running", 0))
            kv[d["resource"]].append(d["metrics"].get("vllm:kv_cache_usage_perc", d["metrics"].get("vllm:gpu_cache_usage_perc", 0)))
    row = {"level": l, "round": r, "arm": m}
    for res in first:
        dl = {k: last[res]["metrics"].get(v, 0) - first[res]["metrics"].get(v, 0) for k, v in KEYS.items()}
        n = dl["requests"] or 1
        row[res] = {**dl, "queue_per_req_s": dl["queue_s"] / n, "prefill_per_req_s": dl["prefill_s"] / n, "decode_per_req_s": dl["decode_s"] / n, "prompt_per_req": dl["prompt"] / n,
                    "running_mean": st.mean(run[res]) if run[res] else None, "kv_max": max(kv[res]) if kv[res] else None}
    result["engine"].append(row)

# accuracy against pooled relevance judgments (judge/judgments.jsonl), if present
jpath = ROOT / "judge" / "judgments.jsonl"
if jpath.exists():
    judged = defaultdict(dict)  # judge -> (qid, key) -> score (pass 1)
    for line in jpath.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("pass") == 1 and row.get("score") is not None:
            judged[row["judge"]][row["question_id"], row["law_key"]] = row["score"]
    for judge, scores in judged.items():
        relevant = defaultdict(set)
        strict = defaultdict(set)
        for (q, k), s in scores.items():
            if s >= 1:
                relevant[q].add(k)
            if s >= 2:
                strict[q].add(k)
        per_cell = {}
        per_q = defaultdict(lambda: defaultdict(list))  # (level, arm) -> qid -> metric rows

        def metrics(q, rec):
            keys = [law_key(x) for x in (rec.get("response") or {}).get("laws") or [] if isinstance(x, dict)]
            keys = list(dict.fromkeys(keys))
            judged_keys = [k for k in keys if (q, k) in scores]
            prec = np.mean([scores[q, k] >= 1 for k in judged_keys]) if judged_keys else None
            prec2 = np.mean([scores[q, k] >= 2 for k in judged_keys]) if judged_keys else None
            rel, strc = relevant.get(q, set()), strict.get(q, set())
            recall = len(rel & set(keys)) / len(rel) if rel else None
            recall2 = len(strc & set(keys)) / len(strc) if strc else None
            hit = float(bool(rel & set(keys))) if rel else None
            hit2 = float(bool(strc & set(keys))) if strc else None
            return {"returned": len(keys), "precision": prec, "precision_strict": prec2, "recall": recall, "recall_strict": recall2, "any_relevant": hit, "any_strict": hit2,
                    "empty": float(not keys)}

        for (l, r, m), v in sorted(cells.items()):
            rows = [metrics(q, x) for q, x in v.items() if valid(x)]
            for q, x in v.items():
                if valid(x):
                    per_q[l, m][q].append(metrics(q, x))
            per_cell[f"{l}-{r}-{m}"] = {k: float(np.mean([row[k] for row in rows if row[k] is not None])) if any(row[k] is not None for row in rows) else None
                                        for k in ("returned", "precision", "precision_strict", "recall", "recall_strict", "any_relevant", "any_strict", "empty")}
        comparisons = []
        for l in levels:
            for a, b in PAIRS:
                if (l, a) not in per_q or (l, b) not in per_q:
                    continue
                for metric in ("precision", "precision_strict", "recall", "recall_strict", "any_relevant", "any_strict"):
                    diffs = defaultdict(list)
                    for q in per_q[l, a]:
                        if q in per_q[l, b]:
                            ra, rb = per_q[l, a][q], per_q[l, b][q]
                            for i in range(min(len(ra), len(rb))):
                                if ra[i][metric] is not None and rb[i][metric] is not None:
                                    diffs[q].append(rb[i][metric] - ra[i][metric])
                    if diffs:
                        pt, lo, hi = cluster_boot(diffs, lambda v: float(np.mean(v)))
                        comparisons.append({"level": l, "pair": f"{a} vs {b}", "metric": metric, "pairs": sum(len(v) for v in diffs.values()), "questions": len(diffs),
                                            "difference_b_minus_a": pt, "ci": [lo, hi]})
        result["accuracy"][judge] = {"relevant_questions": len(relevant), "relevant_pairs": sum(len(v) for v in relevant.values()),
                                     "strict_pairs": sum(len(v) for v in strict.values()), "cells": per_cell, "comparisons": comparisons}
    jsum = ROOT / "judge" / "judge-summary.json"
    if jsum.exists():
        result["accuracy"]["judge_summary"] = json.loads(jsum.read_text(encoding="utf-8"))

with OUT.with_suffix(".requests.csv").open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["question_id", "level", "round", "arm", "position", "status", "valid", "duration_ms", "law_ids", "selection_chars_before", "selection_chars_after"])
    for (l, r, m), v in sorted(cells.items()):
        for q, x in sorted(v.items()):
            ins = sel_in.get(((l, r), m, q), [])
            e = ins[0] if ins else {}
            w.writerow([q, l, r, m, order[l, r].index(m) + 1, x.get("status"), int(valid(x)), x.get("duration_ms"), len(uniq(x.get("law_ids") or [])), e.get("chars_before", e.get("chars")), e.get("chars_after", e.get("chars"))])

OUT.write_text(json.dumps(result, indent=1, default=float) + "\n")
for c in result["cells"]:
    print(f"cell C{c['level']:<3d} r{c['round']} pos{c['position']} {c['arm']:8s} valid {c['valid']}/{c['n']} mean {c['mean_s']:.2f} p50 {c['p50_s']:.2f} p95 {c['p95_s']:.1f} wall {c['wall_s']:.0f}s no_law {c['no_law']}")
for x in result["latency"]:
    if x["kind"] != "within_round":
        print(f"{x['pair']:40s} n={x['pairs']} {x['mean_a']:.2f} vs {x['mean_b']:.2f} saving {x['saving']:+.2f}s ({x['pct']:+.1f}%) CI [{fmt(x['ci'][0], '.2f')},{fmt(x['ci'][1], '.2f')}] median {x['median']:+.2f} q+/- {x['question_pos']}/{x['question_neg']} gain95 [{x['mean_gain_lower']*100:.1f}%,{x['mean_gain_upper']*100:.1f}%] p95r {x['p95_ratio_point']:.3f} (up {x['p95_ratio_upper']:.3f}) recall {fmt(None if x['nonempty_recall'] is None else x['nonempty_recall']*100, '.1f')} exact {x['exact_set']*100:.1f}")
for x in result["interaction"]:
    print("interaction", x)
for x in result["agreement"]:
    print("agreement", x)
for k, v in result["accuracy"].items():
    if k == "judge_summary":
        print("judge summary", json.dumps(v, ensure_ascii=False))
        continue
    for kk, vv in v["cells"].items():
        print(f"accuracy[{k}] {kk}: " + " ".join(f"{m}={fmt(vv[m], '.3f')}" for m in ("precision", "recall", "any_relevant", "precision_strict", "recall_strict", "empty")))
    for c in v["comparisons"]:
        print(f"accuracy[{k}] C{c['level']} {c['pair']} {c['metric']}: {c['difference_b_minus_a']:+.3f} CI [{c['ci'][0]:+.3f},{c['ci'][1]:+.3f}] n={c['pairs']}")
for e in result["engine"]:
    r1 = e.get("receiver-1", {})
    print(f"engine C{e['level']:<3d} r{e['round']} {e['arm']:8s} orch req {r1.get('requests', 0):.0f} prompt/req {fmt(r1.get('prompt_per_req'), '.0f')} queue/req {fmt(r1.get('queue_per_req_s'), '.1f')}s decode/req {fmt(r1.get('decode_per_req_s'), '.1f')}s running {fmt(r1.get('running_mean'), '.1f')} kv_max {fmt(r1.get('kv_max'), '.2f')} preempt {r1.get('preempt', 0):.0f}")

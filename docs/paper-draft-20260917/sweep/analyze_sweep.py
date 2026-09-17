"""Offline summary of a concurrency sweep (stdlib only).

Discovers cells from client_requests.jsonl files. Level comes from a path component
`level-NNN` or `cNNN`, round from `round-N` (default 1), arm from the parent directory.
Per level and round: paired completed latency between the first arm (reference,
default `original`) and each other arm, throughput, timeouts, SLO attainment, source
agreement; per cell: engine counters (worker queue, waiting, KV) when metrics exist;
per arm and level: non-handler boundary time from application logs when present.
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

def sign_p(pos, neg):
    n, k = pos+neg, min(pos, neg)
    return min(1.0, 2*sum(math.comb(n, i) for i in range(k+1))/2**n) if n else None

def boot_ci(x, b=4000, seed=20260917):
    rng = random.Random(seed); n = len(x); vals = []
    for _ in range(b):
        vals.append(sum(x[rng.randrange(n)] for _ in range(n))/n)
    vals.sort(); return vals[int(0.025*b)], vals[int(0.975*b)-1]

def ts(s):
    m = re.match(r"(\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d)\.(\d+)([+-]\d\d:\d\d)", s)
    return datetime.fromisoformat(m.group(1)+m.group(3)).timestamp()+float("0."+m.group(2))

KEYS = {"requests": "vllm:e2e_request_latency_seconds_count", "queue_s": "vllm:request_queue_time_seconds_sum",
        "waiting": "vllm:num_requests_waiting", "running": "vllm:num_requests_running", "kv": "vllm:kv_cache_usage_perc",
        "preempt": "vllm:num_preemptions_total", "prompt": "vllm:prompt_tokens_total", "generation": "vllm:generation_tokens_total"}


def discover(root):
    cells = {}
    for p in root.rglob("client_requests.jsonl"):
        parts = p.parts
        level = round_ = None
        for part in parts:
            m = re.fullmatch(r"level-(\d+)|c(\d+)", part)
            if m: level = int(m.group(1) or m.group(2))
            m = re.fullmatch(r"round-(\d+)", part)
            if m: round_ = int(m.group(1))
        if level is None: continue
        arm = p.parent.name
        rows = [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
        if len(rows) < 2: continue
        cells[level, round_ or 1, arm] = {x["question_id"]: x for x in rows}
    return cells


def cell_stats(cell, slo):
    recs = list(cell.values()); ok = [r for r in recs if valid(r)]
    t0 = min(datetime.fromisoformat(r["started_at"]).timestamp() for r in recs)
    t1 = max(datetime.fromisoformat(r["completed_at"]).timestamp() for r in recs)
    lat = sorted(r["duration_ms"]/1000 for r in recs)
    return {"n": len(recs), "valid": len(ok), "failures": len(recs)-len(ok), "wall_s": t1-t0, "valid_per_s": len(ok)/(t1-t0),
            "mean_s": statistics.mean(lat), "p50_s": lat[len(lat)//2], "p95_s": lat[int(0.95*(len(lat)-1))],
            "slo_attainment": sum(l <= slo for l in lat)/len(lat)}


def main(root, logs, output, reference, slo):
    cells = discover(root)
    levels = sorted({k[0] for k in cells}); result = {"reference": reference, "slo_s": slo, "cells": [], "paired": [], "level_summary": [], "engine": [], "boundary": []}
    for (L, r, arm), c in sorted(cells.items()):
        result["cells"].append({"level": L, "round": r, "arm": arm, **cell_stats(c, slo)})
    for L in levels:
        rounds = sorted({r for (l, r, a) in cells if l == L})
        arms = sorted({a for (l, r, a) in cells if l == L} - {reference})
        for arm in arms:
            per_round = []
            for r in rounds:
                if (L, r, reference) not in cells or (L, r, arm) not in cells: continue
                A, Bc = cells[L, r, reference], cells[L, r, arm]
                qs = [q for q in A if q in Bc and valid(A[q]) and valid(Bc[q])]
                if not qs: continue
                d = [(A[q]["duration_ms"]-Bc[q]["duration_ms"])/1000 for q in qs]
                pos, neg = sum(x > 0 for x in d), sum(x < 0 for x in d)
                rec = []; same = []
                for q in qs:
                    g, p = set(uniq(A[q]["law_ids"])), set(uniq(Bc[q]["law_ids"]))
                    if g: rec.append(len(g & p)/len(g))
                    same.append(g == p)
                row = {"level": L, "round": r, "arm": arm, "pairs": len(qs), "mean_reference": statistics.mean(A[q]["duration_ms"]/1000 for q in qs),
                       "mean_arm": statistics.mean(Bc[q]["duration_ms"]/1000 for q in qs), "mean_saving": statistics.mean(d), "ci": boot_ci(d),
                       "median_saving": statistics.median(d), "pos": pos, "neg": neg, "sign_p": sign_p(pos, neg), "sd": statistics.pstdev(d),
                       "nonempty_recall": statistics.mean(rec) if rec else None, "exact_set": sum(same)/len(same)}
                result["paired"].append(row); per_round.append(row)
            if per_round:
                mr = statistics.mean(x["mean_reference"] for x in per_round); ma = statistics.mean(x["mean_arm"] for x in per_round)
                result["level_summary"].append({"level": L, "arm": arm, "rounds": len(per_round), "reference_mean": mr, "arm_mean": ma,
                                                "saving": mr-ma, "pct": (mr-ma)/mr*100, "per_round_saving": [x["mean_saving"] for x in per_round]})
    for p in sorted(root.rglob("*-metrics.jsonl")):
        parts = p.parts; L = r = None
        for part in parts:
            m = re.fullmatch(r"level-(\d+)|c(\d+)", part)
            if m: L = int(m.group(1) or m.group(2))
            m = re.fullmatch(r"round-(\d+)", part)
            if m: r = int(m.group(1))
        if L is None: continue
        arm = p.name.split("-metrics")[0]
        first, last, mx = {}, {}, defaultdict(float)
        for line in p.open():
            d = json.loads(line); res = d["resource"]; met = d.get("metrics")
            if not met: continue
            first.setdefault(res, d); last[res] = d
            for k in ("waiting", "running", "kv"): mx[res, k] = max(mx[res, k], met.get(KEYS[k], 0))
        row = {"level": L, "round": r or 1, "arm": arm}
        for res in sorted(first):
            a, b = first[res]["metrics"], last[res]["metrics"]
            row[res] = {"requests": b.get(KEYS["requests"], 0)-a.get(KEYS["requests"], 0), "queue_s": b.get(KEYS["queue_s"], 0)-a.get(KEYS["queue_s"], 0),
                        "preempt": b.get(KEYS["preempt"], 0)-a.get(KEYS["preempt"], 0), "prompt": b.get(KEYS["prompt"], 0)-a.get(KEYS["prompt"], 0),
                        "generation": b.get(KEYS["generation"], 0)-a.get(KEYS["generation"], 0), "max_waiting": mx[res, "waiting"], "max_running": mx[res, "running"], "max_kv": mx[res, "kv"]}
        result["engine"].append(row)
    # boundary timing per arm and level from application logs, assigned by batch time windows
    if logs and logs.exists():
        windows = {k: (min(datetime.fromisoformat(x["started_at"]).timestamp() for x in c.values())-2, max(datetime.fromisoformat(x["completed_at"]).timestamp() for x in c.values())+2) for k, c in cells.items()}
        for path in sorted(logs.glob("*-application.log")):
            arm = path.name.split("-application")[0]
            acc = defaultdict(list)
            for line in path.open(errors="replace"):
                i = line.find(' {"event": "review_boundary"')
                if i < 0: continue
                try: e = json.loads(line[i+1:])
                except json.JSONDecodeError: continue
                t = ts(line[:35])
                for (L, r, a), (lo, hi) in windows.items():
                    if a == arm and lo <= t <= hi:
                        acc[L, r].append(e); break
            for (L, r), evs in sorted(acc.items()):
                ok = [e for e in evs if e.get("status") == "ok"]
                result["boundary"].append({"level": L, "round": r, "arm": arm, "branches": len(evs), "non_handler_ms_mean": statistics.mean(e["non_handler_ms"] for e in ok) if ok else None,
                                           "handler_ms_mean": statistics.mean(e["handler_ms"] for e in ok) if ok else None,
                                           "non_handler_ms_p95": sorted(e["non_handler_ms"] for e in ok)[int(0.95*(len(ok)-1))] if ok else None})
    output.write_text(json.dumps(result, indent=1)+"\n")
    print(f"{'L':>4} {'r':>2} {'arm':11s} {'pairs':>5} {'ref':>8} {'arm':>8} {'saving':>7} {'%':>6} {'95% CI':>17} {'median':>7} {'pos/neg':>8} {'sign p':>7} {'recall':>7} {'set':>6}")
    for x in result["paired"]:
        print(f"{x['level']:4d} {x['round']:2d} {x['arm']:11s} {x['pairs']:5d} {x['mean_reference']:8.2f} {x['mean_arm']:8.2f} {x['mean_saving']:7.2f} {x['mean_saving']/x['mean_reference']*100:6.1f} [{x['ci'][0]:6.2f},{x['ci'][1]:6.2f}] {x['median_saving']:7.2f} {x['pos']:3d}/{x['neg']:<4d} {x['sign_p']:7.3f} {(x['nonempty_recall'] or 0)*100:7.1f} {x['exact_set']*100:6.1f}")
    for x in result["level_summary"]:
        print(f"level {x['level']:3d} {x['arm']:11s} rounds {x['rounds']} reference {x['reference_mean']:.2f} arm {x['arm_mean']:.2f} saving {x['saving']:.2f} ({x['pct']:.1f}%) per round {[round(v,2) for v in x['per_round_saving']]}")
    for x in result["cells"]:
        print(f"cell L={x['level']:3d} r={x['round']} {x['arm']:11s} valid {x['valid']}/{x['n']} wall {x['wall_s']:.0f}s {x['valid_per_s']:.3f}/s mean {x['mean_s']:.1f} p95 {x['p95_s']:.1f} SLO{slo:.0f} {x['slo_attainment']*100:.0f}%")
    for x in result["engine"]:
        w = x.get("receiver-2", {}); o = x.get("receiver-1", {})
        print(f"engine L={x['level']:3d} r={x['round']} {x['arm']:11s} worker req {w.get('requests',0):.0f} queue {w.get('queue_s',0):.0f}s maxwait {w.get('max_waiting',0):.0f} | orch req {o.get('requests',0):.0f} queue {o.get('queue_s',0):.0f}s maxwait {o.get('max_waiting',0):.0f} maxKV {o.get('max_kv',0):.2f} preempt {o.get('preempt',0):.0f}")
    for x in result["boundary"]:
        print(f"boundary L={x['level']:3d} r={x['round']} {x['arm']:11s} branches {x['branches']} non-handler mean {x['non_handler_ms_mean']} p95 {x['non_handler_ms_p95']} handler mean {x['handler_ms_mean']}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--logs", type=Path, default=None)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--reference", default="original")
    p.add_argument("--slo", type=float, default=60.0)
    a = p.parse_args(); main(a.root, a.logs, a.output, a.reference, a.slo)

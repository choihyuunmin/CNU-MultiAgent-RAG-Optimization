"""Analyze the two-arm combined-harness experiment: latency, recall, stage work.

Latency is paired at the question unit (repetitions averaged) with a clustered
bootstrap CI. Source recall is a silver known-item label, not expert correctness.
Stage work sums are observed RPC intervals, not exclusive GPU service time, and
must not be added to reconstruct end-to-end latency.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from moleg_paper_metrics import paired_cluster_ci, paired_difference_ci  # noqa: E402
from moleg_scaling_metrics import describe  # noqa: E402


def paired(rows):
    grouped = defaultdict(dict)
    for r in rows:
        grouped[(r["users"], r["arm"])].setdefault((r["case_id"], r["repeat"]), r)
    out = []
    for users in sorted({r["users"] for r in rows}):
        before, after = grouped[(users, "baseline")], grouped[(users, "combined")]
        if not before or set(before) != set(after):
            out.append({"users": users, "error": "incomplete paired set",
                        "baseline_obs": len(before), "combined_obs": len(after)})
            continue
        by_q_lat = defaultdict(list)
        exact = []
        for key in sorted(before):
            b, a = before[key], after[key]
            if b["question_sha256"] != a["question_sha256"]:
                raise ValueError("paired questions differ")
            by_q_lat[key[0]].append((b["elapsed_s"], a["elapsed_s"]))
            if b["ok"] and a["ok"]:
                exact.append(b.get("evidence_ids_sha256") == a.get("evidence_ids_sha256"))
        latency = [(statistics.fmean(x for x, _ in v), statistics.fmean(y for _, y in v))
                   for v in by_q_lat.values()]
        entry = {"users": users, "paired_questions": len(by_q_lat),
                 "latency": paired_cluster_ci(latency),
                 "evidence_list_exact_fraction": statistics.fmean(exact) if exact else None,
                 "compared_ok_pairs": len(exact)}
        for field in ("source_law_hit", "source_chunk_hit", "ok"):
            fp = []
            for key in sorted({k[0] for k in before}):
                bs, cs = [], []
                for (cid, rep) in before:
                    if cid != key:
                        continue
                    bv, av = before[(cid, rep)].get(field), after[(cid, rep)].get(field)
                    if bv is not None:
                        bs.append(float(bv))
                    if av is not None:
                        cs.append(float(av))
                if bs and cs:
                    fp.append((statistics.fmean(bs), statistics.fmean(cs)))
            entry[field + "_change"] = paired_difference_ci(fp)
        out.append(entry)
    return out


def headline(rows):
    out = []
    for users in sorted({r["users"] for r in rows}):
        row = {"users": users}
        for arm in ("baseline", "combined"):
            sub = [r for r in rows if r["users"] == users and r["arm"] == arm]
            ok = [r for r in sub if r["ok"]]
            row[arm] = {"n": len(sub), "success": len(ok),
                        "mean_s": describe([r["elapsed_s"] for r in sub])["mean"],
                        "p95_s": describe([r["elapsed_s"] for r in sub])["p95"],
                        "slo30_success_rate": sum(r["ok"] and r["elapsed_s"] <= 30 for r in sub) / len(sub) if sub else None,
                        "ttft_mean_s": describe([r.get("ttft_s") for r in sub])["mean"]}
        out.append(row)
    return out


def completion_conditioned(rows):
    out = []
    for users, arm in sorted({(r["users"], r["arm"]) for r in rows}):
        sub = [r for r in rows if r["users"] == users and r["arm"] == arm]
        out.append({"users": users, "arm": arm,
                    "successful_s": describe([r["elapsed_s"] for r in sub if r["ok"]]),
                    "failed_s": describe([r["elapsed_s"] for r in sub if not r["ok"]])})
    return out


def stage_work(workflow_path):
    """Arm-level stage/model RPC sums and worker reasoning from a workflow trace."""
    spans = []
    for line in Path(workflow_path).read_text().splitlines():
        spans.extend(json.loads(line)["spans"])
    stages = sorted({(s["stage"], s["model"]) for s in spans if s["kind"] == "model_rpc_lifetime"})
    stage_rows = []
    per_trace = defaultdict(lambda: defaultdict(float))
    # group RPC durations per (stage, model) across all requests
    by_key = defaultdict(list)
    for s in spans:
        if s["kind"] == "model_rpc_lifetime":
            by_key[(s["stage"], s["model"])].append(s["end_s"] - s["start_s"])
    for stage, model in stages:
        stage_rows.append({"stage": stage, "model": model,
                           "rpc_s": describe(by_key[(stage, model)])})
    models = {}
    for model in sorted({s["model"] for s in spans if s["kind"] == "reasoning_size"}):
        rs = [s for s in spans if s["kind"] == "reasoning_size" and s["model"] == model]
        models[model] = {
            "reasoning_characters": describe([s.get("reasoning_characters") for s in rs]),
            "visible_characters": describe([s.get("visible_characters") for s in rs]),
            "first_visible_s": describe([s.get("first_visible_s") for s in rs]),
            "reasoning_stalled": sum(s.get("reasoning_stalled") is True for s in rs),
            "stream_truncated": sum(s.get("stream_truncated") is True for s in rs),
            "unfinished_streams": sum(s.get("stream_finished") is False for s in rs)}
    return {"stage_model_rpc": stage_rows, "model_reasoning": models,
            "scope": "observed RPC intervals across the whole run; not per-load, not exclusive GPU time"}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--directory", type=Path, required=True)
    args = p.parse_args()
    root = args.directory
    rows = [json.loads(l) for l in (root / "requests.jsonl").read_text().splitlines()]
    summary = json.loads((root / "summary.json").read_text())
    result = {
        "complete": summary.get("complete", False),
        "trials": len(summary.get("trials", [])),
        "total_requests": len(rows),
        "total_success": sum(r["ok"] for r in rows),
        "latency_endpoint": "observed time to completion or 240s failure deadline; failures retained",
        "headline": headline(rows),
        "paired": paired(rows),
        "completion_conditioned_latency": completion_conditioned(rows),
        "failure_rows": [{k: r.get(k) for k in ("arm", "users", "repeat", "case_id", "kind",
                          "elapsed_s", "ttft_s", "error_type", "status", "pipeline_ok")}
                         for r in rows if not r["ok"]],
        "stage_work": {arm: stage_work(root / "apps" / f"{arm}.workflow.jsonl")
                       for arm in ("baseline", "combined")
                       if (root / "apps" / f"{arm}.workflow.jsonl").exists()},
        "per_trial_telemetry": [{"arm": t["arm"], "users": t["users"], "repeat": t["repeat"],
            "orchestrator": t.get("telemetry", {}).get("orchestrator", {}).get("histograms"),
            "worker": t.get("telemetry", {}).get("worker", {}).get("histograms"),
            "orchestrator_peak_kv": t.get("telemetry", {}).get("orchestrator", {}).get("gauges", {}).get("kv_cache_usage_perc", {}).get("max"),
            "orchestrator_preemptions": t.get("telemetry", {}).get("orchestrator", {}).get("counters", {}).get("num_preemptions_total")}
            for t in summary.get("trials", [])],
        "scope": "established regression subset; silver source labels; not expert correctness; no production change",
    }
    with (root / "combined-analysis.json").open("w") as sink:
        json.dump(result, sink, ensure_ascii=False, indent=2)
        sink.write("\n")
    for row in result["paired"]:
        if "latency" in row:
            lat = row["latency"]
            print(json.dumps({"users": row["users"], "reduction_pct": lat.get("reduction_pct"),
                              "ci": lat.get("reduction_95ci_pct"),
                              "evidence_exact": row.get("evidence_list_exact_fraction"),
                              "law_hit_change": row["source_law_hit_change"].get("delta")}))


if __name__ == "__main__":
    main()

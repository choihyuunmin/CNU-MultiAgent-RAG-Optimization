"""Aggregate scaling trials and paired question-cluster confidence intervals."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from moleg_paper_metrics import paired_cluster_ci, paired_difference_ci
from moleg_scaling_metrics import describe


def average(values):
    return statistics.fmean(values) if values else None


def key_of(row):
    return (row["users"], row["arrival_rate"], row["policy"])


def event_spans(row):
    """Observed milestone spans, not model service times or a causal trace."""
    spans = defaultdict(float)
    events = row.get("events", [])
    if not events:
        return {"before_first_event": row["elapsed_s"]}
    spans["before_first_event"] = events[0]["s"]
    for i, event in enumerate(events):
        end = events[i + 1]["s"] if i + 1 < len(events) else row["elapsed_s"]
        spans[event["stage"]] += max(0, end - event["s"])
    return dict(spans)


def aggregate(directory):
    state = json.loads((directory / "summary.json").read_text())
    rows = [json.loads(l) for l in (directory / "requests.jsonl").open()]
    completed_trials = {t["trial"] for t in state["trials"]}
    partial_count = sum(r["trial"] not in completed_trials for r in rows)
    rows = [r for r in rows if r["trial"] in completed_trials]
    records, trials = defaultdict(list), defaultdict(list)
    for row in rows:
        records[key_of(row)].append(row)
    for trial in state["trials"]:
        trials[key_of(trial)].append(trial)
    groups = []
    for key, ts in trials.items():
        rs = records[key]
        wall = sum(t["wall_s"] for t in ts)
        metrics = {k: describe([r.get(k) for r in rs]) for k in
                   ("elapsed_s", "ttft_s", "first_event_s", "admission_wait_s", "wire_s")}
        metrics["after_first_event_s"] = describe([r["elapsed_s"] - r["first_event_s"]
                                                  for r in rs if r["first_event_s"] is not None])
        metrics["wire_before_first_event_s"] = describe([
            r["first_event_s"] - r["admission_wait_s"] - r["scheduling_lag_s"]
            for r in rs if r["first_event_s"] is not None])
        servers = {}
        for server in ts[0]["telemetry"]:
            valid = [t["telemetry"][server] for t in ts if t["telemetry"][server].get("available")]
            histograms = {}
            for metric in ("request_queue_time_seconds", "request_prefill_time_seconds",
                           "request_decode_time_seconds", "request_prompt_tokens",
                           "request_generation_tokens", "request_prefill_kv_computed_tokens"):
                valid_hist = [v["histograms"][metric] for v in valid
                              if v["histograms"][metric]["count"] is not None
                              and v["histograms"][metric]["sum"] is not None]
                count = sum(v["count"] for v in valid_hist)
                histograms[metric] = {"count": count if valid_hist else None,
                    "mean": sum(v["sum"] for v in valid_hist) / count if count else None}
            def peak(name):
                xs = [v["gauges"][name]["max"] for v in valid if v["gauges"][name]["max"] is not None]
                return max(xs) if xs else None
            preempts = [v["counters"]["num_preemptions_total"] for v in valid]
            servers[server] = {"histograms": histograms, "peak_waiting": peak("num_requests_waiting"),
                "peak_running": peak("num_requests_running"), "peak_kv_usage": peak("kv_cache_usage_perc"),
                "preemptions": sum(preempts) if preempts and all(v is not None for v in preempts) else None,
                "scope": "whole shared instance; not request-attributed"}
        group = {"users": key[0], "arrival_rate": key[1], "policy": key[2], "n": len(rs),
            "unique_questions": len({r["case_id"] for r in rs}), "repeats": len(ts),
            "success": sum(r["ok"] for r in rs), "wall_s": wall, "metrics": metrics,
            "throughput_rps": sum(r["ok"] for r in rs) / wall,
            "slo_goodput_rps": sum(t["slo_goodput_rps"] * t["wall_s"] for t in ts) / wall,
            "slo_attainment": sum(t["slo_attainment"] * t["n"] for t in ts) / sum(t["n"] for t in ts),
            "mean_outstanding": sum(t["mean_outstanding"] * t["wall_s"] for t in ts) / wall,
            "source_law_hit": average([float(r["source_law_hit"]) for r in rs if r["source_law_hit"] is not None]),
            "source_chunk_hit": average([float(r["source_chunk_hit"]) for r in rs if r["source_chunk_hit"] is not None]),
            "telemetry": servers}
        spans = [event_spans(r) for r in rs]
        stage_names = sorted({name for span in spans for name in span})
        group["event_spans_per_api_s"] = {name: {
            "requests_with_stage": sum(name in span for span in spans),
            "timing_all_requests": describe([span.get(name, 0.) for span in spans])}
            for name in stage_names}
        group["by_kind"] = {}
        for kind in sorted({r["kind"] for r in rs}):
            kr = [r for r in rs if r["kind"] == kind]
            group["by_kind"][kind] = {"n": len(kr), "success": sum(r["ok"] for r in kr),
                "latency_s": describe([r["elapsed_s"] for r in kr]),
                "ttft_s": describe([r["ttft_s"] for r in kr]),
                "after_first_token_s": describe([r["elapsed_s"] - r["ttft_s"] for r in kr if r["ttft_s"] is not None]),
                "answer_chars": describe([r["answer_chars"] for r in kr])}
        groups.append(group)
    comparisons = []
    for key, rs in records.items():
        reference_key = (key[0], key[1], "baseline") if key[2] != "baseline" else (1, None, "baseline")
        if reference_key not in records or reference_key == key:
            continue
        before, after = defaultdict(list), defaultdict(list)
        for r in records[reference_key]:
            before[r["case_id"]].append(r)
        for r in rs:
            after[r["case_id"]].append(r)
        pairs = [(average([r["elapsed_s"] for r in before[q]]), average([r["elapsed_s"] for r in after[q]]))
                 for q in sorted(before.keys() & after.keys())]
        src_pairs = [(average([float(r["source_law_hit"]) for r in before[q] if r["source_law_hit"] is not None]),
                      average([float(r["source_law_hit"]) for r in after[q] if r["source_law_hit"] is not None]))
                     for q in sorted(before.keys() & after.keys())]
        src_pairs = [(a, b) for a, b in src_pairs if a is not None and b is not None]
        evidence = []
        for q in sorted(before.keys() & after.keys()):
            for b in before[q]:
                a = next((r for r in after[q] if r["repeat"] == b["repeat"]), None)
                if a and b["ok"] and a["ok"]:
                    evidence.append(b["evidence_ids_sha256"] == a["evidence_ids_sha256"])
        comparisons.append({"reference": list(reference_key), "candidate": list(key),
            "latency": paired_cluster_ci(pairs), "source_law_change": paired_difference_ci(src_pairs),
            "returned_id_list_exact": average(evidence), "paired_successful_responses": len(evidence)})
    return {"complete": state["complete"], "partial_trial_rows_not_aggregated": partial_count,
            "groups": groups, "comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    args = parser.parse_args()
    output = aggregate(args.directory)
    (args.directory / "aggregate.json").write_text(json.dumps(output, indent=2) + "\n")
    print("users rate policy n mean_s p95_s rps goodput mean_outstanding")
    for g in output["groups"]:
        print(g["users"], g["arrival_rate"], g["policy"], g["n"],
              *(round(v, 3) for v in [g["metrics"]["elapsed_s"]["mean"], g["metrics"]["elapsed_s"]["p95"],
                                     g["throughput_rps"], g["slo_goodput_rps"], g["mean_outstanding"]]))


if __name__ == "__main__":
    main()

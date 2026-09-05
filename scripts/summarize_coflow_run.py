#!/usr/bin/env python3
"""Summarize measured request/trace artifacts, including block bootstrap CIs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import statistics
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def percentile(values, percent):
    if not values:
        return None
    ordered = sorted(values)
    position = (len(ordered) - 1) * percent / 100
    low = int(position)
    high = min(low + 1, len(ordered) - 1)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def check_quality_gate(quality, protocol_gate):
    if not quality:
        return "not_evaluated"
    names = {"success_rate": "success_rate", "document_recall": "law_recall_mean",
             "ndcg10": "law_ndcg10_mean", "top1": "top1_agreement_mean",
             "answer_similarity": "comment_similarity_mean", "fidelity": "baseline_fidelity_mean"}
    for name, field in names.items():
        limit = float(protocol_gate[name])
        try:
            measured = float(quality[field])
        except (KeyError, TypeError, ValueError):
            return "fail"
        if not math.isfinite(measured) or measured < limit:
            return "fail"
    return "pass"


def paired_bootstrap(control, candidate, *, seed=20260905, iterations=2000):
    if set(control) != set(candidate):
        raise ValueError("control/candidate question IDs do not align")
    blocks = defaultdict(list)
    for key, row in control.items():
        if row["status"] == "ok" and candidate[key]["status"] == "ok":
            if row["block_index"] != candidate[key]["block_index"]:
                raise ValueError("mismatched question block")
            blocks[row["block_index"]].append((row["duration_ms"], candidate[key]["duration_ms"]))
    rng = random.Random(seed)
    pairs = [pair for block in blocks.values() for pair in block]
    if not pairs:
        return {}
    samples = []
    names = list(blocks)
    for _ in range(iterations):
        resampled = [pair for block in rng.choices(names, k=len(names)) for pair in blocks[block]]
        base = sum(x for x, _ in resampled)
        changed = sum(y for _, y in resampled)
        samples.append(100 * (1 - changed / base))
    return {"paired_questions": len(pairs), "blocks": len(blocks),
            "mean_reduction_percent": 100 * (1 - sum(y for _, y in pairs) / sum(x for x, _ in pairs)),
            "block_bootstrap_ci95": [percentile(samples, 2.5), percentile(samples, 97.5)],
            "faster_fraction": sum(y < x for x, y in pairs) / len(pairs)}


def quality_difference(reference, candidate, question_blocks, *, seed=20260905, iterations=2000):
    """Compare two runs' agreement scores against the SAME original control.

    This estimates excess disagreement beyond a repeated unchanged control.
    It is not a direct answer-to-answer comparison or a correctness estimate.
    All scored rows are included, including failed rows scored by the evaluator.
    """
    if not reference or set(reference) != set(candidate) or set(reference) != set(question_blocks):
        raise ValueError("quality question IDs do not align")
    fields = ("law_recall", "law_ndcg10", "top1_agreement", "comment_similarity", "baseline_fidelity")
    grouped = defaultdict(list)
    for key in reference:
        grouped[question_blocks[key]].append(key)
    result = {}
    names = list(grouped)
    for field in fields:
        differences = {key: float(candidate[key][field]) - float(reference[key][field]) for key in reference}
        if any(not -1 <= value <= 1 for value in differences.values()):
            raise ValueError("invalid quality difference")
        rng = random.Random(seed)
        samples = []
        for _ in range(iterations):
            values = [differences[key] for block in rng.choices(names, k=len(names)) for key in grouped[block]]
            samples.append(100 * statistics.mean(values))
        result[field] = {"mean_difference_percentage_points": 100 * statistics.mean(differences.values()),
                         "block_bootstrap_ci95": [percentile(samples, 2.5), percentile(samples, 97.5)]}
    return {"paired_questions": len(reference), "blocks": len(grouped), "metrics": result}


def timeline_partition(trace):
    """Disjoint wall-time regions; never subtract overlapping call-time sums.

    LLM API time includes transport, upstream queuing, and computation. Other
    work can overlap it; these regions describe observation, not causation.
    """
    total = float(trace["duration_ms"])
    events = defaultdict(lambda: [0, 0])
    events[0.0]
    events[total]
    call_intervals = list(trace.get("llm_calls", []))
    # The existing non-streaming client can miss a completion trace when its
    # coroutine is cancelled. The admission wrapper records its service span
    # in finally, on every arm, so retain those observed busy intervals too.
    for event in trace.get("events", []):
        attrs = event.get("attributes", {})
        if event.get("name") == "coflow_admission" and "service_ms" in attrs and "offset_ms" in event:
            call_intervals.append({"start_offset_ms": event["offset_ms"] - attrs["service_ms"],
                                   "end_offset_ms": event["offset_ms"]})
    for kind, intervals in enumerate((call_intervals, [
        span for span in trace.get("spans", []) if span.get("name") == "coflow_admission"
    ])):
        for interval in intervals:
            if "start_offset_ms" not in interval or "end_offset_ms" not in interval:
                continue
            start = max(0.0, min(total, float(interval["start_offset_ms"])))
            end = max(start, min(total, float(interval["end_offset_ms"])))
            events[start][kind] += 1
            events[end][kind] -= 1
    regions = dict.fromkeys(("llm_api_only_ms", "local_admission_only_ms", "llm_api_and_admission_ms", "neither_observed_ms"), 0.0)
    active = [0, 0]
    previous = 0.0
    for at in sorted(events):
        name = ("llm_api_and_admission_ms" if all(active) else "llm_api_only_ms" if active[0]
                else "local_admission_only_ms" if active[1] else "neither_observed_ms")
        regions[name] += at - previous
        active = [active[index] + events[at][index] for index in (0, 1)]
        previous = at
    return regions


def trace_stats(path, run_id, active_windows):
    waits, services, durations, token_counts = [], [], [], []
    resources, windows = Counter(), Counter()
    receiver_waiting = defaultdict(list)
    counter_deltas = defaultdict(float)
    last_samples = {}
    covered_seconds = defaultdict(float)
    errors = 0
    failed_span_requests = 0
    failed_spans = Counter()
    timelines = []
    llm_statuses = Counter()
    request_ids, recorded_call_ids = set(), set()
    sent_calls = {}
    timeout_request_ids = set()
    interrupted_admissions = 0
    interrupted_requests = set()
    post_generate = []
    for raw in path.open(encoding="utf-8", errors="replace"):
        if "RAG_TRACE_V1 " in raw:
            row, _ = json.JSONDecoder().raw_decode(raw.split("RAG_TRACE_V1 ", 1)[1])
            if row.get("run_id") != run_id:
                continue
            request_ids.add(row["request_id"])
            timelines.append(timeline_partition(row))
            generate = next((span for span in row.get("spans", []) if span.get("name") == "generate"), None)
            if generate is not None and "end_offset_ms" in generate:
                post_generate.append(max(0.0, row["duration_ms"] - generate["end_offset_ms"]))
            bad_spans = [span for span in row.get("spans", []) if span.get("status") == "error"]
            failed_span_requests += bool(bad_spans)
            failed_spans.update(span.get("name", "unknown") for span in bad_spans)
            for call in row.get("llm_calls", []):
                if call.get("call_id"):
                    recorded_call_ids.add(call["call_id"])
                llm_statuses[call.get("status", "unknown")] += 1
                if isinstance(call.get("duration_ms"), (int, float)):
                    durations.append(call["duration_ms"])
                if isinstance(call.get("completion_tokens"), (int, float)):
                    token_counts.append(call["completion_tokens"])
            for event in row.get("events", []):
                if event.get("name") == "coflow_admission":
                    attrs = event["attributes"]
                    waits.append(attrs["wait_ms"])
                    if "service_ms" in attrs:
                        services.append(attrs["service_ms"])
                    if attrs.get("resource"):
                        resources[attrs["resource"]] += 1
                    if "window" in attrs:
                        windows[attrs["window"]] += 1
                    if attrs.get("success") is False:
                        interrupted_admissions += 1
                        interrupted_requests.add(row["request_id"])
        elif raw.startswith("CNU_RECEIVER_V1 "):
            row, _ = json.JSONDecoder().raw_decode(raw.split(" ", 1)[1])
            window = next((index for index, (start, end) in enumerate(active_windows)
                           if start <= row["time"] <= end), None)
            if window is None:
                continue
            receiver_waiting[row["resource"]].append(row["waiting"])
            previous = last_samples.get(row["resource"])
            if previous is not None and previous[0] == window:
                old = previous[1]
                gap = row["time"] - old["time"]
                if 0 < gap <= 8:
                    covered_seconds[row["resource"]] += gap
                    for name, value in row.get("counters", {}).items():
                        prior = old.get("counters", {}).get(name)
                        if prior is not None and value >= prior:
                            counter_deltas[row["resource"] + "|" + name] += value - prior
            last_samples[row["resource"]] = (window, row)
        elif raw.startswith("CNU_RECEIVER_ERROR "):
            errors += 1
        else:
            sent = re.search(r"\[([^\]]+)\]\s+LLM sent\s+\|\s+call=(\w+)", raw)
            if sent:
                sent_calls[sent.group(2)] = sent.group(1)
            timed_out = re.search(r"\[([^\]]+)\]\s+tool_loop LLM timeout after", raw)
            if timed_out:
                timeout_request_ids.add(timed_out.group(1))
    unrecorded_calls = {call for call, request in sent_calls.items()
                        if request in request_ids and call not in recorded_call_ids}
    return {"llm_calls": len(durations), "completion_tokens": sum(token_counts),
            "mean_llm_call_service_ms": statistics.mean(durations) if durations else None,
            "admission_calls": len(waits), "calls_waiting_over_1ms": sum(x > 1 for x in waits),
            "mean_admission_wait_ms": statistics.mean(waits) if waits else None,
            "p95_admission_wait_ms": percentile(waits, 95), "max_admission_wait_ms": max(waits, default=0),
            "calls_by_receiver": dict(resources), "windows_at_dispatch": dict(windows),
            "receiver_queue_p95": {key: percentile(value, 95) for key, value in receiver_waiting.items()},
            "telemetry_errors": errors,
            "requests_with_failed_spans": failed_span_requests,
            "failed_spans": dict(failed_spans),
            "llm_call_statuses": dict(llm_statuses),
            "sent_calls_missing_trace": len(unrecorded_calls),
            "observed_llm_attempts_lower_bound": len(recorded_call_ids | unrecorded_calls),
            "requests_with_tool_loop_timeout": len(timeout_request_ids & request_ids),
            "interrupted_admissions": interrupted_admissions,
            "requests_with_interrupted_admissions": len(interrupted_requests),
            "traced_requests": len(timelines),
            "mean_timeline_regions_ms": {key: statistics.mean(row[key] for row in timelines)
                                         for key in timelines[0]} if timelines else {},
            "mean_after_generate_span_ms": statistics.mean(post_generate) if post_generate else None,
            "telemetry_covered_seconds": dict(covered_seconds),
            "shared_receiver_counter_deltas": dict(counter_deltas)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_root", type=Path)
    args = parser.parse_args()
    root = args.run_root.resolve()
    protocol = json.loads((root / "protocol.json").read_text())
    quality = {}
    quality_path = root / "quality" / "variant_summary.csv"
    if quality_path.exists():
        for row in csv.DictReader(quality_path.open(encoding="utf-8-sig")):
            if row["complexity"] == "ALL":
                quality[row["variant"]] = row
    results = {}
    aligned = {}
    for arm in protocol["arms"]:
        path = root / arm / "client_requests.jsonl"
        rows = [json.loads(raw) for raw in path.read_text().splitlines() if raw.strip()]
        by_id = {row["question_id"]: row for row in rows}
        if len(by_id) != len(rows):
            raise ValueError(f"duplicate question IDs in {arm}")
        aligned[arm] = by_id
        ok = [row for row in rows if row["status"] == "ok"]
        values = [row["duration_ms"] / 1000 for row in ok]
        walls = []
        by_block = defaultdict(list)
        for row in rows:
            by_block[row["block_index"]].append(row)
        active_windows = [(
            min(datetime.fromisoformat(row["started_at"]).timestamp() for row in block),
            max(datetime.fromisoformat(row["completed_at"]).timestamp() for row in block),
        ) for block in by_block.values()]
        for path in sorted((root / "cells").glob(f"*/{protocol['run_id']}/{arm}/summary.json")):
            walls.extend(cell["wall_seconds"] for cell in json.loads(path.read_text())["cells"])
        fidelity = quality.get(arm, {})
        gate_status = check_quality_gate(fidelity, protocol["quality_gate"])
        results[arm] = {"requests": len(rows), "successes": len(ok), "expected": protocol["questions_per_arm"],
                        "empty_retrieval_results": sum(not row.get("law_ids") for row in rows),
                        "mean_seconds": statistics.mean(values) if values else None,
                        "p50_seconds": percentile(values, 50), "p95_seconds": percentile(values, 95),
                        "p99_seconds": percentile(values, 99), "throughput_rps": len(ok) / sum(walls) if walls else None,
                        "quality": fidelity, "quality_passed": gate_status == "pass", "quality_gate_status": gate_status,
                        "complete": len(ok) == len(rows) == protocol["questions_per_arm"],
                        "trace": trace_stats(root / arm / "server.log", protocol["run_id"], active_windows)}
    for arm in protocol["arms"][1:]:
        results[arm]["paired_latency"] = paired_bootstrap(aligned["control"], aligned[arm])
    if "control_repeat" in aligned:
        scored = defaultdict(dict)
        scored_path = root / "quality" / "question_fidelity.csv"
        if scored_path.exists():
            for row in csv.DictReader(scored_path.open(encoding="utf-8-sig")):
                if row["question_id"] in scored[row["variant"]]:
                    raise ValueError("duplicate quality question ID")
                scored[row["variant"]][row["question_id"]] = row
        for arm in protocol["arms"]:
            if arm in {"control", "control_repeat"}:
                continue
            results[arm]["paired_latency_vs_control_repeat"] = paired_bootstrap(aligned["control_repeat"], aligned[arm])
            if scored:
                results[arm]["quality_difference_vs_control_repeat"] = quality_difference(
                    scored["control_repeat"], scored[arm],
                    {key: row["block_index"] for key, row in aligned["control"].items()})
    (root / "measured_summary.json").write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    lines = ["# Measured coflow experiment", "", f"Run: {protocol['run_id']}", "",
             "| Method | Success | Mean (s) | p95 (s) | Recorded LLM calls | Fidelity | Gate |",
             "|---|---:|---:|---:|---:|---:|---|"]
    for arm, row in results.items():
        fidelity = row["quality"].get("baseline_fidelity_mean")
        fidelity_text = f"{float(fidelity):.4f}" if fidelity is not None else ("Reference" if arm == "control" else "Pending")
        gate = "Reference" if arm == "control" else row["quality_gate_status"]
        lines.append(f"| {arm} | {row['successes']}/{row['expected']} | {row['mean_seconds']:.3f} | {row['p95_seconds']:.3f} | {row['trace']['llm_calls']} | {fidelity_text} | {gate} |")
    lines.extend(["", "Fidelity measures agreement with the current control, not expert correctness.",
                  "Inference counters are shared-server observations and include unrelated traffic.",
                  "Recorded LLM calls can omit interrupted non-streaming calls; missing sent-call traces and timeouts are reported separately.",
                  "Latency confidence intervals resample complete question blocks; see measured_summary.json."])
    (root / "measured_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()

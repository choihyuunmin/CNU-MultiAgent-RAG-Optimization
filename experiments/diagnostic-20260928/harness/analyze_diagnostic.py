"""Analyze the content-free diagnostic records, preserving incomplete cells."""
import argparse
import json
import math
import random
import statistics as st
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


def percentile(values, p):
    ordered = sorted(values)
    if not ordered:
        return None
    k = (len(ordered) - 1) * p / 100
    lo = math.floor(k)
    hi = math.ceil(k)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def describe(values):
    return {"n": len(values), "mean": st.mean(values), "median": st.median(values),
            "p95": percentile(values, 95)} if values else {"n": 0}


def time_s(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def bootstrap_question(pairs, draws=4000, seed=20260928):
    grouped = defaultdict(list)
    for q, delta in pairs:
        grouped[q].append(delta)
    keys = sorted(grouped)
    if not keys:
        return None
    rng = random.Random(seed)
    estimates = []
    for _ in range(draws):
        values = [v for _ in keys for v in grouped[rng.choice(keys)]]
        estimates.append(st.mean(values))
    return [percentile(estimates, 2.5), percentile(estimates, 97.5)]


def metrics(path):
    rows = [json.loads(x) for x in path.open() if x.strip()]
    output = {}
    for resource in ("receiver-1", "receiver-2"):
        series = [r["metrics"] for r in rows if r.get("resource") == resource and r.get("metrics")]
        if not series:
            continue
        counter_names = {
            "completed_requests": "vllm:e2e_request_latency_seconds_count",
            "queue_count": "vllm:request_queue_time_seconds_count",
            "queue_sum_s": "vllm:request_queue_time_seconds_sum",
            "prefill_count": "vllm:request_prefill_time_seconds_count",
            "prefill_sum_s": "vllm:request_prefill_time_seconds_sum",
            "decode_count": "vllm:request_decode_time_seconds_count",
            "decode_sum_s": "vllm:request_decode_time_seconds_sum",
            "prompt_tokens": "vllm:prompt_tokens_total",
            "generation_tokens": "vllm:generation_tokens_total",
            "preemptions": "vllm:num_preemptions_total",
            "prefix_queries": "vllm:prefix_cache_queries_total",
            "prefix_hits": "vllm:prefix_cache_hits_total",
        }
        delta = {out: series[-1].get(name, 0) - series[0].get(name, 0)
                 for out, name in counter_names.items()}
        resets = {out: sum(series[i].get(name, 0) < series[i - 1].get(name, 0)
                           for i in range(1, len(series)))
                  for out, name in counter_names.items()}
        n = delta["queue_count"]
        output[resource] = {"samples": len(series), "deltas": delta,
                            "counter_resets": {k: v for k, v in resets.items() if v},
                            "queue_mean_s": delta["queue_sum_s"] / n if n > 0 else None,
                            "waiting": describe([s.get("vllm:num_requests_waiting", 0) for s in series]),
                            "running": describe([s.get("vllm:num_requests_running", 0) for s in series]),
                            "kv_fraction": describe([s.get("vllm:kv_cache_usage_perc", 0) for s in series]),
                            "kv_fraction_max": max(s.get("vllm:kv_cache_usage_perc", 0) for s in series)}
    output["collector_errors"] = sum(bool(r.get("error_type")) for r in rows)
    return output


def classify(row, events):
    stages = [e for e in events if e.get("event") in ("stage_start", "stage_end")]
    branches = [e for e in events if e.get("event") == "branch_end"]
    selection = [e for e in events if e.get("event") == "selection_call_end"]
    filters = [e for e in events if e.get("event") == "candidate_filter"]
    search_done = [e for e in stages if e.get("event") == "stage_end" and e.get("stage") == "execute_search"]
    selected = [e for e in selection if e.get("status") == "ok"
                and (e.get("state") or {}).get("has_relevant_laws")
                and (e.get("state") or {}).get("selected_count", 0) > 0]
    candidate_max = max([0] + [(e.get("state") or {}).get("candidates_count", 0)
                               for e in search_done] +
                        [(e.get("candidate") or {}).get("candidates_count", 0)
                         for e in events if e.get("event") == "selection_call_start"])
    filtered_max = max([0] + [e.get("after_count", 0) for e in filters])
    final_max = max([0] + [(e.get("state") or {}).get("final_count", 0)
                           for e in stages if e.get("event") == "stage_end"])
    errors = [e for e in events if (e.get("event") in
              ("stage_end", "branch_end", "selection_call_end", "model_finish")
              and (e.get("status") == "error" or e.get("error_type")))]
    branch_success = sum(e.get("status") == "ok" for e in branches)
    branch_error = sum(e.get("status") == "error" for e in branches)
    branch_timeout = sum(e.get("error_type") == "TimeoutError" for e in branches)
    model_events = [e for e in events if e.get("event") == "model_finish"]
    search_started = any(e.get("event") == "stage_start" and e.get("stage") == "execute_search"
                         for e in events)
    searched = bool(search_done or search_started)
    fallback = any(e.get("event") == "stage_start" and e.get("stage") == "fallback_chain"
                   for e in events)
    request_finished = any(e.get("event") == "request_finished" for e in events)
    no_law = not row.get("law_ids")
    cause = None
    if no_law:
        if selected and final_max > 0 and not errors:
            cause = "selected_id_final_delivery_loss"
        elif selected and errors:
            cause = "internal_error_or_timeout_fallback"
        elif errors and branch_success == 0:
            cause = "internal_error_or_timeout_fallback"
        elif not searched and request_finished:
            cause = "search_not_executed"
        elif searched and candidate_max == 0 and search_done:
            cause = "search_no_results"
        elif candidate_max > 0 and selection and not selected:
            cause = "candidates_no_selection"
        elif selected and not errors:
            cause = "selected_id_final_delivery_loss"
        else:
            cause = "insufficient_record"
    return {"cause": cause, "search_executed": searched, "candidate_max": candidate_max,
            "filtered_candidate_max": filtered_max, "selection_calls": len(selection),
            "selected_relevant_calls": len(selected), "final_stage_max": final_max,
            "branch_success": branch_success, "branch_error": branch_error,
            "branch_timeout": branch_timeout,
            "model_cancelled": sum(e.get("error_type") == "CancelledError" for e in model_events),
            "model_bad_request": sum(e.get("error_type") == "BadRequestError" for e in model_events),
            "partial_branch_error": bool(branch_error and branch_success),
            "internal_error_events": len(errors), "fallback_chain_executed": fallback,
            "request_finished_event": request_finished,
            # A valid early HITL/assistant response can finish before _run_single
            # creates any law-search branch.
            "internal_completed": bool(request_finished and not errors),
            "normal_no_relevant_judgment": bool(selection and not selected and not errors),
            "hitl": bool(row.get("hitl"))}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, required=True)
    args = p.parse_args()
    root = args.dir
    requests = [json.loads(x) for x in (root / "raw" / "requests.jsonl").open() if x.strip()]
    events = [json.loads(x) for x in (root / "raw" / "events.jsonl").open() if x.strip()]
    by_case = defaultdict(list)
    for e in events:
        if e.get("case_id"):
            by_case[(e["arm"], e["case_id"])].append(e)
    details = []
    for row in requests:
        # The frozen load generator reuses case_id in round 2. The attempt_id
        # is unique but is not sent to the app, so join by case AND cell time.
        start, end = time_s(row["started_at"]), time_s(row["completed_at"])
        # Client timestamps are rounded to milliseconds while CRI log stamps
        # have nanoseconds; the response-final event can land just after end.
        e = [event for event in by_case.get((row["arm"], row["case_id"]), [])
             if start - 2 <= time_s(event["timestamp"]) <= end + 2]
        details.append({"experiment_id": row["experiment_id"], "arm": row["arm"],
                        "round": row["round"], "question_id": row["question_id"],
                        "case_id": row["case_id"], "duration_ms": row["duration_ms"],
                        "client_status": row["status"], "format_valid": row["format_valid"],
                        "law_ids": row["law_ids"], **classify(row, e)})
    with (root / "classified-requests.jsonl").open("w") as dest:
        for item in details:
            dest.write(json.dumps(item, ensure_ascii=False) + "\n")
    result = {"experiment_id": "diagnostic-20260928", "total_requests": len(requests),
              "conditions": {}, "rounds": {}, "paired": {}, "stages": {}, "engines": {},
              "engine_by_condition": {},
              "selection": {}, "search_boundary": {}, "model_finish": {},
              "limitations": ["The stage durations are per invocation; parallel stages are not additive.",
                              "Engine metrics are shared-service aggregates, not request KV footprints."]}
    for arm in ("original", "combined"):
        subset = [r for r in requests if r["arm"] == arm]
        det = [d for d in details if d["arm"] == arm]
        if not subset:
            continue
        # The two arm runs are separated by another condition. Throughput must
        # use only this arm's active cell windows, not the span across rounds.
        cell_walls = {}
        for round_id in (1, 2):
            cell_rows = [r for r in subset if r["round"] == round_id]
            if cell_rows:
                cell_walls[round_id] = (max(time_s(r["completed_at"]) for r in cell_rows)
                                        - min(time_s(r["started_at"]) for r in cell_rows))
        wall = sum(cell_walls.values())
        with_law = [r for r in subset if r["law_ids"]]
        no_law = [r for r in subset if not r["law_ids"]]
        result["conditions"][arm] = {
            "requests": len(subset), "format_valid": sum(r["format_valid"] for r in subset),
            "client_errors": sum(r["status"] != "ok" for r in subset),
            "internal_completed": sum(d["internal_completed"] for d in det),
            "internal_error_requests": sum(d["internal_error_events"] > 0 for d in det),
            "branch_timeout_requests": sum(d["branch_timeout"] > 0 for d in det),
            "branch_timeout_invocations": sum(d["branch_timeout"] for d in det),
            "model_cancelled_requests": sum(d["model_cancelled"] > 0 for d in det),
            "model_bad_request_requests": sum(d["model_bad_request"] > 0 for d in det),
            "partial_branch_error_requests": sum(d["partial_branch_error"] for d in det),
            "fallback_chain_requests": sum(d["fallback_chain_executed"] for d in det),
            "with_law": len(with_law), "no_law": len(no_law),
            "no_law_causes": dict(Counter(d["cause"] for d in det if not d["law_ids"])),
            "duration_s": describe([r["duration_ms"] / 1000 for r in subset]),
            "with_law_duration_s": describe([r["duration_ms"] / 1000 for r in with_law]),
            "no_law_duration_s": describe([r["duration_ms"] / 1000 for r in no_law]),
            "wall_s": wall, "format_per_s": sum(r["format_valid"] for r in subset) / wall,
            "law_per_s": len(with_law) / wall,
            "trace_request_finished_coverage": sum(d["request_finished_event"] for d in det),
            "multi_branch_requests": sum(d["branch_success"] + d["branch_error"] > 1 for d in det),
            "candidate_max_per_request": describe([d["candidate_max"] for d in det]),
            "filtered_candidate_max_per_request": describe([d["filtered_candidate_max"] for d in det]),
            "selected_relevant_calls_per_request": describe([d["selected_relevant_calls"] for d in det]),
            "final_law_ids_per_response": describe([len(r["law_ids"]) for r in subset]),
        }
        for round_id in (1, 2):
            cell = [r for r in subset if r["round"] == round_id]
            if not cell:
                continue
            span = cell_walls[round_id]
            result["rounds"][f"r{round_id}-{arm}"] = {
                "requests": len(cell), "with_law": sum(bool(r["law_ids"]) for r in cell),
                "wall_s": span, "duration_s": describe([r["duration_ms"] / 1000 for r in cell])}
            path = root / "raw" / "metrics" / f"r{round_id}-{arm}.jsonl"
            if path.exists():
                result["engines"][f"r{round_id}-{arm}"] = metrics(path)
        stage_groups = defaultdict(list)
        for e in events:
            if e["arm"] == arm and e.get("event") == "stage_end" and e.get("duration_ms") is not None:
                stage_groups[e["stage"]].append(e)
        result["stages"][arm] = {name: {"duration_ms": describe([e["duration_ms"] for e in es]),
                                         "errors": sum(e.get("status") == "error" for e in es)}
                                 for name, es in stage_groups.items()}
        branch_ends = [e for e in events if e["arm"] == arm and e.get("event") == "branch_end"]
        result["stages"][arm]["full_branch_span"] = {
            "duration_ms": describe([e["duration_ms"] for e in branch_ends if e.get("duration_ms") is not None]),
            "errors": sum(e.get("status") == "error" for e in branch_ends)}
        selections = [e for e in events if e["arm"] == arm and e.get("event") == "selection_input"]
        parses = [e for e in events if e["arm"] == arm and e.get("event") == "selection_parse"]
        filters = [e for e in events if e["arm"] == arm and e.get("event") == "candidate_filter"]
        result["selection"][arm] = {
            "input_events": len(selections),
            "chars_before": describe([e.get("chars_before", e.get("chars")) for e in selections
                                      if e.get("chars_before", e.get("chars")) is not None]),
            "chars_after": describe([e.get("chars_after", e.get("chars")) for e in selections
                                     if e.get("chars_after", e.get("chars")) is not None]),
            "documents_before": describe([e["documents_before"] for e in selections if "documents_before" in e]),
            "documents_after": describe([e["documents_after"] for e in selections if "documents_after" in e]),
            "model_outputs_observed": len(parses),
            "parsed_json_objects": sum(bool(e.get("json_object")) for e in parses),
            "selected_ids_field_present": sum(bool(e.get("selected_ids_field")) for e in parses),
            "filter_calls": len(filters),
            "filter_before": describe([e["before_count"] for e in filters]),
            "filter_after": describe([e["after_count"] for e in filters]),
            "filter_after_zero": sum(e["after_count"] == 0 for e in filters),
        }
        boundary = [e for e in events if e["arm"] == arm and e.get("event") == "review_boundary"]
        handlers = [e for e in events if e["arm"] == arm and e.get("event") == "review_handler"]
        result["search_boundary"][arm] = {
            "branches": len(boundary), "handler_events": len(handlers),
            "handler_calls": sum(e.get("handler_calls", 0) for e in boundary),
            "prepared_arguments_equal": sum(e.get("arguments_equal") is True for e in handlers),
            "boundary_total_ms": describe([e["total_ms"] for e in boundary if e.get("total_ms") is not None]),
            "handler_ms": describe([e["handler_ms"] for e in boundary if e.get("handler_ms") is not None]),
            "non_handler_ms": describe([e["non_handler_ms"] for e in boundary if e.get("non_handler_ms") is not None]),
            "errors": sum(e.get("status") == "error" for e in boundary),
            "effective_method": dict(Counter(e.get("effective_method") for e in boundary)),
        }
        finishes = [e for e in events if e["arm"] == arm and e.get("event") == "model_finish"]
        result["model_finish"][arm] = {
            "events": len(finishes), "by_reason": dict(Counter(e.get("finish_reason") or "unknown" for e in finishes)),
            "error_types": dict(Counter(e["error_type"] for e in finishes if e.get("error_type"))),
        }
        call_groups = defaultdict(list)
        for e in events:
            if e["arm"] == arm and e.get("event") == "request_finished":
                for call in e.get("calls", []):
                    call_groups[call.get("role") or "unknown"].append(call)
        result.setdefault("model_calls", {})[arm] = {
            role: {"calls": len(calls),
                   "prompt_tokens": sum(c.get("prompt_tokens") or 0 for c in calls),
                   "completion_tokens": sum(c.get("completion_tokens") or 0 for c in calls),
                   "api_ms": describe([c["api_ms"] for c in calls if c.get("api_ms") is not None])}
            for role, calls in call_groups.items()}
        for resource, label in (("receiver-1", "orchestration"), ("receiver-2", "worker")):
            records = [result["engines"].get(f"r{round_id}-{arm}", {}).get(resource)
                       for round_id in (1, 2)]
            records = [record for record in records if record]
            if not records:
                continue
            names = records[0]["deltas"]
            delta = {name: sum(record["deltas"][name] for record in records) for name in names}
            completed = delta["completed_requests"]
            queue_count = delta["queue_count"]
            result["engine_by_condition"][f"{arm}_{label}"] = {
                "deltas": delta,
                "queue_mean_s": delta["queue_sum_s"] / queue_count if queue_count else None,
                "prompt_tokens_per_completed": delta["prompt_tokens"] / completed if completed else None,
                "generation_tokens_per_completed": delta["generation_tokens"] / completed if completed else None,
                "waiting_sample_mean": (sum(r["waiting"]["mean"] * r["samples"] for r in records)
                                        / sum(r["samples"] for r in records)),
                "running_sample_mean": (sum(r["running"]["mean"] * r["samples"] for r in records)
                                        / sum(r["samples"] for r in records)),
                "kv_fraction_max": max(r["kv_fraction_max"] for r in records),
                "counter_resets": sum(bool(r["counter_resets"]) for r in records),
                "collector_errors": sum(result["engines"][f"r{round_id}-{arm}"]["collector_errors"]
                                        for round_id in (1, 2) if f"r{round_id}-{arm}" in result["engines"]),
            }
    lookup = {(r["arm"], r["round"], r["question_id"]): r for r in requests}
    all_pairs, both = [], []
    law_binary = []
    for (arm, round_id, q), original in lookup.items():
        if arm != "original":
            continue
        combined = lookup.get(("combined", round_id, q))
        if combined is None:
            continue
        all_pairs.append((q, (original["duration_ms"] - combined["duration_ms"]) / 1000))
        law_binary.append((q, int(bool(combined["law_ids"])) - int(bool(original["law_ids"]))))
        if original["law_ids"] and combined["law_ids"]:
            both.append((q, (original["duration_ms"] - combined["duration_ms"]) / 1000))
    for name, pairs in (("all_duration_s", all_pairs), ("both_law_duration_s", both),
                        ("law_inclusion_difference", law_binary)):
        if pairs:
            result["paired"][name] = {"pairs": len(pairs), "questions": len({q for q, _ in pairs}),
                                      "mean": st.mean(v for _, v in pairs),
                                      "question_cluster_bootstrap_95ci": bootstrap_question(pairs)}
    (root / "analysis.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"total_requests": len(requests), "conditions": result["conditions"],
                      "paired": result["paired"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

"""Analyze program-harness timing, quality and native timeout evidence."""
import argparse
import importlib.util
import json
import re
import statistics
import tarfile
from datetime import datetime
from pathlib import Path

from analyze_capability_scale import read_rows, summarize
from analyze_scaling_experiment import valid_response


def quality_detail(original, candidate, law_metrics):
    gold = {row["question_id"]: row for row in original}
    pred = {row["question_id"]: row for row in candidate}
    if len(gold) != 200 or len(pred) != 200 or gold.keys() != pred.keys():
        raise ValueError("expected 200 unique paired questions")
    nonempty = [key for key, row in gold.items() if valid_response(row) and row.get("law_ids")]
    scores = [law_metrics(gold[key]["law_ids"], pred[key].get("law_ids", []))["law_recall"]
              if valid_response(pred[key]) else 0 for key in nonempty]
    return {
        "original_with_laws": sum(bool(row.get("law_ids")) and valid_response(row) for row in original),
        "candidate_with_laws": sum(bool(row.get("law_ids")) and valid_response(row) for row in candidate),
        "nonempty_reference_denominator": len(nonempty),
        "nonempty_reference_recall": statistics.mean(scores) if scores else None,
        "empty_reference_new_laws": sum(
            not row.get("law_ids") and bool(pred[key].get("law_ids")) and valid_response(pred[key])
            for key, row in gold.items()),
        "exact_law_sets": sum(
            valid_response(pred[key]) and valid_response(row)
            and set(row.get("law_ids", [])) == set(pred[key].get("law_ids", []))
            for key, row in gold.items()),
    }


def log_detail(text, records):
    """Link native timeout logs to this batch without storing prompt contents."""
    decoder = json.JSONDecoder()
    cases = {row["case_id"]: row for row in records}
    identifiers, timeouts, llm_timeouts = {}, [], []
    log_times = []
    begin = min(datetime.fromisoformat(row["started_at"]).timestamp() for row in records)
    end = max(datetime.fromisoformat(row["completed_at"]).timestamp() for row in records)
    for line in text.splitlines():
        try:
            log_times.append(datetime.fromisoformat(line.split(" ", 1)[0]).timestamp())
        except ValueError:
            pass
        marker = "CNU_APP_ADAPTER_V1 "
        if marker in line:
            try:
                row, _ = decoder.raw_decode(line.split(marker, 1)[1])
            except ValueError:
                row = {}
            request_id = row.get("application_request_id")
            if row.get("case_id") in cases and request_id not in (None, "", "-"):
                identifiers[request_id] = row["case_id"]
        received = re.search(
            r"\[([^\]]+)\] REQ  recv .*?session=exp-capability-trial-(.+)-[a-f0-9]{8} ", line)
        if received and received[2] in cases:
            identifiers[received[1]] = received[2]
        timeout = re.search(r"\[([^\]]+)\] sub-task \d+/\d+ timeout:", line)
        if timeout:
            timeouts.append(timeout[1])
        timeout = re.search(r"\[([^\]]+)\] LLM call timeout", line)
        if timeout:
            llm_timeouts.append(timeout[1])
    affected = {identifiers[item] for item in timeouts if item in identifiers}
    return {
        "mapped_questions": len(set(identifiers.values())),
        "log_start_covers_batch": bool(log_times) and min(log_times) <= begin,
        "log_end_covers_batch": bool(log_times) and max(log_times) >= end,
        "linked_subtask_timeout_events": sum(item in identifiers for item in timeouts),
        "linked_subtask_timeout_questions": len(affected),
        "timeout_questions_with_empty_laws": sum(not cases[key].get("law_ids") for key in affected),
        "linked_llm_timeout_events": sum(item in identifiers for item in llm_timeouts),
        "caveat": "Timeout counts are linked log observations, not proof that every empty result has one cause.",
    }


def archive_text(path, mode):
    with tarfile.open(path) as archive:
        return "\n".join(
            archive.extractfile(member).read().decode(errors="replace")
            for member in archive.getmembers() if member.isfile() and f"-{mode}_" in member.name)


def harness_detail(text, records):
    begin = min(datetime.fromisoformat(row["started_at"]).timestamp() for row in records)
    end = max(datetime.fromisoformat(row["completed_at"]).timestamp() for row in records)
    decoder = json.JSONDecoder()
    workflows = []
    for line in text.splitlines():
        marker = "CNU_PROGRAM_HARNESS_V1 "
        if marker not in line:
            continue
        try:
            stamp = datetime.fromisoformat(line.split(" ", 1)[0]).timestamp()
            row, _ = decoder.raw_decode(line.split(marker, 1)[1])
        except (ValueError, IndexError):
            continue
        if begin <= stamp <= end and row.get("event") == "workflow_closed":
            workflows.append(row)
    nodes = [node for workflow in workflows for node in workflow.get("nodes", [])]
    ontology = [node for node in nodes if node.get("name") == "ontology_search"]
    numeric = lambda field: [node[field] for node in ontology if node.get(field) is not None]
    return {
        "workflows": len(workflows),
        "nodes": len(nodes),
        "cancelled_nodes": sum(workflow.get("cancelled", 0) for workflow in workflows),
        "ontology_nodes": len(ontology),
        "ontology_success": sum(node.get("status") == "ok" for node in ontology),
        "ontology_run_mean_ms": statistics.mean(numeric("run_ms")) if numeric("run_ms") else None,
        "ontology_join_wait_mean_ms": statistics.mean(numeric("join_wait_ms")) if numeric("join_wait_ms") else None,
        "ontology_overlap_mean_ms": statistics.mean(numeric("overlap_before_join_ms")) if numeric("overlap_before_join_ms") else None,
        "ontology_ready_before_join": sum((node.get("join_wait_ms") or 0) < 1 for node in ontology),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--logs", type=Path, required=True)
    parser.add_argument("--evaluator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeat-baseline", type=Path,
                        help="Optional prior unchanged-baseline client_requests.jsonl")
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("original_quality", args.evaluator)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    report = summarize(args.results, evaluator.law_metrics)
    for cell in report["cells"]:
        if cell["state"] != "complete":
            continue
        root = args.results / f'c{cell["concurrency"]:03d}' / "responses/capability-trial"
        groups = {mode: read_rows(root / mode / "client_requests.jsonl") for mode in ("original", "capability")}
        cell["quality_detail"] = quality_detail(groups["original"], groups["capability"], evaluator.law_metrics)
        cell["log_detail"] = {mode: log_detail(archive_text(args.logs, mode), rows)
                              for mode, rows in groups.items()}
        cell["harness_detail"] = harness_detail(archive_text(args.logs, "capability"), groups["capability"])
        if args.repeat_baseline:
            repeated = read_rows(args.repeat_baseline)
            cell["baseline_repeat_detail"] = quality_detail(groups["original"], repeated, evaluator.law_metrics)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

"""Verify completed trial coverage, no duplicate cases, and timing arithmetic."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import math
from pathlib import Path


def audit(root):
    manifest = json.loads((root / "manifest.json").read_text())
    summary = json.loads((root / "summary.json").read_text())
    rows = [json.loads(line) for line in (root / "requests.jsonl").open()]
    expected = {c["case_id"]: c for c in manifest["selected_cases"]}
    errors = []
    completion_issues = []
    if not summary["complete"]:
        completion_issues.append("campaign is incomplete")
    expected_trials = len(manifest["rates"] or manifest["users"]) * len(manifest["policies"]) * manifest["repeats"]
    if len(summary["trials"]) != expected_trials:
        completion_issues.append("trial count does not match manifest")
    expected_conditions = {(None if manifest["rates"] else load,
                            load if manifest["rates"] else None, policy, repeat)
                           for load in (manifest["rates"] or manifest["users"])
                           for policy in manifest["policies"] for repeat in range(manifest["repeats"])}
    grouped = defaultdict(list)
    for row in rows:
        grouped[row["trial"]].append(row)
        if row["case_id"] not in expected or row["question_sha256"] != expected[row["case_id"]]["question_sha256"]:
            errors.append("case id/hash differs from manifest")
        if row["elapsed_s"] < 0 or not math.isfinite(row["elapsed_s"]):
            errors.append("invalid user latency")
        if row["wire_s"] is not None:
            reconstructed = row["scheduling_lag_s"] + row["admission_wait_s"] + row["wire_s"]
            if abs(reconstructed - row["elapsed_s"]) > 1e-5:
                errors.append("user latency excludes some scheduling/admission time")
        for key in ("ttft_s", "first_event_s"):
            if row[key] is not None and not 0 <= row[key] <= row["elapsed_s"]:
                errors.append("first event/token outside request interval")
        if any(key in row for key in ("question", "prompt", "result", "answer", "api_key", "endpoint")):
            errors.append("raw content field in public request row")
    seen_trials = set()
    seen_conditions = set()
    coverage = []
    for trial in summary["trials"]:
        tid = trial["trial"]
        if tid in seen_trials:
            errors.append("duplicate trial id")
        seen_trials.add(tid)
        condition = tuple(trial[k] for k in ("users", "arrival_rate", "policy", "repeat"))
        if condition not in expected_conditions or condition in seen_conditions:
            errors.append("unexpected or duplicate experimental condition")
        seen_conditions.add(condition)
        rs = grouped[tid]
        if any(any(r[k] != trial[k] for k in ("users", "arrival_rate", "policy", "repeat")) for r in rs):
            errors.append("row condition differs from trial condition")
        ids = Counter(r["case_id"] for r in rs)
        if set(ids) != set(expected) or any(n != 1 for n in ids.values()):
            errors.append("missing or duplicate question in trial")
        if len(rs) != trial["n"] or sum(r["ok"] for r in rs) != trial["success"]:
            errors.append("summary request/success count mismatch")
        if rs:
            mean = sum(r["elapsed_s"] for r in rs) / len(rs)
            if abs(mean - trial["metrics"]["elapsed_s"]["mean"]) > 1e-8:
                errors.append("summary latency mismatch")
        coverage.append({"trial": tid, "users": trial["users"], "arrival_rate": trial["arrival_rate"],
                         "policy": trial["policy"], "repeat": trial["repeat"], "rows": len(rs)})
    if set(grouped) != seen_trials:
        errors.append("raw rows include an unfinished/unrecognized trial")
    if 'planned_order' in manifest:
        actual_order = [{k: t[k] for k in ['repeat', 'users', 'policy']} for t in summary['trials']]
        if actual_order != manifest['planned_order'][:len(actual_order)]:
            errors.append('trial execution order differs from declared order')
    return {"passed": not errors and not completion_issues, "integrity_passed": not errors,
            "campaign_complete": summary["complete"], "completion_issues": completion_issues,
            "planned_trials": expected_trials, "completed_trials": len(summary["trials"]),
            "errors": sorted(set(errors)), "rows": len(rows),
            "unique_questions": len(expected), "trials": coverage,
            "sha256": {name: hashlib.sha256((root / name).read_bytes()).hexdigest()
                       for name in ("manifest.json", "requests.jsonl", "telemetry.jsonl", "summary.json")},
            "scope": "artifact coverage and timing consistency, not answer correctness or causal proof"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--allow-incomplete", action="store_true",
                        help="succeed on valid stopped artifacts without calling the campaign complete")
    args = parser.parse_args()
    result = audit(args.directory)
    (args.directory / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("passed", "integrity_passed", "campaign_complete", "errors", "completion_issues", "rows", "unique_questions")}))
    if not result["integrity_passed"] or (not result["passed"] and not args.allow_incomplete):
        raise SystemExit(1)


if __name__ == "__main__":
    main()

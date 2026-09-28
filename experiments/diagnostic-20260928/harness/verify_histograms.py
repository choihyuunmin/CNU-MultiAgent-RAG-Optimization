"""Audit queue-histogram denominators and counter continuity in archived cells."""
import argparse
import json
from collections import defaultdict
from pathlib import Path

QUEUE = "vllm:request_queue_time_seconds_count"
E2E = "vllm:e2e_request_latency_seconds_count"
SUM = "vllm:request_queue_time_seconds_sum"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("output", type=Path)
    args = p.parse_args()
    checks = []
    for file in sorted(args.root.glob("level-*/round-*/*-metrics.jsonl")):
        by_resource = defaultdict(list)
        sample_errors = 0
        for line in file.open():
            row = json.loads(line)
            sample_errors += bool(row.get("error_type"))
            if row.get("metrics"):
                by_resource[row["resource"]].append(row["metrics"])
        for resource, samples in by_resource.items():
            if QUEUE not in samples[0] or E2E not in samples[0]:
                checks.append({"cell": str(file.relative_to(args.root)), "resource": resource,
                               "error": "histogram count missing"})
                continue
            queue_delta = samples[-1][QUEUE] - samples[0][QUEUE]
            e2e_delta = samples[-1][E2E] - samples[0][E2E]
            queue_sum_delta = samples[-1].get(SUM, 0) - samples[0].get(SUM, 0)
            resets = sum(samples[i].get(QUEUE, 0) < samples[i - 1].get(QUEUE, 0)
                         or samples[i].get(E2E, 0) < samples[i - 1].get(E2E, 0)
                         for i in range(1, len(samples)))
            checks.append({"cell": str(file.relative_to(args.root)), "resource": resource,
                           "samples": len(samples), "sample_errors": sample_errors,
                           "queue_count_delta": queue_delta, "e2e_count_delta": e2e_delta,
                           "queue_sum_delta_s": queue_sum_delta, "counter_resets": resets,
                           "histogram_count_equal": queue_delta == e2e_delta,
                           "queue_mean_s": queue_sum_delta / queue_delta if queue_delta > 0 else None})
    result = {"cells": len({r["cell"] for r in checks}), "series": len(checks),
              "count_mismatches": sum(r.get("histogram_count_equal") is False for r in checks),
              "missing_counts": sum("error" in r for r in checks),
              "counter_resets": sum(r.get("counter_resets", 0) for r in checks),
              "sample_errors": sum(r.get("sample_errors", 0) for r in checks),
              "checks": checks}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print({k: v for k, v in result.items() if k != "checks"})


if __name__ == "__main__":
    main()

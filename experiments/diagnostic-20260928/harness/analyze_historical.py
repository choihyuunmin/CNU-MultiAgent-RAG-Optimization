"""Recheck the archived C100 comparison without inferring missing branch events."""
import argparse
import csv
import json
import random
import statistics as st
from collections import defaultdict
from pathlib import Path


def pct(values, p):
    values = sorted(values)
    if not values:
        return None
    i = (len(values) - 1) * p / 100
    lo = int(i)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (i - lo)


def describe(values):
    return {"n": len(values), "mean_s": st.mean(values),
            "median_s": st.median(values), "p95_s": pct(values, 95)} if values else {"n": 0}


def ci_by_question(pairs, seed=20260928, draws=4000):
    groups = defaultdict(list)
    for question, delta in pairs:
        groups[question].append(delta)
    questions = sorted(groups)
    rng = random.Random(seed)
    boot = []
    for _ in range(draws):
        sample = [value for _ in questions for value in groups[rng.choice(questions)]]
        boot.append(st.mean(sample))
    return [pct(boot, 2.5), pct(boot, 97.5)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--old", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    old = args.old
    rows = list(csv.DictReader((old / "analysis.requests.csv").open(newline="")))
    archive = json.loads((old / "analysis.json").read_text())
    rows = [r for r in rows if r["level"] == "100" and r["arm"] in ("original", "combined")]
    for r in rows:
        r["duration_s"] = float(r["duration_ms"]) / 1000
        r["has_law"] = int(r["law_ids"]) > 0
        r["selection_chars"] = float(r["selection_chars_after"] or 0)
    result = {"source": {"requests_csv": str(old / "analysis.requests.csv"),
                         "analysis_json": str(old / "analysis.json"),
                         "c100_raw_application_events_available": False,
                         "selection_chars_semantics": "first selection_input event only"},
              "conditions": {}, "pairs": {}, "engines": {}}
    cells = {(r["arm"], r["round"], r["question_id"]): r for r in rows}
    for arm in ("original", "combined"):
        group = [r for r in rows if r["arm"] == arm]
        no = [r for r in group if not r["has_law"]]
        cell_summaries = [c for c in archive["cells"] if c["level"] == 100 and c["arm"] == arm]
        wall = sum(c["wall_s"] for c in cell_summaries)
        result["conditions"][arm] = {
            "requests": len(group), "format_valid": sum(r["valid"] == "1" for r in group),
            "client_error": sum(r["status"] != "ok" for r in group),
            "has_law": len(group) - len(no), "no_law": len(no),
            "no_law_selection_chars_positive": sum(r["selection_chars"] > 0 for r in no),
            "no_law_selection_chars_unrecorded": sum(not r["selection_chars_after"] for r in no),
            "duration": describe([r["duration_s"] for r in group]),
            "with_law_duration": describe([r["duration_s"] for r in group if r["has_law"]]),
            "no_law_duration": describe([r["duration_s"] for r in no]),
            "wall_s": wall,
            "format_per_s": sum(r["valid"] == "1" for r in group) / wall,
            "law_per_s": (len(group) - len(no)) / wall,
        }
        engine_rows = [e for e in archive["engine"] if e["level"] == 100 and e["arm"] == arm]
        for receiver, name in (("receiver-1", "orchestration"), ("receiver-2", "worker")):
            records = [e[receiver] for e in engine_rows if receiver in e]
            req = sum(x["requests"] for x in records)
            result["engines"][f"{arm}_{name}"] = {
                "completed_requests": req,
                "queue_s_per_completed_request": sum(x["queue_s"] for x in records) / req,
                "prefill_s_per_completed_request": sum(x["prefill_s"] for x in records) / req,
                "decode_s_per_completed_request": sum(x["decode_s"] for x in records) / req,
                "prompt_tokens_per_completed_request": sum(x["prompt"] for x in records) / req,
                "generation_tokens_per_completed_request": sum(x["generation"] for x in records) / req,
                "preemptions": sum(x["preempt"] for x in records),
                "running_mean_of_round_means": st.mean(x["running_mean"] for x in records),
                "kv_max": max(x["kv_max"] for x in records),
            }
    matched = []
    both_law = []
    for (arm, round_id, q), original in cells.items():
        if arm != "original":
            continue
        combined = cells.get(("combined", round_id, q))
        if not combined:
            continue
        pair = (q, original["duration_s"] - combined["duration_s"])
        matched.append(pair)
        if original["has_law"] and combined["has_law"]:
            both_law.append(pair)
    for name, pairs in (("all", matched), ("both_returned_law", both_law)):
        result["pairs"][name] = {"pairs": len(pairs), "questions": len({q for q, _ in pairs}),
                                 "mean_original_minus_combined_s": st.mean(d for _, d in pairs),
                                 "median_original_minus_combined_s": st.median(d for _, d in pairs),
                                 "question_cluster_bootstrap_95ci_s": ci_by_question(pairs)}
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired end-to-end /api/generate check for baseline and candidate servers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import statistics
import sys
import time
from typing import Any
import urllib.request
import uuid

from evaluate_moleg_search_dispatch import _ids, _percentile


def _post(base_url: str, question: str, timeout: float) -> tuple[float, dict[str, Any]]:
    body = json.dumps(
        {"prompt": question, "session_id": uuid.uuid4().hex}, ensure_ascii=False
    ).encode("utf-8")
    request = urllib.request.Request(
        base_url.rstrip("/") + "/api/generate",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        data = json.loads(response.read().decode("utf-8"))
    return time.perf_counter() - start, data


def _payload(data: dict[str, Any]) -> dict[str, Any]:
    early = data.get("early_exit_response")
    return early if isinstance(early, dict) else data


def _laws(data: dict[str, Any]) -> list[dict[str, Any]]:
    value = _payload(data).get("laws")
    return [row for row in value if isinstance(row, dict)] if isinstance(value, list) else []


def _comment(data: dict[str, Any]) -> str:
    return str(_payload(data).get("comment") or "").strip()


def _char_bigrams(text: str) -> set[str]:
    normalized = "".join(text.lower().split())
    if len(normalized) < 2:
        return {normalized} if normalized else set()
    return {normalized[index : index + 2] for index in range(len(normalized) - 1)}


def _jaccard(left: str, right: str) -> float:
    a = _char_bigrams(left)
    b = _char_bigrams(right)
    return len(a & b) / len(a | b) if a or b else 1.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-root", required=True)
    parser.add_argument("--baseline-url", default="http://127.0.0.1:28010")
    parser.add_argument("--candidate-url", default="http://127.0.0.1:28011")
    parser.add_argument("--per-kind", type=int, default=5)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    root = Path(args.rag_root).resolve()
    sys.path.insert(0, str(root))
    from scripts.load_test_questions import build_test_queries

    all_questions = build_test_queries()
    questions = all_questions[: args.per_kind] + all_questions[25 : 25 + args.per_kind]
    # Excluded warm-up for model/client/application initialization.
    _post(args.baseline_url, questions[0], args.timeout)
    _post(args.candidate_url, questions[0], args.timeout)

    rows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    for index, question in enumerate(questions):
        order = ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        outputs: dict[str, dict[str, Any]] = {}
        elapsed: dict[str, float] = {}
        for variant in order:
            url = args.baseline_url if variant == "baseline" else args.candidate_url
            elapsed[variant], outputs[variant] = _post(url, question, args.timeout)
        baseline_laws = _laws(outputs["baseline"])
        candidate_laws = _laws(outputs["candidate"])
        baseline_ids = _ids(baseline_laws)
        candidate_ids = _ids(candidate_laws)
        baseline_set = set(baseline_ids)
        rows.append(
            {
                "case_index": index,
                "kind": "action" if index < args.per_kind else "named_law",
                "baseline_s": elapsed["baseline"],
                "candidate_s": elapsed["candidate"],
                "baseline_result_count": len(baseline_ids),
                "candidate_result_count": len(candidate_ids),
                "exact_ranked_ids": baseline_ids == candidate_ids,
                "recall_vs_baseline": (
                    len(baseline_set.intersection(candidate_ids)) / len(baseline_set)
                    if baseline_set
                    else 1.0
                ),
                "top1_agreement": baseline_ids[:1] == candidate_ids[:1],
                "comment_bigram_jaccard": _jaccard(
                    _comment(outputs["baseline"]), _comment(outputs["candidate"])
                ),
            }
        )
        print(
            f"[{index + 1}/{len(questions)}] baseline={elapsed['baseline']:.3f}s "
            f"candidate={elapsed['candidate']:.3f}s exact={baseline_ids == candidate_ids}",
            file=sys.stderr,
            flush=True,
        )

    baseline_times = [row["baseline_s"] for row in rows]
    candidate_times = [row["candidate_s"] for row in rows]
    result = {
        "protocol": {
            "question_count": len(questions),
            "action_questions": args.per_kind,
            "named_law_questions": args.per_kind,
            "paired_order": "alternating",
            "excluded_warmups_per_server": 1,
            "baseline_url": args.baseline_url,
            "candidate_url": args.candidate_url,
        },
        "baseline": {
            "mean_s": statistics.fmean(baseline_times),
            "median_s": statistics.median(baseline_times),
            "p95_s": _percentile(baseline_times, 95),
        },
        "candidate": {
            "mean_s": statistics.fmean(candidate_times),
            "median_s": statistics.median(candidate_times),
            "p95_s": _percentile(candidate_times, 95),
        },
        "comparison": {
            "mean_latency_reduction_pct": (
                (statistics.fmean(baseline_times) - statistics.fmean(candidate_times))
                / statistics.fmean(baseline_times)
                * 100.0
            ),
            "exact_ranked_id_rate": statistics.fmean(row["exact_ranked_ids"] for row in rows),
            "recall_vs_baseline": statistics.fmean(row["recall_vs_baseline"] for row in rows),
            "top1_agreement": statistics.fmean(row["top1_agreement"] for row in rows),
            "mean_comment_bigram_jaccard": statistics.fmean(
                row["comment_bigram_jaccard"] for row in rows
            ),
        },
        "wall_s": time.perf_counter() - wall_start,
        "rows": rows,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

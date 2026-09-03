#!/usr/bin/env python3
"""Measure the deployed reranker authentication gap as a quality ablation."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
from pathlib import Path
import statistics
import sys
import time
from typing import Any

from evaluate_moleg_search_dispatch import (
    _ids,
    _named_law_rank,
    _percentile,
    _structured_case,
)


def _metrics(rows: list[dict[str, Any]], variant: str) -> dict[str, Any]:
    selected = [row for row in rows if row["variant"] == variant]
    times = [row["elapsed_s"] for row in selected]
    return {
        "mean_s": statistics.fmean(times),
        "median_s": statistics.median(times),
        "p95_s": _percentile(times, 95),
        "named_law_hit_at_20": statistics.fmean(row["named_law_rank"] is not None for row in selected),
        "named_law_mrr_at_20": statistics.fmean(
            1.0 / row["named_law_rank"] if row["named_law_rank"] else 0.0 for row in selected
        ),
        "recall_vs_current": statistics.fmean(row["recall_vs_current"] for row in selected),
        "top1_agreement": statistics.fmean(row["top1_agreement"] for row in selected),
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    rag_root = Path(args.rag_root).resolve()
    sys.path.insert(0, str(rag_root))
    sys.path.insert(0, str(rag_root / "src"))

    from scripts.load_test_questions import build_test_queries
    import tools.search_tool.engine as engine_module

    key = os.environ.get("RERANK_API_KEY", "").strip()
    if not key:
        raise RuntimeError("RERANK_API_KEY is required; it is never written to results")

    logging.disable(logging.CRITICAL)
    search_engine = engine_module.search_engine
    original_post = engine_module.requests.post
    questions = build_test_queries()
    cases = [_structured_case(question) for question in questions if "의 " in question]
    cases = cases[: args.limit]

    def authenticated_post(url: str, *post_args: Any, **post_kwargs: Any):
        headers = dict(post_kwargs.pop("headers", {}) or {})
        headers["Authorization"] = f"Bearer {key}"
        return original_post(url, *post_args, headers=headers, **post_kwargs)

    async def search(case: dict[str, Any], authenticated: bool) -> dict[str, Any]:
        engine_module.requests.post = authenticated_post if authenticated else original_post
        try:
            return await asyncio.to_thread(
                search_engine.search_laws,
                case["country"],
                case["keywords"],
                case["search_terms"],
            )
        finally:
            engine_module.requests.post = original_post

    # Initialize clients/connections; warm-up is excluded.
    await search(cases[0], authenticated=False)
    await search(cases[0], authenticated=True)

    rows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    for index, case in enumerate(cases):
        order = (False, True) if index % 2 == 0 else (True, False)
        outputs: dict[bool, dict[str, Any]] = {}
        elapsed: dict[bool, float] = {}
        for authenticated in order:
            start = time.perf_counter()
            outputs[authenticated] = await search(case, authenticated)
            elapsed[authenticated] = time.perf_counter() - start
        current_laws = outputs[False].get("laws") or []
        current_ids = _ids(current_laws)
        current_set = set(current_ids)
        for authenticated, variant in ((False, "current_unauthorized"), (True, "authenticated_rerank")):
            laws = outputs[authenticated].get("laws") or []
            ids = _ids(laws)
            rows.append(
                {
                    "case_index": index,
                    "variant": variant,
                    "elapsed_s": elapsed[authenticated],
                    "result_count": len(ids),
                    "named_law_rank": _named_law_rank(laws, case["expected_title_term"]),
                    "recall_vs_current": (
                        len(current_set.intersection(ids)) / len(current_set) if current_set else 1.0
                    ),
                    "top1_agreement": ids[:1] == current_ids[:1],
                }
            )
        print(f"[{index + 1}/{len(cases)}] rerank pair complete", file=sys.stderr, flush=True)

    current = _metrics(rows, "current_unauthorized")
    authenticated = _metrics(rows, "authenticated_rerank")
    authenticated["mean_latency_change_pct"] = (
        (authenticated["mean_s"] - current["mean_s"]) / current["mean_s"] * 100.0
    )
    return {
        "protocol": {
            "rag_root": str(rag_root),
            "named_law_question_count": len(cases),
            "paired_order": "alternating",
            "excluded_warmups_per_variant": 1,
            "label_type": "normalized expected-title substring with four explicit aliases",
            "secret_recorded": False,
        },
        "summary": {
            "current_unauthorized": current,
            "authenticated_rerank": authenticated,
        },
        "wall_s": time.perf_counter() - wall_start,
        "rows": rows if args.include_rows else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rag-root",
        default=os.environ.get("MOLEG_RAG_ROOT", "/data/project/vllm/fine-tune/2025-moleg-rag"),
    )
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--include-rows", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    result = asyncio.run(_run(args))
    rendered = json.dumps(result, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()

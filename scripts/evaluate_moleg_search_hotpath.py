#!/usr/bin/env python3
"""Ablate safe and aggressive optimizations in the MOLEG hybrid-search hot path.

The script runs against the deployed application's ``SearchEngine`` and uses
its fixed load-test query generator.  It does not copy domain questions into
this repository.  ``pipeline_cached`` and ``embedding_singleflight`` preserve
the query and both indices; ``translation_only`` and ``reduced_k`` are
aggressive comparison arms that may lose ranked documents.
"""

from __future__ import annotations

import argparse
import asyncio
from concurrent.futures import Future
import json
import logging
import os
from pathlib import Path
import statistics
import sys
from threading import Lock
import time
from typing import Any

from evaluate_moleg_search_dispatch import (
    _ids,
    _named_law_rank,
    _percentile,
    _structured_case,
)


VARIANTS = (
    "current",
    "pipeline_cached",
    "embedding_singleflight",
    "safe_hotpath",
    "translation_only",
    "reduced_k",
)


def _latency(values: list[float]) -> dict[str, float]:
    return {
        "mean_s": statistics.fmean(values) if values else 0.0,
        "median_s": statistics.median(values) if values else 0.0,
        "p95_s": _percentile(values, 95),
    }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    rag_root = Path(args.rag_root).resolve()
    sys.path.insert(0, str(rag_root))
    sys.path.insert(0, str(rag_root / "src"))

    from scripts.load_test_questions import build_test_queries
    import tools.search_tool.engine as engine_module
    import infra.vector_db.vector_store as vector_module

    logging.disable(logging.CRITICAL)
    search_engine = engine_module.search_engine
    store = search_engine.vector_store
    original_ensure = store._ensure_search_pipeline
    original_embed = store.embeddings.embed_query
    original_origin = engine_module.ENABLE_WORLD_LAW_ORIGIN_SEARCH
    original_full_k = vector_module.HYBRID_VECTOR_FULL_TOP_K
    original_para_k = vector_module.HYBRID_VECTOR_PARA_TOP_K
    original_size = vector_module.HYBRID_REQUEST_SIZE

    questions = build_test_queries()[: args.limit]
    cases = [_structured_case(question) for question in questions]
    await asyncio.to_thread(
        search_engine.search_laws,
        cases[0]["country"],
        cases[0]["keywords"],
        cases[0]["search_terms"],
    )

    async def run_variant(variant: str, case: dict[str, Any]) -> tuple[dict[str, Any], int]:
        actual_embedding_calls = 0
        store._ensure_search_pipeline = original_ensure
        store.embeddings.embed_query = original_embed
        engine_module.ENABLE_WORLD_LAW_ORIGIN_SEARCH = original_origin
        vector_module.HYBRID_VECTOR_FULL_TOP_K = original_full_k
        vector_module.HYBRID_VECTOR_PARA_TOP_K = original_para_k
        vector_module.HYBRID_REQUEST_SIZE = original_size

        if variant in {"pipeline_cached", "safe_hotpath"}:
            # Warm-up above guarantees the named pipeline exists.  Production
            # integration should use a process-level once/TTL guard.
            store._ensure_search_pipeline = lambda: None

        if variant in {"embedding_singleflight", "safe_hotpath"}:
            futures: dict[str, Future[list[float]]] = {}
            lock = Lock()

            def embed_once(text: str):
                nonlocal actual_embedding_calls
                with lock:
                    future = futures.get(text)
                    owner = future is None
                    if future is None:
                        future = Future()
                        futures[text] = future
                if owner:
                    try:
                        actual_embedding_calls += 1
                        future.set_result(original_embed(text))
                    except BaseException as exc:
                        future.set_exception(exc)
                return future.result()

            store.embeddings.embed_query = embed_once

        if variant == "translation_only":
            engine_module.ENABLE_WORLD_LAW_ORIGIN_SEARCH = False
        elif variant == "reduced_k":
            vector_module.HYBRID_VECTOR_FULL_TOP_K = args.reduced_vector_k
            vector_module.HYBRID_VECTOR_PARA_TOP_K = args.reduced_vector_k
            vector_module.HYBRID_REQUEST_SIZE = args.reduced_request_size

        try:
            result = await asyncio.to_thread(
                search_engine.search_laws,
                case["country"],
                case["keywords"],
                case["search_terms"],
            )
            if variant not in {"embedding_singleflight", "safe_hotpath"}:
                actual_embedding_calls = 1 if variant == "translation_only" else 2
            return result, actual_embedding_calls
        finally:
            store._ensure_search_pipeline = original_ensure
            store.embeddings.embed_query = original_embed
            engine_module.ENABLE_WORLD_LAW_ORIGIN_SEARCH = original_origin
            vector_module.HYBRID_VECTOR_FULL_TOP_K = original_full_k
            vector_module.HYBRID_VECTOR_PARA_TOP_K = original_para_k
            vector_module.HYBRID_REQUEST_SIZE = original_size

    rows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    for index, case in enumerate(cases):
        rotated = VARIANTS[index % len(VARIANTS) :] + VARIANTS[: index % len(VARIANTS)]
        outputs: dict[str, dict[str, Any]] = {}
        elapsed: dict[str, float] = {}
        embedding_calls: dict[str, int] = {}
        for variant in rotated:
            start = time.perf_counter()
            outputs[variant], embedding_calls[variant] = await run_variant(variant, case)
            elapsed[variant] = time.perf_counter() - start

        baseline_laws = outputs["current"].get("laws") or []
        baseline_ids = _ids(baseline_laws)
        baseline_set = set(baseline_ids)
        for variant in VARIANTS:
            laws = outputs[variant].get("laws") or []
            ids = _ids(laws)
            overlap = len(baseline_set.intersection(ids))
            expected = case["expected_title_term"]
            rows.append(
                {
                    "case_index": index,
                    "variant": variant,
                    "elapsed_s": elapsed[variant],
                    "embedding_calls": embedding_calls[variant],
                    "result_count": len(ids),
                    "exact_ranked_ids": ids == baseline_ids,
                    "recall_vs_current": overlap / len(baseline_set) if baseline_set else 1.0,
                    "top1_agreement": ids[:1] == baseline_ids[:1],
                    "named_law_rank": _named_law_rank(laws, expected) if expected else None,
                    "country_precision": (
                        sum(str(row.get("country") or "").strip() == case["country"] for row in laws)
                        / len(laws)
                        if laws
                        else 1.0
                    ),
                    "named_law": case["named_law"],
                }
            )
        print(f"[{index + 1}/{len(cases)}] hot-path variants complete", file=sys.stderr, flush=True)

    summary: dict[str, Any] = {}
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        named = [row for row in selected if row["named_law"]]
        summary[variant] = {
            **_latency([row["elapsed_s"] for row in selected]),
            "mean_embedding_calls": statistics.fmean(row["embedding_calls"] for row in selected),
            "exact_ranked_id_rate": statistics.fmean(row["exact_ranked_ids"] for row in selected),
            "recall_vs_current": statistics.fmean(row["recall_vs_current"] for row in selected),
            "top1_agreement": statistics.fmean(row["top1_agreement"] for row in selected),
            "country_precision": statistics.fmean(row["country_precision"] for row in selected),
            "named_law_hit_at_20": (
                statistics.fmean(row["named_law_rank"] is not None for row in named) if named else None
            ),
            "named_law_mrr_at_20": (
                statistics.fmean(
                    1.0 / row["named_law_rank"] if row["named_law_rank"] else 0.0
                    for row in named
                )
                if named
                else None
            ),
        }

    current_mean = summary["current"]["mean_s"]
    for variant in VARIANTS[1:]:
        summary[variant]["mean_latency_reduction_pct"] = (
            (current_mean - summary[variant]["mean_s"]) / current_mean * 100.0
            if current_mean
            else 0.0
        )

    return {
        "protocol": {
            "rag_root": str(rag_root),
            "question_count": len(cases),
            "paired_order": "six-way rotating",
            "excluded_warmups": 1,
            "reduced_vector_k": args.reduced_vector_k,
            "reduced_request_size": args.reduced_request_size,
        },
        "summary": summary,
        "wall_s": time.perf_counter() - wall_start,
        "rows": rows if args.include_rows else [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rag-root",
        default=os.environ.get("MOLEG_RAG_ROOT", "/data/project/vllm/fine-tune/2025-moleg-rag"),
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--reduced-vector-k", type=int, default=40)
    parser.add_argument("--reduced-request-size", type=int, default=24)
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

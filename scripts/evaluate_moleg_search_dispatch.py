#!/usr/bin/env python3
"""Measure redundant LLM search dispatch against direct typed dispatch.

This adapter intentionally imports the deployed ``2025-moleg-rag`` source at
runtime.  The benchmark keeps retrieval, OpenSearch indices, embeddings,
thesaurus expansion, and result formatting unchanged.  Only the already-known
``search_laws`` invocation is dispatched differently:

* ``llm_auto``: the application's current ``law_search_agent.run_search``;
* ``typed_direct``: the same ``tool_search_laws`` handler with validated args.

The deployed query generator supplies the question set, so this repository does
not copy application prompts or domain data.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import math
import os
import statistics
import sys
import time
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Any


def _percentile(values: Sequence[float], percentile: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * percentile / 100.0
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(ordered[lower])
    fraction = position - lower
    return float(ordered[lower] + (ordered[upper] - ordered[lower]) * fraction)


def _summarize_latency(values: Sequence[float]) -> dict[str, float]:
    return {
        "mean_s": statistics.fmean(values) if values else 0.0,
        "median_s": statistics.median(values) if values else 0.0,
        "p95_s": _percentile(values, 95),
        "min_s": min(values, default=0.0),
        "max_s": max(values, default=0.0),
    }


def _structured_case(question: str) -> dict[str, Any]:
    """Derive stable search arguments from the deployed fixed-query wording."""
    country = ""
    term = question.strip()
    if "에서 " in question:
        country, term = question.split("에서 ", 1)
        term = term.removesuffix(" 하려는데 관련 법령 찾아주세요").strip()
    elif "의 " in question:
        country, term = question.split("의 ", 1)
        term = term.removesuffix(" 찾아주세요").strip()
    return {
        "question": question,
        "country": country.strip(),
        "search_terms": term,
        "keywords": [term] if term else [],
        "named_law": "의 " in question,
        "expected_title_term": term if "의 " in question else "",
    }


def _ids(laws: Sequence[dict[str, Any]]) -> list[str]:
    return [str(row.get("id") or row.get("law_id") or "").strip() for row in laws]


def _normal(text: str) -> str:
    return "".join(ch.lower() for ch in str(text) if ch.isalnum())


def _named_law_rank(
    laws: Sequence[dict[str, Any]], expected: str, *, cutoff: int = 20
) -> int | None:
    needle = _normal(expected)
    aliases = {
        "pipeda": ("pipeda", "개인정보보호및전자문서법"),
        "lgpd": ("lgpd", "일반개인정보보호법"),
        "popia": ("popia", "개인정보보호법"),
        "gdpr집행관련법": ("gdpr", "개인정보보호"),
    }
    accepted = aliases.get(needle, (needle,))
    for index, law in enumerate(laws[:cutoff], start=1):
        title = _normal(str(law.get("title") or ""))
        if any(token and token in title for token in accepted):
            return index
    return None


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    rag_root = Path(args.rag_root).resolve()
    sys.path.insert(0, str(rag_root))
    sys.path.insert(0, str(rag_root / "src"))

    from scripts.load_test_questions import build_test_queries
    from agent import law_search_agent
    from infra.llm.law_tool_executors import tool_search_laws

    # The application already logs detailed tool payloads.  The benchmark JSON
    # is the durable record, so keep console output small and secret-safe.
    logging.disable(logging.CRITICAL)

    questions = build_test_queries()[: args.limit]
    cases = [_structured_case(question) for question in questions]
    tracker: dict[str, Any] = {"calls": 0, "tool_calls": 0}
    original_loop = law_search_agent.run_tool_loop

    async def tracking_loop(*loop_args: Any, **loop_kwargs: Any):
        tracker["calls"] += 1
        result = await original_loop(*loop_args, **loop_kwargs)
        trace = result[2] if len(result) >= 3 else []
        tracker["tool_calls"] += len(trace)
        return result

    law_search_agent.run_tool_loop = tracking_loop

    async def llm_auto(case: dict[str, Any]) -> dict[str, Any]:
        return await law_search_agent.run_search(
            country=case["country"],
            merged_keywords=case["keywords"],
            search_terms=case["search_terms"],
            request_id="dispatch-auto-" + uuid.uuid4().hex[:8],
        )

    async def typed_direct(case: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "country": case["country"] or None,
            "keywords": list(case["keywords"]),
            "search_terms": case["search_terms"],
        }
        if not payload["search_terms"] and not payload["keywords"]:
            raise ValueError("search_terms and keywords cannot both be empty")
        raw = await tool_search_laws(
            payload,
            "dispatch-direct-" + uuid.uuid4().hex[:8],
        )
        laws, search_law_ids, source = law_search_agent._parse_search_payload(raw)
        return {
            "search_tool_result": raw,
            "laws_list": laws,
            "search_law_ids": search_law_ids,
            "search_source": source,
        }

    # Excluded warm-ups initialize clients, DB pools, and OpenSearch connections.
    for warmup_index in range(args.warmups):
        case = cases[warmup_index % len(cases)]
        await llm_auto(case)
        await typed_direct(case)
    tracker["calls"] = 0
    tracker["tool_calls"] = 0

    rows: list[dict[str, Any]] = []
    wall_start = time.perf_counter()
    for repeat in range(args.repeats):
        for index, case in enumerate(cases):
            order = ("llm_auto", "typed_direct")
            if (index + repeat) % 2:
                order = tuple(reversed(order))
            outputs: dict[str, dict[str, Any]] = {}
            elapsed: dict[str, float] = {}
            for variant in order:
                start = time.perf_counter()
                outputs[variant] = await (
                    llm_auto(case) if variant == "llm_auto" else typed_direct(case)
                )
                elapsed[variant] = time.perf_counter() - start

            baseline_laws = outputs["llm_auto"].get("laws_list") or []
            direct_laws = outputs["typed_direct"].get("laws_list") or []
            baseline_ids = _ids(baseline_laws)
            direct_ids = _ids(direct_laws)
            baseline_set = set(baseline_ids)
            overlap = len(baseline_set.intersection(direct_ids))
            expected = case["expected_title_term"]
            rows.append(
                {
                    "repeat": repeat,
                    "case_index": index,
                    "country": case["country"],
                    "named_law": case["named_law"],
                    "llm_auto_s": elapsed["llm_auto"],
                    "typed_direct_s": elapsed["typed_direct"],
                    "saved_s": elapsed["llm_auto"] - elapsed["typed_direct"],
                    "baseline_count": len(baseline_ids),
                    "direct_count": len(direct_ids),
                    "exact_ranked_ids": baseline_ids == direct_ids,
                    "recall_vs_baseline": overlap / len(baseline_set) if baseline_set else 1.0,
                    "top1_agreement": baseline_ids[:1] == direct_ids[:1],
                    "baseline_named_law_rank": _named_law_rank(baseline_laws, expected) if expected else None,
                    "direct_named_law_rank": _named_law_rank(direct_laws, expected) if expected else None,
                    "baseline_country_precision": (
                        sum(str(row.get("country") or "").strip() == case["country"] for row in baseline_laws)
                        / len(baseline_laws)
                        if baseline_laws
                        else 1.0
                    ),
                    "direct_country_precision": (
                        sum(str(row.get("country") or "").strip() == case["country"] for row in direct_laws)
                        / len(direct_laws)
                        if direct_laws
                        else 1.0
                    ),
                }
            )
            print(
                f"[{len(rows)}/{len(cases) * args.repeats}] "
                f"auto={elapsed['llm_auto']:.3f}s direct={elapsed['typed_direct']:.3f}s "
                f"exact={baseline_ids == direct_ids}",
                file=sys.stderr,
                flush=True,
            )

    llm_times = [row["llm_auto_s"] for row in rows]
    direct_times = [row["typed_direct_s"] for row in rows]
    named = [row for row in rows if row["named_law"]]
    result = {
        "protocol": {
            "rag_root": str(rag_root),
            "question_count": len(cases),
            "repeats": args.repeats,
            "excluded_warmups_per_variant": args.warmups,
            "paired_order": "alternating",
            "retrieval_stack_changed": False,
        },
        "llm_auto": {
            **_summarize_latency(llm_times),
            "llm_calls": tracker["calls"],
            "observed_tool_calls": tracker["tool_calls"],
            "country_precision": statistics.fmean(row["baseline_country_precision"] for row in rows),
            "named_law_hit_at_20": (
                statistics.fmean(row["baseline_named_law_rank"] is not None for row in named) if named else None
            ),
            "named_law_mrr_at_20": (
                statistics.fmean(
                    1.0 / row["baseline_named_law_rank"] if row["baseline_named_law_rank"] else 0.0
                    for row in named
                )
                if named
                else None
            ),
        },
        "typed_direct": {
            **_summarize_latency(direct_times),
            "llm_calls": 0,
            "country_precision": statistics.fmean(row["direct_country_precision"] for row in rows),
            "named_law_hit_at_20": (
                statistics.fmean(row["direct_named_law_rank"] is not None for row in named) if named else None
            ),
            "named_law_mrr_at_20": (
                statistics.fmean(
                    1.0 / row["direct_named_law_rank"] if row["direct_named_law_rank"] else 0.0
                    for row in named
                )
                if named
                else None
            ),
        },
        "comparison": {
            "mean_saved_s": statistics.fmean(row["saved_s"] for row in rows),
            "mean_latency_reduction_pct": (
                (statistics.fmean(llm_times) - statistics.fmean(direct_times))
                / statistics.fmean(llm_times)
                * 100.0
            ),
            "exact_ranked_id_rate": statistics.fmean(row["exact_ranked_ids"] for row in rows),
            "mean_recall_vs_baseline": statistics.fmean(row["recall_vs_baseline"] for row in rows),
            "top1_agreement": statistics.fmean(row["top1_agreement"] for row in rows),
        },
        "wall_s": time.perf_counter() - wall_start,
        "rows": rows if args.include_rows else [],
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rag-root",
        default=os.environ.get("MOLEG_RAG_ROOT", "/data/project/vllm/fine-tune/2025-moleg-rag"),
    )
    parser.add_argument("--limit", type=int, default=50)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--warmups", type=int, default=1)
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

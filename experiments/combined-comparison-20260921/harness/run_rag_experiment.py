#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import hashlib
import json
import math
import os
import random
import statistics
import time
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Sequence

import httpx

from collect_vllm_metrics import parse_targets, sample_vllm_metrics, write_metrics


@dataclass(frozen=True)
class Question:
    question_id: str
    query: str
    category: str
    complexity: str
    expected_law_ids: tuple[str, ...] = ()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def _safe_id(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "-_." else "-" for ch in value)[:100]


def load_questions(path: Path) -> list[Question]:
    questions: list[Question] = []
    seen: set[str] = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            question_id = str(row.get("id") or "").strip()
            query = str(row.get("query") or "").strip()
            if not question_id or not query:
                raise ValueError(f"{path}:{line_no}: id and query are required")
            if question_id in seen:
                raise ValueError(f"{path}:{line_no}: duplicate id {question_id!r}")
            seen.add(question_id)
            questions.append(
                Question(
                    question_id=question_id,
                    query=query,
                    category=str(row.get("category") or "unspecified"),
                    complexity=str(row.get("complexity") or "unspecified"),
                    expected_law_ids=tuple(
                        str(item) for item in (row.get("expected_law_ids") or [])
                    ),
                )
            )
    if not questions:
        raise ValueError(f"question set is empty: {path}")
    return questions


def percentile(values: Iterable[float], p: float) -> float | None:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    if len(ordered) == 1:
        return ordered[0]
    position = (len(ordered) - 1) * p / 100.0
    low = math.floor(position)
    high = math.ceil(position)
    if low == high:
        return ordered[low]
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def _response_metadata(result: Any, *, store_response: bool) -> dict[str, Any]:
    if not isinstance(result, dict):
        return {"response_type": type(result).__name__}
    laws = result.get("laws") or []
    law_ids: list[str] = []
    if isinstance(laws, list):
        for law in laws:
            if isinstance(law, dict):
                law_id = str(law.get("law_id") or law.get("item_id") or "").strip()
                if law_id:
                    law_ids.append(law_id)
    comment = str(result.get("comment") or "")
    canonical = json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
    metadata: dict[str, Any] = {
        "law_count": len(laws) if isinstance(laws, list) else 0,
        "law_ids": law_ids,
        "comment_chars": len(comment),
        "response_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
    }
    if store_response:
        metadata["response"] = result
    return metadata


async def _request_stream(
    client: httpx.AsyncClient,
    *,
    url: str,
    question: Question,
    run_id: str,
    variant: str,
    case_id: str,
    session_id: str,
    api_key: str,
    store_response: bool,
) -> dict[str, Any]:
    headers = {
        "X-Experiment-Run-ID": run_id,
        "X-Experiment-Variant": variant,
        "X-Experiment-Case-ID": case_id,
    }
    if api_key:
        headers["X-API-Key"] = api_key
    payload = {"prompt": question.query, "session_id": session_id}
    started_at = _utc_now()
    started_perf = time.perf_counter()
    first_event_ms: float | None = None
    first_token_ms: float | None = None
    done_result: Any = None
    error_event: str | None = None
    status_code: int | None = None
    try:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            status_code = response.status_code
            if response.status_code >= 400:
                body = (await response.aread()).decode("utf-8", errors="replace")
                raise RuntimeError(f"HTTP {response.status_code}: {body[:500]}")
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                now_ms = (time.perf_counter() - started_perf) * 1000.0
                if first_event_ms is None:
                    first_event_ms = now_ms
                raw_event = line[5:].strip()
                if not raw_event:
                    continue
                event = json.loads(raw_event)
                stage = event.get("stage")
                if stage == "token" and first_token_ms is None:
                    first_token_ms = now_ms
                elif stage == "done":
                    done_result = event.get("result")
                elif stage == "error":
                    error_event = str(event.get("message") or "server stream error")
    except Exception as exc:
        duration_ms = (time.perf_counter() - started_perf) * 1000.0
        return {
            **asdict(question),
            "case_id": case_id,
            "session_id": session_id,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "status": "error",
            "status_code": status_code,
            "duration_ms": round(duration_ms, 3),
            "first_event_ms": first_event_ms,
            "ttft_ms": first_token_ms,
            "error_type": type(exc).__name__,
            "error": str(exc)[:800],
        }
    duration_ms = (time.perf_counter() - started_perf) * 1000.0
    row = {
        **asdict(question),
        "case_id": case_id,
        "session_id": session_id,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "status": "error" if error_event else "ok",
        "status_code": status_code,
        "duration_ms": round(duration_ms, 3),
        "first_event_ms": round(first_event_ms, 3) if first_event_ms is not None else None,
        "ttft_ms": round(first_token_ms, 3) if first_token_ms is not None else None,
    }
    if error_event:
        row["error"] = error_event
    row.update(_response_metadata(done_result, store_response=store_response))
    return row


async def _request_json(
    client: httpx.AsyncClient,
    *,
    url: str,
    question: Question,
    run_id: str,
    variant: str,
    case_id: str,
    session_id: str,
    api_key: str,
    store_response: bool,
) -> dict[str, Any]:
    headers = {
        "X-Experiment-Run-ID": run_id,
        "X-Experiment-Variant": variant,
        "X-Experiment-Case-ID": case_id,
    }
    if api_key:
        headers["X-API-Key"] = api_key
    started_at = _utc_now()
    started_perf = time.perf_counter()
    status_code: int | None = None
    try:
        response = await client.post(
            url,
            json={"prompt": question.query, "session_id": session_id},
            headers=headers,
        )
        status_code = response.status_code
        response.raise_for_status()
        result = response.json()
    except Exception as exc:
        return {
            **asdict(question),
            "case_id": case_id,
            "session_id": session_id,
            "started_at": started_at,
            "completed_at": _utc_now(),
            "status": "error",
            "status_code": status_code,
            "duration_ms": round((time.perf_counter() - started_perf) * 1000.0, 3),
            "first_event_ms": None,
            "ttft_ms": None,
            "error_type": type(exc).__name__,
            "error": str(exc)[:800],
        }
    row = {
        **asdict(question),
        "case_id": case_id,
        "session_id": session_id,
        "started_at": started_at,
        "completed_at": _utc_now(),
        "status": "ok",
        "status_code": status_code,
        "duration_ms": round((time.perf_counter() - started_perf) * 1000.0, 3),
        "first_event_ms": None,
        "ttft_ms": None,
    }
    row.update(_response_metadata(result, store_response=store_response))
    return row


def summarize(rows: Sequence[dict[str, Any]], wall_seconds: float) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "ok"]
    latencies = [float(row["duration_ms"]) for row in successful]
    ttfts = [float(row["ttft_ms"]) for row in successful if row.get("ttft_ms") is not None]
    return {
        "requests": len(rows),
        "successes": len(successful),
        "errors": len(rows) - len(successful),
        "error_rate": (len(rows) - len(successful)) / len(rows) if rows else 0.0,
        "wall_seconds": round(wall_seconds, 3),
        "throughput_rps": round(len(successful) / wall_seconds, 6) if wall_seconds else 0.0,
        "latency_mean_ms": round(statistics.mean(latencies), 3) if latencies else None,
        "latency_p50_ms": percentile(latencies, 50),
        "latency_p95_ms": percentile(latencies, 95),
        "latency_p99_ms": percentile(latencies, 99),
        "ttft_p50_ms": percentile(ttfts, 50),
        "ttft_p95_ms": percentile(ttfts, 95),
        "ttft_samples": len(ttfts),
    }


def _row_key(row: dict[str, Any]) -> tuple[str, int, int]:
    return (
        str(row.get("question_id") or ""),
        int(row.get("repeat") or 0),
        int(row.get("concurrency") or 0),
    )


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid checkpoint JSON: {exc}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: checkpoint row must be an object")
            rows.append(row)
    return rows


def _dedupe_rows(rows: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    latest: dict[tuple[str, int, int], dict[str, Any]] = {}
    for row in rows:
        key = _row_key(row)
        previous = latest.get(key)
        if previous is None or row.get("status") == "ok" or previous.get("status") != "ok":
            latest[key] = row
    return list(latest.values())


def _observed_wall_seconds(rows: Sequence[dict[str, Any]]) -> float:
    by_attempt: dict[str, list[tuple[datetime, datetime]]] = {}
    for row in rows:
        try:
            started = datetime.fromisoformat(str(row["started_at"]))
            completed = datetime.fromisoformat(str(row["completed_at"]))
        except (KeyError, TypeError, ValueError):
            continue
        attempt_id = str(row.get("attempt_id") or "legacy")
        by_attempt.setdefault(attempt_id, []).append((started, completed))
    return sum(
        (max(end for _, end in intervals) - min(start for start, _ in intervals)).total_seconds()
        for intervals in by_attempt.values()
        if intervals
    )


async def run_cell(
    *,
    client: httpx.AsyncClient,
    url: str,
    stream: bool,
    questions: Sequence[Question],
    run_id: str,
    variant: str,
    concurrency: int,
    repeats: int,
    api_key: str,
    store_response: bool,
    rng: random.Random,
    attempt_id: str,
    checkpoint_path: Path,
    completed_keys: set[tuple[str, int, int]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    cases = [
        (question, repeat)
        for repeat in range(1, repeats + 1)
        for question in questions
    ]
    rng.shuffle(cases)
    cases = [
        (question, repeat)
        for question, repeat in cases
        if (question.question_id, repeat, concurrency) not in completed_keys
    ]
    semaphore = asyncio.Semaphore(concurrency)
    checkpoint_lock = asyncio.Lock()
    request_fn = _request_stream if stream else _request_json

    async def one(question: Question, repeat: int) -> dict[str, Any]:
        case_id = _safe_id(f"{question.question_id}-r{repeat}-c{concurrency}")
        session_id = _safe_id(f"exp-{run_id}-{case_id}-{uuid.uuid4().hex[:8]}")
        async with semaphore:
            row = await request_fn(
                client,
                url=url,
                question=question,
                run_id=run_id,
                variant=variant,
                case_id=case_id,
                session_id=session_id,
                api_key=api_key,
                store_response=store_response,
            )
            row["repeat"] = repeat
            row["concurrency"] = concurrency
            row["attempt_id"] = attempt_id
            async with checkpoint_lock:
                with checkpoint_path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")
                    handle.flush()
            return row

    wall_start = time.perf_counter()
    rows = await asyncio.gather(*(one(question, repeat) for question, repeat in cases))
    wall_seconds = time.perf_counter() - wall_start
    summary = summarize(rows, wall_seconds)
    summary["concurrency"] = concurrency
    return rows, summary


def _write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, default=str) + "\n")


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    fields = [
        "case_id",
        "question_id",
        "category",
        "complexity",
        "repeat",
        "concurrency",
        "status",
        "status_code",
        "started_at",
        "completed_at",
        "first_event_ms",
        "ttft_ms",
        "duration_ms",
        "law_count",
        "comment_chars",
        "response_sha256",
        "error_type",
        "error",
    ]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


async def amain(args: argparse.Namespace) -> int:
    question_path = Path(args.questions).resolve()
    questions = load_questions(question_path)
    if args.complexity:
        allowed_complexities = {
            item.strip() for item in args.complexity.split(",") if item.strip()
        }
        questions = [q for q in questions if q.complexity in allowed_complexities]
        if not questions:
            raise ValueError(f"no questions match complexity={sorted(allowed_complexities)}")
    if args.limit:
        questions = questions[: args.limit]
    run_id = _safe_id(args.run_id or datetime.now().strftime("%Y%m%dT%H%M%S"))
    variant = _safe_id(args.variant)
    output_dir = Path(args.output_dir).resolve() / run_id / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output_dir / "client_requests.checkpoint.jsonl"
    if checkpoint_path.exists() and not args.resume:
        raise ValueError(
            f"checkpoint already exists: {checkpoint_path}; pass --resume or use a new run id"
        )
    base_url = args.base_url.rstrip("/")
    endpoint = args.endpoint if args.endpoint.startswith("/") else f"/{args.endpoint}"
    url = f"{base_url}{endpoint}"
    stream = endpoint.endswith("/stream")
    api_key = os.getenv(args.api_key_env, "") if args.api_key_env else ""
    concurrencies = [int(item) for item in args.concurrency.split(",")]
    if any(item < 1 for item in concurrencies):
        raise ValueError("concurrency values must be >= 1")

    timeout = httpx.Timeout(args.timeout, connect=min(15.0, args.timeout))
    limits = httpx.Limits(
        max_connections=max(concurrencies) + 10,
        max_keepalive_connections=max(concurrencies) + 10,
    )
    rng = random.Random(args.seed)
    all_rows = _dedupe_rows(_load_jsonl(checkpoint_path) if args.resume else [])
    summaries: list[dict[str, Any]] = []
    attempt_id = uuid.uuid4().hex[:12]
    metrics_stop = asyncio.Event()
    metrics_task: asyncio.Task[list[dict[str, Any]]] | None = None
    if args.vllm_targets:
        metrics_task = asyncio.create_task(
            sample_vllm_metrics(
                parse_targets(args.vllm_targets),
                interval_seconds=args.metrics_interval,
                stop_event=metrics_stop,
            )
        )
    async with httpx.AsyncClient(timeout=timeout, limits=limits) as client:
        for warmup_index in range(args.warmup):
            question = questions[warmup_index % len(questions)]
            await (_request_stream if stream else _request_json)(
                client,
                url=url,
                question=question,
                run_id=f"{run_id}-warmup",
                variant=variant,
                case_id=f"warmup-{warmup_index + 1}",
                session_id=f"exp-warmup-{uuid.uuid4().hex}",
                api_key=api_key,
                store_response=False,
            )
        for concurrency in concurrencies:
            completed_keys = {
                _row_key(row)
                for row in all_rows
                if row.get("status") == "ok"
            }
            rows, cell_summary = await run_cell(
                client=client,
                url=url,
                stream=stream,
                questions=questions,
                run_id=run_id,
                variant=variant,
                concurrency=concurrency,
                repeats=args.repeats,
                api_key=api_key,
                store_response=args.store_responses,
                rng=rng,
                attempt_id=attempt_id,
                checkpoint_path=checkpoint_path,
                completed_keys=completed_keys,
            )
            all_rows = _dedupe_rows([*all_rows, *rows])
            cell_rows = [
                row for row in all_rows if int(row.get("concurrency") or 0) == concurrency
            ]
            cell_summary = summarize(cell_rows, _observed_wall_seconds(cell_rows))
            cell_summary["concurrency"] = concurrency
            cell_summary["attempts"] = len(
                {str(row.get("attempt_id") or "legacy") for row in cell_rows}
            )
            summaries.append(cell_summary)
            print(json.dumps(cell_summary, ensure_ascii=False), flush=True)

    metrics_stop.set()
    if metrics_task is not None:
        write_metrics(output_dir / "vllm_metrics.csv", await metrics_task)

    metadata = {
        "schema_version": 1,
        "generated_at": _utc_now(),
        "run_id": run_id,
        "variant": variant,
        "base_url": base_url,
        "endpoint": endpoint,
        "questions": str(question_path),
        "question_count": len(questions),
        "complexity_filter": args.complexity or None,
        "repeats": args.repeats,
        "warmup": args.warmup,
        "concurrencies": concurrencies,
        "seed": args.seed,
        "store_responses": args.store_responses,
        "vllm_targets": sorted(parse_targets(args.vllm_targets)) if args.vllm_targets else [],
        "metrics_interval_seconds": args.metrics_interval if args.vllm_targets else None,
        "cells": summaries,
    }
    _write_jsonl(output_dir / "client_requests.jsonl", all_rows)
    _write_csv(output_dir / "client_requests.csv", all_rows)
    (output_dir / "summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"results={output_dir}")
    return 0 if all(row.get("status") == "ok" for row in all_rows) else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://127.0.0.1:28100")
    parser.add_argument("--endpoint", default="/api/generate/stream")
    parser.add_argument("--questions", required=True)
    parser.add_argument("--limit", type=int, default=0, help="0 means all questions")
    parser.add_argument("--complexity", default="", help="optional comma-separated levels, e.g. L1,L2")
    parser.add_argument("--output-dir", default="experiment-results")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--variant", default="B0")
    parser.add_argument("--concurrency", default="1,4,8")
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=600.0)
    parser.add_argument("--seed", type=int, default=20250809)
    parser.add_argument("--api-key-env", default=os.environ.get("CNU_APP_KEY_ATTR", "APP_API_KEY"))
    parser.add_argument("--store-responses", action="store_true")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="reuse successful checkpoint rows and rerun only missing or failed cases",
    )
    parser.add_argument(
        "--vllm-targets",
        default="",
        help="optional Prometheus targets: name=url,name=url",
    )
    parser.add_argument("--metrics-interval", type=float, default=0.5)
    args = parser.parse_args(argv)
    if args.repeats < 1 or args.warmup < 0 or args.metrics_interval <= 0 or args.limit < 0:
        parser.error("repeats >= 1, warmup >= 0, metrics-interval > 0 required")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain(parse_args())))

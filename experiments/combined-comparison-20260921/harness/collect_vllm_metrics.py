#!/usr/bin/env python3
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

import httpx


METRIC_BASES = {
    "vllm:num_requests_running",
    "vllm:num_requests_waiting",
    "vllm:kv_cache_usage_perc",
    "vllm:num_preemptions_total",
    "vllm:prompt_tokens_total",
    "vllm:generation_tokens_total",
    "vllm:prefix_cache_queries_total",
    "vllm:prefix_cache_hits_total",
    "vllm:time_to_first_token_seconds",
    "vllm:inter_token_latency_seconds",
    "vllm:request_time_per_output_token_seconds",
    "vllm:e2e_request_latency_seconds",
    "vllm:request_queue_time_seconds",
    "vllm:request_inference_time_seconds",
    "vllm:request_prefill_time_seconds",
    "vllm:request_decode_time_seconds",
}
_SAMPLE_RE = re.compile(
    r"^(?P<name>[A-Za-z_:][A-Za-z0-9_:]*)(?:\{(?P<labels>.*)\})?\s+"
    r"(?P<value>[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|[-+]?Inf|NaN)$"
)
_LABEL_RE = re.compile(r'([A-Za-z_][A-Za-z0-9_]*)="((?:\\.|[^"\\])*)"')


def parse_targets(raw: str) -> dict[str, str]:
    targets: dict[str, str] = {}
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"target must be name=url: {item!r}")
        name, url = item.split("=", 1)
        name = name.strip()
        url = url.strip()
        if not name or not url.startswith(("http://", "https://")):
            raise ValueError(f"invalid target: {item!r}")
        targets[name] = url if url.endswith("/metrics") else f"{url.rstrip('/')}/metrics"
    if not targets:
        raise ValueError("at least one metrics target is required")
    return targets


def _metric_base(metric_name: str) -> str:
    for suffix in ("_bucket", "_sum", "_count", "_created"):
        if metric_name.endswith(suffix):
            return metric_name[: -len(suffix)]
    return metric_name


def parse_prometheus(text: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        match = _SAMPLE_RE.match(line)
        if not match:
            continue
        name = match.group("name")
        if _metric_base(name) not in METRIC_BASES or name.endswith("_created"):
            continue
        labels: dict[str, str] = {}
        for label_match in _LABEL_RE.finditer(match.group("labels") or ""):
            labels[label_match.group(1)] = bytes(
                label_match.group(2), "utf-8"
            ).decode("unicode_escape")
        rows.append(
            {
                "metric": name,
                "labels": labels,
                "value": float(match.group("value")),
            }
        )
    return rows


async def _fetch_one(
    client: httpx.AsyncClient,
    target_name: str,
    url: str,
) -> tuple[str, list[dict[str, Any]], str | None]:
    try:
        response = await client.get(url)
        response.raise_for_status()
        return target_name, parse_prometheus(response.text), None
    except Exception as exc:
        return target_name, [], f"{type(exc).__name__}: {str(exc)[:300]}"


async def sample_vllm_metrics(
    targets: Mapping[str, str],
    *,
    interval_seconds: float,
    stop_event: asyncio.Event,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=httpx.Timeout(8.0, connect=3.0)) as client:
        while not stop_event.is_set():
            sample_perf = time.perf_counter()
            sampled_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
            results = await asyncio.gather(
                *(
                    _fetch_one(client, target_name, url)
                    for target_name, url in targets.items()
                )
            )
            for target_name, metrics, error in results:
                if error:
                    rows.append(
                        {
                            "sampled_at": sampled_at,
                            "target": target_name,
                            "metric": "collector_error",
                            "labels": {},
                            "value": None,
                            "error": error,
                        }
                    )
                for metric in metrics:
                    rows.append(
                        {
                            "sampled_at": sampled_at,
                            "target": target_name,
                            **metric,
                        }
                    )
            wait_seconds = max(
                0.0, interval_seconds - (time.perf_counter() - sample_perf)
            )
            try:
                await asyncio.wait_for(stop_event.wait(), timeout=wait_seconds)
            except asyncio.TimeoutError:
                pass
    return rows


def write_metrics(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["sampled_at", "target", "metric", "labels", "value", "error"],
            extrasaction="ignore",
        )
        writer.writeheader()
        for row in rows:
            output = dict(row)
            output["labels"] = json.dumps(output.get("labels") or {}, ensure_ascii=False)
            writer.writerow(output)


async def amain(args: argparse.Namespace) -> int:
    targets = parse_targets(args.targets)
    stop_event = asyncio.Event()

    async def stop_later() -> None:
        await asyncio.sleep(args.duration)
        stop_event.set()

    stopper = asyncio.create_task(stop_later())
    rows = await sample_vllm_metrics(
        targets,
        interval_seconds=args.interval,
        stop_event=stop_event,
    )
    await stopper
    output = Path(args.output).resolve()
    write_metrics(output, rows)
    errors = sum(1 for row in rows if row.get("error"))
    print(f"samples={len(rows)} errors={errors} output={output}")
    return 0 if not errors else 2


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", required=True, help="name=url,name=url")
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--duration", type=float, required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args(argv)
    if args.interval <= 0 or args.duration <= 0:
        parser.error("--interval and --duration must be > 0")
    return args


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain(parse_args())))

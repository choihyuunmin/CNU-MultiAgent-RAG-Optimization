"""Prompt-free Prometheus and load-test metrics; no missing-value zero filling."""
from __future__ import annotations

from collections import defaultdict
import math
import re
import statistics


GAUGES = {"num_requests_running", "num_requests_waiting", "kv_cache_usage_perc",
          "gpu_cache_usage_perc"}
COUNTERS = {"num_preemptions_total", "prompt_tokens_total", "generation_tokens_total",
            "prefix_cache_queries_total", "prefix_cache_hits_total"}
HISTOGRAMS = {"request_queue_time_seconds", "request_prefill_time_seconds",
              "request_decode_time_seconds", "request_inference_time_seconds",
              "time_to_first_token_seconds", "inter_token_latency_seconds",
              "e2e_request_latency_seconds", "request_prompt_tokens",
              "request_generation_tokens", "request_prefill_kv_computed_tokens"}
ALLOWED = GAUGES | COUNTERS | {h + s for h in HISTOGRAMS for s in ("_sum", "_count")}


def parse_metrics(payload):
    values = defaultdict(float)
    configs = []
    for line in payload.splitlines():
        match = re.match(r'^vllm:([\w]+)(?:\{([^}]*)\})?\s+([^\s]+)', line)
        if not match:
            continue
        name, labels, number = match.groups()
        if name == "cache_config_info":
            entries = dict(re.findall(r'(\w+)="([^"\\]*)"', labels or ""))
            configs.append({k: entries[k] for k in (
                "block_size", "num_gpu_blocks", "gpu_memory_utilization",
                "cache_dtype", "enable_prefix_caching", "sliding_window"
            ) if k in entries})
        if name not in ALLOWED:
            continue
        try:
            value = float(number)
        except ValueError:
            continue
        if math.isfinite(value):
            # Capacity pressure is the worst engine; counters cover all engines.
            if name in {"kv_cache_usage_perc", "gpu_cache_usage_perc"}:
                values[name] = max(values[name], value)
            else:
                values[name] += value
    return {"values": dict(values), "cache_config": configs}


def delta(before, after, key):
    if key not in before or key not in after or after[key] < before[key]:
        return None
    return after[key] - before[key]


def describe(values):
    xs = sorted(x for x in values if x is not None and math.isfinite(x))
    if not xs:
        return {"n": 0, "mean": None, "p50": None, "p95": None, "p99": None, "max": None}
    def quantile(q):
        pos = (len(xs) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)
    return {"n": len(xs), "mean": statistics.fmean(xs), "p50": quantile(.5),
            "p95": quantile(.95), "p99": quantile(.99), "max": xs[-1]}


def summarize_telemetry(samples, server):
    usable = [s["servers"][server]["values"] for s in samples
              if "values" in s.get("servers", {}).get(server, {})]
    if not usable:
        return {"samples": 0, "available": False}
    first, last = usable[0], usable[-1]
    hist = {}
    for name in sorted(HISTOGRAMS):
        count = delta(first, last, name + "_count")
        total = delta(first, last, name + "_sum")
        hist[name] = {"count": count, "sum": total,
                      "mean": total / count if count and total is not None else None}
    hits = delta(first, last, "prefix_cache_hits_total")
    queries = delta(first, last, "prefix_cache_queries_total")
    return {"available": True, "samples": len(usable),
            "scope": "whole serving instance during trial; other traffic may contribute",
            "gauges": {k: describe([v.get(k) for v in usable]) for k in sorted(GAUGES)},
            "counters": {k: delta(first, last, k) for k in sorted(COUNTERS)},
            "prefix_hit_ratio": hits / queries if queries and hits is not None else None,
            "histograms": hist,
            "counter_reset_detected": any(k in first and k in last and last[k] < first[k]
                                           for k in ALLOWED - GAUGES)}

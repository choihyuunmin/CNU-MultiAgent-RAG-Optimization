#!/usr/bin/env python3
"""Summarize real scaling runs without mistaking output agreement for accuracy.

Reuse the application's unchanged law-agreement and text-similarity functions.
No LLM judge, generated labels, response filtering, or missing-result imputation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path


def rows(path):
    return [json.loads(s) for s in path.read_text().splitlines() if s.strip()]


def mean(values):
    values = list(values)
    return statistics.mean(values) if values else None


def percentile(values, p):
    values = sorted(values)
    if not values:
        return None
    offset = (len(values) - 1) * p
    lo, hi = math.floor(offset), math.ceil(offset)
    return values[lo] + (values[hi] - values[lo]) * (offset - lo)


def interval_union_ms(intervals):
    """Parallel call durations must not be summed as elapsed request latency."""
    total, end = 0.0, -math.inf
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("negative interval")
        total += max(0.0, stop - max(start, end))
        end = max(end, stop)
    return total


def counter_delta(samples, key):
    values = [s[key] for s in samples if key in s]
    if len(values) != len(samples) or len(values) < 2 or not all(math.isfinite(v) for v in values):
        return None
    if any(b < a for a, b in zip(values, values[1:])):
        return None  # Restart/reset; do not silently turn a negative delta into zero.
    return values[-1] - values[0]


def receiver_summary(path):
    grouped, errors = defaultdict(list), defaultdict(int)
    for row in rows(path):
        if "metrics" in row:
            grouped[row["resource"]].append(row["metrics"])
        else:
            errors[row["resource"]] += 1
    result = {}
    for resource in sorted(set(grouped) | set(errors)):
        samples = grouped[resource]
        item = {"samples": len(samples), "errors": errors[resource]}
        item["initial_running"] = samples[0].get("vllm:num_requests_running") if samples else None
        item["initial_waiting"] = samples[0].get("vllm:num_requests_waiting") if samples else None
        item["final_running"] = samples[-1].get("vllm:num_requests_running") if samples else None
        item["final_waiting"] = samples[-1].get("vllm:num_requests_waiting") if samples else None
        for key in ("num_requests_running", "num_requests_waiting", "kv_cache_usage_perc"):
            values = [s["vllm:" + key] for s in samples if "vllm:" + key in s]
            item[key + "_max"] = max(values) if values else None
        item["preemptions_delta"] = counter_delta(samples, "vllm:num_preemptions_total")
        for phase in ("queue", "inference", "prefill", "decode"):
            prefix = f"vllm:request_{phase}_time_seconds"
            count, seconds = (counter_delta(samples, prefix + suffix) for suffix in ("_count", "_sum"))
            item[phase + "_count_delta"] = count
            item[phase + "_mean_seconds"] = seconds / count if count and seconds is not None else None
        result[resource] = item
    return result


def parse_tagged_records(text):
    """Decode intact JSON even when another logger appends text on the same line."""
    records = {name: [] for name in ("RAG_TRACE_V1", "CNU_INGRESS_V1", "CNU_COMPLETION_V1", "CNU_HARNESS_V1", "CNU_APP_ADAPTER_V1")}
    pattern = re.compile(r"(?<!\w)(" + "|".join(records) + r")\s+")
    decoder, malformed = json.JSONDecoder(), 0
    for line in text.splitlines():
        consumed = 0
        for match in pattern.finditer(line):
            if match.start() < consumed:
                continue  # A marker embedded inside an already-decoded JSON string.
            try:
                value, end = decoder.raw_decode(line, match.end())
                if not isinstance(value, dict):
                    raise ValueError("expected object")
                records[match.group(1)].append(value)
                consumed = end
            except (ValueError, TypeError):
                malformed += 1  # Never repair truncated JSON or invent missing fields.
    return records, malformed


def trace_summary(path):
    records, malformed = parse_tagged_records(path.read_text(errors="replace"))
    traces, ingress, completion, harness = (records[name] for name in
        ("RAG_TRACE_V1", "CNU_INGRESS_V1", "CNU_COMPLETION_V1", "CNU_HARNESS_V1"))
    call_groups = defaultdict(list)
    union_times, generate_wait, stage_wait, post_generate = [], [], [], []
    for trace in traces:
        calls = trace.get("llm_calls", [])
        union_times.append(interval_union_ms((c["start_offset_ms"], c["end_offset_ms"])
                                            for c in calls if "start_offset_ms" in c and "end_offset_ms" in c))
        for call in calls:
            call_groups[call.get("model", "unknown")].append(call)
        spans = trace.get("spans", [])
        finished_generate = [s["end_offset_ms"] for s in spans if s["kind"] == "service" and s["name"] == "generate" and "end_offset_ms" in s]
        if finished_generate:
            post_generate.append(max(0, trace["duration_ms"] - max(finished_generate)))
        generate_wait.append(sum(s["duration_ms"] for s in spans if s["kind"] == "queue_wait"
                                 and s["name"].startswith("generate-")))
        stage_wait.append(sum(s["duration_ms"] for s in spans if s["kind"] == "queue_wait"
                              and s["name"] == "stage_credit"))
    return {
        "trace_count": len(traces), "ingress_count": len(ingress), "malformed_records": malformed,
        "outer_wait_mean_ms": mean(r["outer_wait_ms"] for r in ingress),
        "outer_wait_p95_ms": percentile([r["outer_wait_ms"] for r in ingress], .95),
        "generate_wait_mean_ms": mean(generate_wait),
        "stage_wait_work_mean_ms": mean(stage_wait),
        "post_generate_tail_mean_ms": mean(post_generate),
        "llm_union_elapsed_mean_ms": mean(union_times),
        "llm_calls_per_request": sum(len(c) for c in call_groups.values()) / len(traces) if traces else None,
        "calls_by_model": {model: {"calls": len(calls), "api_duration_mean_ms": mean(c["duration_ms"] for c in calls),
                                   "failed_calls": sum(c.get("status") != "ok" for c in calls),
                                   "completion_tokens_mean": mean(c["completion_tokens"] for c in calls if c.get("completion_tokens") is not None)}
                           for model, calls in call_groups.items()},
        "harness": {"upstreams": sum(e.get("event") == "upstream_closed" for e in harness),
                    "buffer_peak_chunks": max((e.get("buffer_peak_chunks", 0) for e in harness), default=None),
                    "total_buffer_blocked_ms": sum(e.get("buffer_blocked_ms", 0) for e in harness),
                    "incomplete_relays": sum(e.get("event") == "relay_closed" and not e.get("fully_consumed") for e in harness)},
        "notes": "LLM interval union is observed occupancy, not proof of critical path. Stage-wait work can overlap. API duration includes network and engine queue.",
    }


def indexed(records):
    result = {}
    for row in records:
        key = row["question_id"]
        if key in result:
            raise ValueError("duplicate question ID; analyze each repeat separately")
        result[key] = row
    return result


def valid_response(row):
    response = row.get("response")
    return row.get("status") == "ok" and isinstance(response, dict) and isinstance(response.get("laws"), list) and isinstance(response.get("comment"), str)


def paired_latency(control, candidate):
    """Descriptive paired differences only; concurrent requests are not IID."""
    left, right = indexed(control), indexed(candidate)
    if set(left) != set(right):
        raise ValueError("unaligned question sets")
    differences = [right[k]["duration_ms"] - left[k]["duration_ms"] for k in sorted(left)]
    return {"questions": len(differences), "includes_failed_attempts": True,
            "candidate_minus_reference_mean_ms": mean(differences),
            "candidate_minus_reference_median_ms": percentile(differences, .5),
            "candidate_faster_questions": sum(d < 0 for d in differences),
            "candidate_slower_questions": sum(d > 0 for d in differences),
            "equal_duration_questions": sum(d == 0 for d in differences),
            "note": "Descriptive only: shared queues couple observations; one fixed-order run cannot establish causal significance."}


def agreement(control, candidate, evaluation):
    left, right = indexed(control), indexed(candidate)
    if set(left) != set(right):
        raise ValueError("unaligned question sets")
    pairs = [(left[k], right[k]) for k in sorted(left) if valid_response(left[k]) and valid_response(right[k])]
    scores = [evaluation.law_metrics(a.get("law_ids", []), b.get("law_ids", [])) for a, b in pairs]
    comments = [evaluation.sparse_cosine(evaluation.char_ngram_vector(evaluation.response_comment(a)),
                                        evaluation.char_ngram_vector(evaluation.response_comment(b))) for a, b in pairs]
    return {"total_questions": len(left), "successful_pairs": len(pairs),
            "control_failures": sum(r["status"] != "ok" for r in left.values()),
            "candidate_failures": sum(r["status"] != "ok" for r in right.values()),
            "control_missing_or_invalid_responses": sum(not valid_response(r) for r in left.values()),
            "candidate_missing_or_invalid_responses": sum(not valid_response(r) for r in right.values()),
            "empty_law_pairs": sum(not a.get("law_ids") and not b.get("law_ids") for a, b in pairs),
            "law_recall_mean": mean(s["law_recall"] for s in scores),
            "law_ndcg10_mean": mean(s["law_ndcg10"] for s in scores),
            "top1_agreement_mean": mean(s["top1_agreement"] for s in scores),
            "ordered_law_ids_exact_rate": mean(a.get("law_ids", []) == b.get("law_ids", []) for a, b in pairs),
            "comment_exact_rate": mean(evaluation.response_comment(a) == evaluation.response_comment(b) for a, b in pairs),
            "comment_trigram_cosine_mean": mean(comments),
            "expert_accuracy": None,
            "verdict": "inconclusive: pseudo-gold agreement only; no expert labels or accepted non-inferiority margin"}


def summarize_cell(folder, cell):
    result_path = folder / "responses" / folder.name / cell
    records = rows(result_path / "client_requests.jsonl")
    indexed(records)
    original = json.loads((result_path / "summary.json").read_text())["cells"][0]
    trace = trace_summary(folder / cell / "server.log")
    receivers = receiver_summary(folder / cell / "receiver_metrics.local.jsonl")
    topology = json.loads((folder / "topology.local.json").read_text())
    expected = defaultdict(int)
    for alias, calls in trace["calls_by_model"].items():
        expected[topology["aliases"].get(alias, "unmapped")] += calls["calls"] - calls["failed_calls"]
    reconciliation = {resource: {"app_successful_calls": expected.get(resource, 0),
                                  "engine_completed_calls": values["inference_count_delta"],
                                  "counts_match": values["inference_count_delta"] == expected.get(resource, 0)}
                      for resource, values in receivers.items()}
    return {"cell": cell, "run_id": folder.name, "latency": original,
            "valid_response_count": sum(valid_response(r) for r in records),
            "all_attempt_latency_mean_ms": mean(r["duration_ms"] for r in records),
            "all_attempt_latency_p95_ms": percentile([r["duration_ms"] for r in records], .95),
            "client_error_types": dict(Counter(r.get("error_type", "unknown") for r in records if r["status"] != "ok")),
            "trace": trace, "receivers": receivers,
            "resource_call_reconciliation": reconciliation,
            "reconciliation_note": "A mismatch can reflect unfinished/cancelled own calls, trace or measurement boundaries, or other traffic; it does not by itself establish external contamination."}, records


def markdown(report):
    def fmt(value, divisor=1, digits=2):
        return "미측정" if value is None else f"{value / divisor:.{digits}f}"
    names = {"control": "기존 방식", "request_window": "요청 창 확대", "stage_fifo": "단계별 슬롯·도착순", "stage_coflow": "단계별 슬롯·질문 묶음 순서",
             "delivery": "화면 전달 대기 제거", "delivery_coflow": "전달 대기 제거 + 단계별 대기 관리",
             "delivery_repeat": "같은 기준 검색기 재측정", "delivery_fifo": "화면 대기 제거 + 자원별 도착순", "network_harness": "네트워크 하네스: 순환·크레딧·버퍼"}
    lines = ["# 사용자 규모별 실제 RAG 실험", "", f"방법별 동일 {report['questions_per_cell']}문항. 완료된 조건 {len(report['cells'])}개. 모델·프롬프트·vLLM 설정은 변경하지 않았다.", "",
             "| 동시 사용자 | 방법 | 완료/시도 | 평균 응답(초) | p95(초) | 처리량(건/초) | 입구 대기(초) | 기존 법령 재현율 | 첫 법령 일치율 |",
             "|---:|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cell in report["cells"]:
        latency, trace, quality = cell["latency"], cell["trace"], cell.get("agreement", {})
        mode = cell["cell"].split("-", 1)[1]
        lines.append(f"| {latency['concurrency']} | {names[mode]} | {latency['successes']}/{latency['requests']} | "
                     f"{fmt(latency['latency_mean_ms'], 1000)} | {fmt(latency['latency_p95_ms'], 1000)} | "
                     f"{fmt(latency['throughput_rps'], digits=3)} | {fmt(trace['outer_wait_mean_ms'], 1000)} | "
                     f"{fmt(quality.get('law_recall_mean'), .01, 1)} | {fmt(quality.get('top1_agreement_mean'), .01, 1)} |")
    lines += ["", "응답 평균/p95는 원래 평가기의 성공 요청 기준이며 실패 수를 함께 표시했다. 처리량과 성공률도 함께 판단해야 한다.",
              "법령 재현율·첫 법령 일치율 단위는 %. 같은 사용자 수의 control을 기준으로 하고, 없으면 화면 대기를 제거한 delivery를 기준으로 한다. 법적 정확도가 아니다.",
              "모델 출력의 비결정성, 공유 장치의 다른 요청, 냉간 시작, 한 번의 실행에 따른 변동을 통제한 최종 검증은 별도로 필요하다.", "",
              "## 대기열 및 KV 캐시", "", "| 사용자/방법 | 추론 자원 | 실행 최대 | 대기 최대 | KV 최대(%) | 중단·재개 증가 | 평균 엔진 대기(초) | 평균 생성(초) |",
              "|---|---|---:|---:|---:|---:|---:|---:|"]
    for cell in report["cells"]:
        for resource, metric in cell["receivers"].items():
            lines.append(f"| {cell['cell']} | {resource} | {fmt(metric['num_requests_running_max'], digits=0)} | "
                         f"{fmt(metric['num_requests_waiting_max'], digits=0)} | {fmt(metric['kv_cache_usage_perc_max'], .01)} | "
                         f"{fmt(metric['preemptions_delta'], digits=0)} | {fmt(metric['queue_mean_seconds'])} | {fmt(metric['decode_mean_seconds'])} |")
    lines += ["", "KV 캐시는 예약된 캐시 영역의 사용률이다. 물리 VRAM 사용률이나 메모리 대역폭 사용률이 아니다.",
              "엔진 지표는 공유 인스턴스 전체 값이다. 카운터 재시작·누락은 미측정으로 표시한다. 원시 로그와 JSON에 세부 진단을 보존했다.",
              "", "판정: **inconclusive**. 속도와 답변 보존을 모두 검증하기 전에는 개선 완료나 정확도 유지를 주장하지 않는다.", ""]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("runs", nargs="+", type=Path)
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("original_fidelity", args.application_root / "scripts/experiment/evaluate_baseline_fidelity.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    results, raw, protocols = [], {}, []
    for folder in args.runs:
        protocol = json.loads((folder / "protocol.json").read_text())
        protocols.append(protocol)
        for concurrency in protocol["concurrency"]:
            for mode in protocol["modes"]:
                cell = f"c{concurrency:02}-{mode}"
                if not (folder / "responses" / folder.name / cell / "summary.json").exists():
                    continue  # Explicit missing-cell inventory below; never invent a result.
                if cell in raw:
                    raise ValueError("duplicate cells across runs")
                summary, records = summarize_cell(folder, cell)
                results.append(summary)
                raw[cell] = records
    hashes = {p["question_sha256"] for p in protocols}
    if len(hashes) != 1:
        raise ValueError("different question sets; analyze separately")
    for result in results:
        prefix, mode = result["cell"].split("-", 1)
        baseline = "control" if prefix + "-control" in raw else "delivery"
        result["reference_mode"] = baseline
        if mode != baseline and prefix + "-" + baseline in raw:
            result["agreement"] = agreement(raw[prefix + "-" + baseline], raw[result["cell"]], evaluation)
            result["paired_latency"] = paired_latency(raw[prefix + "-" + baseline], raw[result["cell"]])
        if mode not in (baseline, "delivery_repeat") and prefix + "-delivery_repeat" in raw:
            result["agreement_vs_baseline_repeat"] = agreement(raw[prefix + "-delivery_repeat"], raw[result["cell"]], evaluation)
            result["paired_latency_vs_baseline_repeat"] = paired_latency(raw[prefix + "-delivery_repeat"], raw[result["cell"]])
        if mode == "network_harness" and prefix + "-delivery_fifo" in raw:
            result["agreement_vs_fifo"] = agreement(raw[prefix + "-delivery_fifo"], raw[result["cell"]], evaluation)
            result["paired_latency_vs_fifo"] = paired_latency(raw[prefix + "-delivery_fifo"], raw[result["cell"]])
        if mode == "delivery_coflow" and prefix + "-delivery" in raw:
            result["agreement_vs_delivery_only"] = agreement(raw[prefix + "-delivery"], raw[result["cell"]], evaluation)
    expected = {f"c{c:02}-{m}" for p in protocols for c in p["concurrency"] for m in p["modes"]}
    report = {"schema_version": 1, "questions_per_cell": protocols[0]["questions_per_cell"],
              "missing_cells": sorted(expected - set(raw)), "cells": sorted(results, key=lambda r: r["cell"]),
              "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "evaluation_source_sha256": hashlib.sha256(Path(spec.origin).read_bytes()).hexdigest(),
              "expert_accuracy": None, "gpu_device_metrics": "unavailable", "overall_verdict": "inconclusive"}
    args.output_dir.mkdir(parents=True, exist_ok=False)
    (args.output_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output_dir / "REPORT.md").write_text(markdown(report))
    print(markdown(report))


if __name__ == "__main__":
    main()

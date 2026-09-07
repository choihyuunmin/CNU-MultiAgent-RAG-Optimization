#!/usr/bin/env python3
"""Analyze identical-image application trials; agreement is not legal accuracy."""

import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from analyze_scaling_experiment import agreement, indexed, mean, paired_latency, parse_tagged_records, percentile, receiver_summary, rows, valid_response


def timestamp(value):
    return datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp()


def trace_metrics(path, requests):
    if not path.exists():
        return {"available": False}
    tagged, malformed = parse_tagged_records(path.read_text(errors="replace"))
    cases = {r["case_id"] for r in requests}
    events = [r for r in tagged["CNU_APP_ADAPTER_V1"] if r.get("case_id") in cases]
    finished = [e for e in events if e.get("event") == "request_finished"]
    counts = Counter(e["case_id"] for e in finished)
    calls = [e for e in events if e.get("event") == "llm_finished"]
    roots = {e["root_id"] for e in finished} | {e["root_id"] for e in calls}
    relays = [e for e in tagged["CNU_HARNESS_V1"] if e.get("root_id") in roots]
    ingress = [e for e in tagged["CNU_INGRESS_V1"] if e.get("case_id") in cases]
    groups = defaultdict(list)
    for call in calls:
        groups[(call["model"], call.get("role", "-"), call["streaming"])].append(call)
    per_model = []
    for (model, role, streaming), group in sorted(groups.items()):
        api_times = [c["api_ms"] for c in group if c.get("api_ms") is not None]
        per_model.append({"model": model, "role": role, "streaming": streaming,
            "calls": len(group), "successful_calls": sum(c["status"] == "ok" for c in group),
            "admission_wait_mean_ms": mean(c["wait_ms"] for c in group),
            "api_mean_ms": mean(api_times), "api_p95_ms": percentile(api_times, .95),
            "completion_tokens_observed_calls": sum("completion_tokens" in c for c in group),
            "completion_tokens_mean": mean(c["completion_tokens"] for c in group if "completion_tokens" in c),
            "stream_chunks_mean": mean(c["chunks"] for c in group) if streaming else None})
    return {"available": True, "malformed_records_in_log": malformed,
            "request_traces": len(finished), "missing_request_traces": sorted(cases - set(counts)),
            "duplicate_case_traces": sorted(k for k, v in counts.items() if v != 1),
            "llm_calls": len(calls), "managed_calls": sum(e["managed"] for e in calls),
            "internal_call_failures": dict(Counter(e.get("error_type", e["status"]) for e in calls if e["status"] != "ok")),
            "outer_wait_mean_ms": mean(e["outer_wait_ms"] for e in ingress),
            "outer_wait_p95_ms": percentile([e["outer_wait_ms"] for e in ingress], .95),
            "per_call_admission_wait_mean_ms": mean(e["wait_ms"] for e in calls),
            "per_call_api_mean_ms": mean(e["api_ms"] for e in calls if e.get("api_ms") is not None),
            "calls_by_model_role": per_model,
            "relay_upstreams": sum(e["event"] == "upstream_closed" for e in relays),
            "relay_incomplete": sum(e["event"] == "relay_closed" and not e["fully_consumed"] for e in relays),
            "relay_peak_chunks": max((e.get("buffer_peak_chunks", 0) for e in relays), default=None),
            "relay_blocked_ms": sum(e.get("buffer_blocked_ms", 0) for e in relays),
            "note": "Per-call work can overlap. API duration includes SDK/network/engine/stream-consumer time; chunks are not tokens. Do not sum call durations as end-to-end elapsed time."}


def gpu_metrics(samples, requests):
    start = min(timestamp(r["started_at"]) for r in requests)
    end = max(timestamp(r["completed_at"]) for r in requests)
    selected = [s for s in samples if start <= timestamp(s["timestamp"]) <= end]
    result = {}
    for index in sorted({s["index"] for s in selected}):
        device = [s for s in selected if s["index"] == index]
        item = {"samples": len(device), "name": device[0]["name"]}
        for field in ("gpu_utilization_percent", "memory_used_mib", "memory_total_mib", "memory_utilization_percent", "power_draw_w"):
            values = []
            for sample in device:
                try:
                    values.append(float(sample[field]))
                except (TypeError, ValueError):
                    pass
            item[field + "_mean"] = mean(values)
            item[field + "_max"] = max(values) if values else None
        result[index] = item
    return result


def quality_audit(control, candidate, evaluation):
    """Expose reference-law omissions without changing the original evaluator."""
    left, right = indexed(control), indexed(candidate)
    if set(left) != set(right):
        raise ValueError("unaligned question sets")
    nonempty_scores, omissions, changed_first = [], [], []
    for key in sorted(left):
        a, b = left[key], right[key]
        if not valid_response(a) or not valid_response(b):
            continue
        reference, result = a.get("law_ids", []), b.get("law_ids", [])
        if reference:
            nonempty_scores.append(evaluation.law_metrics(reference, result)["law_recall"])
        missing = sorted(set(reference) - set(result))
        if missing:
            omissions.append({"question_id": key, "missing_reference_law_ids": missing,
                              "reference_law_ids": reference, "candidate_law_ids": result})
        if reference[:1] != result[:1]:
            changed_first.append(key)
    return {"successful_nonempty_reference_pairs": len(nonempty_scores),
            "law_recall_nonempty_reference_mean": mean(nonempty_scores),
            "reference_law_omissions": omissions, "top1_changed_question_ids": changed_first,
            "note": "Supplementary only. The main 100-question metric is unchanged. Law IDs do not establish clause coverage or correctness."}


def request_examples(records, logs):
    """Select cases using the reference only, never cherry-pick candidate gains."""
    original = sorted(records.get("original", []), key=lambda r: r["duration_ms"])
    if not original:
        return []
    chosen = [("middle_reference_duration", original[len(original) // 2]["question_id"]),
              ("slowest_reference", original[-1]["question_id"])]
    examples = [{"selection": reason, "question_id": key, "conditions": {}} for reason, key in chosen]
    for mode, data in records.items():
        indexed_data = indexed(data)
        path = logs / (mode + ".log")
        tagged, _ = parse_tagged_records(path.read_text(errors="replace")) if path.exists() else ({}, 0)
        for example in examples:
            row = indexed_data[example["question_id"]]
            case = row["case_id"]
            ingress = [e for e in tagged.get("CNU_INGRESS_V1", []) if e.get("case_id") == case]
            calls = [e for e in tagged.get("CNU_APP_ADAPTER_V1", [])
                     if e.get("event") == "llm_finished" and e.get("case_id") == case]
            item = {"case_id": case, "client_duration_ms": row["duration_ms"],
                    "valid_response": valid_response(row), "law_ids": row.get("law_ids", []),
                    "outer_wait_ms": ingress[0]["outer_wait_ms"] if len(ingress) == 1 else None,
                    "observed_llm_calls": len(calls)}
            if calls:
                longest = max(calls, key=lambda c: c["total_ms"])
                item["longest_observed_llm_call"] = {key: longest.get(key) for key in
                    ("model", "role", "streaming", "status", "wait_ms", "api_ms", "total_ms", "completion_tokens")}
            example["conditions"][mode] = item
    return examples


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--application-root", type=Path, required=True)
    parser.add_argument("--server-logs", type=Path, required=True)
    parser.add_argument("--gpu-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("original_evaluation", args.application_root / "scripts/experiment/evaluate_baseline_fidelity.py")
    evaluation = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluation)
    protocol = json.loads((args.run / "protocol.json").read_text())
    gpu = rows(args.gpu_log) if args.gpu_log.exists() else []
    results, records, missing = [], {}, []
    for mode in protocol["endpoints"]:
        cell = f"c{protocol['concurrency']:02}-{mode}"
        folder = args.run / "responses" / args.run.name / cell
        if not (folder / "summary.json").exists():
            missing.append(mode)
            continue
        data = rows(folder / "client_requests.jsonl")
        indexed(data)
        if len(data) != protocol["questions"]:
            raise ValueError("incomplete question count")
        records[mode] = data
        results.append({"mode": mode, "requests": len(data), "valid_responses": sum(valid_response(r) for r in data),
            "all_attempt_mean_ms": mean(r["duration_ms"] for r in data),
            "all_attempt_p95_ms": percentile([r["duration_ms"] for r in data], .95),
            "original_evaluator": json.loads((folder / "summary.json").read_text())["cells"][0],
            "trace": trace_metrics(args.server_logs / (mode + ".log"), data),
            "receivers": receiver_summary(args.run / cell / "receiver_metrics.local.jsonl"),
            "gpu": gpu_metrics(gpu, data)})
    for result in results:
        mode = result["mode"]
        if mode != "original" and "original" in records:
            result["agreement"] = agreement(records["original"], records[mode], evaluation)
            result["quality_audit"] = quality_audit(records["original"], records[mode], evaluation)
            result["paired_latency"] = paired_latency(records["original"], records[mode])
        if mode == "network" and "delivery" in records:
            result["agreement_vs_delivery"] = agreement(records["delivery"], records[mode], evaluation)
            result["paired_latency_vs_delivery"] = paired_latency(records["delivery"], records[mode])
    report = {"protocol": protocol, "results": results, "missing_modes": missing,
              "request_examples": request_examples(records, args.server_logs),
              "expert_accuracy": None, "verdict": "inconclusive",
              "analysis_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "notes": ["Same-image isolated trial, not a production traffic switch.",
                        "One sequential pass, not a causal significance or accuracy non-inferiority test.",
                        "GPU metrics cover the whole shared device, not only these requests."]}
    names = {"original": "운영 이미지 기준", "delivery": "화면 전달 대기 제거", "network": "네트워크 하네스 추가", "original_repeat": "기준 재측정"}
    text = ["# 운영 이미지에 연결한 어댑터 실험", "", f"동일 {protocol['questions']}문항, 동시 사용자 {protocol['concurrency']}명. 모델·엔진·질문 실행 한도 유지.", "",
            "| 방법 | 유효 응답 | 전체 평균(초) | 전체 p95(초) | 기준 법령 재현율 | 첫 법령 일치율 |",
            "|---|---:|---:|---:|---:|---:|"]
    for r in results:
        q = r.get("agreement")
        score = f"{q['law_recall_mean']:.1%} | {q['top1_agreement_mean']:.1%}" if q and q["law_recall_mean"] is not None else "— | —"
        text.append(f"| {names[r['mode']]} | {r['valid_responses']}/{r['requests']} | {r['all_attempt_mean_ms']/1000:.2f} | {r['all_attempt_p95_ms']/1000:.2f} | {score} |")
    text += ["", "실패 포함 시간이다. 법령 재현율은 기준 답변과의 일치도이며 법적 정확도가 아니다.",
             f"미완료 조건: {', '.join(missing) or '없음'}. 상세 원인·GPU·호출 로그는 summary.json에 보존했다.",
             "운영 트래픽은 전환하지 않았다. 정확도 유지와 방법만의 인과 효과는 미입증이다.", ""]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    (args.output / "REPORT.md").write_text("\n".join(text))
    print("\n".join(text))


if __name__ == "__main__":
    main()

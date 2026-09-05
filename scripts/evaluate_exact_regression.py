#!/usr/bin/env python3
"""Fail closed when candidate outputs regress against aligned pseudo-gold JSONL."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from cnu_rag_optimization import compare_regression_records


def _load(path: Path, key: str) -> dict[str, dict]:
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as stream:
        for line_number, raw in enumerate(stream, 1):
            if not raw.strip():
                continue
            row = json.loads(raw)
            value = str(row.get(key, "")).strip()
            if not value or value in rows:
                raise ValueError(f"invalid or duplicate {key} at {path}:{line_number}")
            rows[value] = row
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("control", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--key-field", default="question_id")
    parser.add_argument("--id-field", default="selected_ids")
    parser.add_argument("--response-field", default="response")
    parser.add_argument("--min-success", type=float, default=1.0)
    parser.add_argument("--min-exact-response", type=float, default=1.0)
    parser.add_argument("--min-id-recall", type=float, default=1.0)
    parser.add_argument("--min-top1", type=float, default=1.0)
    args = parser.parse_args()

    control = _load(args.control, args.key_field)
    candidate = _load(args.candidate, args.key_field)
    if set(control) != set(candidate):
        missing = sorted(set(control) - set(candidate))
        extra = sorted(set(candidate) - set(control))
        raise ValueError(f"unaligned records: missing={missing[:3]} extra={extra[:3]}")
    keys = sorted(control)
    metrics = compare_regression_records(
        [control[key] for key in keys],
        [candidate[key] for key in keys],
        id_field=args.id_field,
        response_field=args.response_field,
    )
    payload = asdict(metrics)
    thresholds = {
        "success_rate": args.min_success,
        "exact_response_rate": args.min_exact_response,
        "id_recall": args.min_id_recall,
        "top1_agreement": args.min_top1,
    }
    payload["thresholds"] = thresholds
    payload["passed"] = all(
        getattr(metrics, name) >= threshold for name, threshold in thresholds.items()
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

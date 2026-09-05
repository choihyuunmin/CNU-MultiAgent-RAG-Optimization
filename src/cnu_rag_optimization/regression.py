"""Domain-neutral pseudo-gold regression metrics."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Mapping, Sequence


def _canonical(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _ids(row: Mapping[str, object], field: str) -> tuple[str, ...]:
    value = row.get(field, [])
    if not isinstance(value, list):
        return ()
    return tuple(str(item).strip() for item in value if str(item).strip())


@dataclass(frozen=True)
class RegressionMetrics:
    count: int
    success_rate: float
    exact_response_rate: float
    id_recall: float
    top1_agreement: float


def compare_regression_records(
    controls: Sequence[Mapping[str, object]],
    candidates: Sequence[Mapping[str, object]],
    *,
    id_field: str = "selected_ids",
    response_field: str = "response",
    status_field: str = "status",
) -> RegressionMetrics:
    """Compare aligned records. Inputs remain unchanged."""
    if len(controls) != len(candidates) or not controls:
        raise ValueError("control and candidate must have same positive length")

    successes = 0
    exact = 0
    recalls: list[float] = []
    top1 = 0
    for control, candidate in zip(controls, candidates, strict=True):
        successes += candidate.get(status_field) == "ok"
        exact += _canonical(control.get(response_field)) == _canonical(
            candidate.get(response_field)
        )
        expected = _ids(control, id_field)
        observed = _ids(candidate, id_field)
        if expected:
            recalls.append(len(set(expected) & set(observed)) / len(set(expected)))
            top1 += bool(observed) and observed[0] == expected[0]
        else:
            recalls.append(1.0 if not observed else 0.0)
            top1 += not observed

    count = len(controls)
    return RegressionMetrics(
        count=count,
        success_rate=successes / count,
        exact_response_rate=exact / count,
        id_recall=sum(recalls) / count,
        top1_agreement=top1 / count,
    )

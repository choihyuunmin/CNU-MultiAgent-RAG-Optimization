"""Joint acceptance gate: end-to-end gain AND independently evaluated quality.

Inputs are measured, paired confidence bounds, not local boundary timings.
This gate does not estimate intervals or replace expert review.
"""
from dataclasses import dataclass
import math

from .quality_gate import QualityEvidence, QualityThresholds, evaluate_quality_gate


@dataclass(frozen=True)
class PerformanceThresholds:
    loads: tuple[int, ...] = (1, 2, 5, 10, 20, 50, 100)
    primary_load: int = 100
    minimum_questions: int = 200
    minimum_rounds: int = 4
    minimum_primary_gain: float = 0.05
    maximum_secondary_regression: float = 0.05
    maximum_p95_ratio: float = 1.05

    def __post_init__(self):
        if (not self.loads or len(set(self.loads)) != len(self.loads)
                or any(type(c) is not int or c < 1 for c in self.loads)
                or self.primary_load not in self.loads):
            raise ValueError('loads must be unique positive integers and include primary_load')
        if any(type(n) is not int or n < 1 for n in (self.minimum_questions, self.minimum_rounds)):
            raise ValueError('positive sample requirements needed')
        for x in (self.minimum_primary_gain, self.maximum_secondary_regression):
            if not math.isfinite(x) or not 0 <= x < 1:
                raise ValueError('gain/regression thresholds must be fractions in [0, 1)')
        if not math.isfinite(self.maximum_p95_ratio) or self.maximum_p95_ratio < 1:
            raise ValueError('maximum_p95_ratio must be finite and at least one')


@dataclass(frozen=True)
class LoadEvidence:
    concurrency: int
    unique_questions: int
    rounds: int
    paired_complete: bool
    original_failures: int
    candidate_failures: int
    # Lower bound of (original mean - candidate mean) / original mean.
    mean_gain_lower: float | None = None
    # Upper bound of candidate p95 / original p95.
    p95_ratio_upper: float | None = None
    scope: str = 'end_to_end'
    question_and_round_resampling: bool = False


def evaluate_release_gate(loads: list[LoadEvidence], qualities: dict[int, QualityEvidence], *,
                          thresholds=PerformanceThresholds(),
                          quality_thresholds=QualityThresholds(minimum_questions=200),
                          thresholds_predeclared=False, execution_equivalence_verified=False):
    reasons = []
    if thresholds_predeclared is not True:
        reasons.append('thresholds_not_predeclared')
    if execution_equivalence_verified is not True:
        reasons.append('execution_equivalence_not_verified')
    if len(loads) != len(thresholds.loads) or {e.concurrency for e in loads} != set(thresholds.loads):
        reasons.append('missing_or_duplicate_loads')
    quality_results = {}
    for e in loads:
        prefix = f'c{e.concurrency}:'
        if e.scope != 'end_to_end':
            reasons.append(prefix + 'not_end_to_end')
        if (type(e.unique_questions) is not int or e.unique_questions < thresholds.minimum_questions
                or type(e.rounds) is not int or e.rounds < thresholds.minimum_rounds):
            reasons.append(prefix + 'insufficient_samples')
        if e.paired_complete is not True or e.question_and_round_resampling is not True:
            reasons.append(prefix + 'pairing_or_resampling_missing')
        if (type(e.original_failures) is not int or type(e.candidate_failures) is not int
                or e.original_failures != 0 or e.candidate_failures != 0):
            reasons.append(prefix + 'failed_requests_require_review')
        target = (thresholds.minimum_primary_gain if e.concurrency == thresholds.primary_load
                  else -thresholds.maximum_secondary_regression)
        if (e.mean_gain_lower is None or not math.isfinite(e.mean_gain_lower)
                or e.mean_gain_lower > 1 or e.mean_gain_lower < target):
            reasons.append(prefix + 'mean_latency_target_not_established')
        if (e.p95_ratio_upper is None or not math.isfinite(e.p95_ratio_upper)
                or not 0 < e.p95_ratio_upper <= thresholds.maximum_p95_ratio):
            reasons.append(prefix + 'tail_latency_target_not_established')
        quality = qualities.get(e.concurrency)
        if quality is None:
            reasons.append(prefix + 'quality_evidence_missing')
        else:
            quality_results[e.concurrency] = evaluate_quality_gate(quality, quality_thresholds)
            if not quality_results[e.concurrency]['release_allowed']:
                reasons.append(prefix + 'quality_not_established')
    return {'release_allowed': not reasons, 'reasons': reasons, 'quality': quality_results,
            'scope': 'predeclared workload only; no universal speed or correctness guarantee'}

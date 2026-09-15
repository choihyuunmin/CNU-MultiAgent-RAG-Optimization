"""Fail-closed release gate for predeclared, paired quality evaluation.

This consumes evaluation evidence; it does not fabricate an answer judge or
equate citation inclusion, identical inputs, or HTTP 200 with answer correctness.
"""
from dataclasses import dataclass
import math


@dataclass(frozen=True)
class QualityThresholds:
    minimum_questions: int = 100
    source_noninferiority_margin: float = 0.0
    correctness_noninferiority_margin: float = 0.0
    groundedness_noninferiority_margin: float = 0.0

    def __post_init__(self):
        if type(self.minimum_questions) is not int or self.minimum_questions < 1:
            raise ValueError("minimum_questions must be positive")
        for margin in (self.source_noninferiority_margin,
                       self.correctness_noninferiority_margin,
                       self.groundedness_noninferiority_margin):
            if not math.isfinite(margin) or not 0 <= margin <= 1:
                raise ValueError("quality margins must be finite fractions")


@dataclass(frozen=True)
class QualityEvidence:
    unique_questions: int
    paired_complete: bool
    baseline_success_rate: float
    candidate_success_rate: float
    # Lower confidence bounds of paired candidate-minus-control differences.
    # Question-level repeats must be collapsed, not counted as new questions.
    source_delta_lower: float | None = None
    correctness_delta_lower: float | None = None
    groundedness_delta_lower: float | None = None
    guardrail_regressions: int | None = None
    blinded_expert_answers: bool = False
    independent_holdout: bool = False
    thresholds_predeclared: bool = False


def evaluate_quality_gate(evidence: QualityEvidence, thresholds=QualityThresholds()) -> dict:
    reasons = []
    if type(evidence.unique_questions) is not int or evidence.unique_questions < thresholds.minimum_questions:
        reasons.append("insufficient_unique_questions")
    if not evidence.paired_complete:
        reasons.append("incomplete_paired_evaluation")
    if evidence.baseline_success_rate != 1 or evidence.candidate_success_rate != 1:
        reasons.append("request_or_pipeline_failures")
    for name, lower, margin in (
        ("source", evidence.source_delta_lower, thresholds.source_noninferiority_margin),
        ("correctness", evidence.correctness_delta_lower, thresholds.correctness_noninferiority_margin),
        ("groundedness", evidence.groundedness_delta_lower, thresholds.groundedness_noninferiority_margin),
    ):
        if lower is None or not math.isfinite(lower) or not -1 <= lower <= 1:
            reasons.append(name + "_evidence_missing_or_invalid")
        elif lower < -margin:
            reasons.append(name + "_noninferiority_not_established")
    if evidence.guardrail_regressions != 0:
        reasons.append("guardrail_evidence_missing_or_regressed")
    if not evidence.blinded_expert_answers:
        reasons.append("expert_answer_evaluation_missing")
    if not evidence.independent_holdout:
        reasons.append("independent_holdout_missing")
    if not evidence.thresholds_predeclared:
        reasons.append("predeclared_thresholds_missing")
    return {"release_allowed": not reasons, "reasons": reasons,
            "scope": "quality noninferiority under the supplied evaluation; not universal correctness"}

from dataclasses import replace

from cnu_rag_optimization.performance_gate import LoadEvidence, PerformanceThresholds, evaluate_release_gate
from cnu_rag_optimization.quality_gate import QualityEvidence


def evidence():
    loads = [LoadEvidence(c, 200, 4, True, 0, 0, .06, 1.01,
                         question_and_round_resampling=True) for c in PerformanceThresholds().loads]
    quality = QualityEvidence(200, True, 1., 1., 0., 0., 0., 0, True, True, True)
    return loads, {e.concurrency: quality for e in loads}


def test_joint_gate_never_substitutes_local_gain_for_user_latency():
    loads, qualities = evidence()
    options = dict(thresholds_predeclared=True, execution_equivalence_verified=True)
    assert evaluate_release_gate(loads, qualities, **options)['release_allowed']
    for field, value in [('scope', 'boundary'), ('rounds', 2), ('candidate_failures', 1),
                         ('mean_gain_lower', .023), ('p95_ratio_upper', 1.2),
                         ('mean_gain_lower', float('nan'))]:
        changed = [*loads[:-1], replace(loads[-1], **{field: value})]
        assert not evaluate_release_gate(changed, qualities, **options)['release_allowed']
    assert not evaluate_release_gate(loads, {}, **options)['release_allowed']
    qualities[100] = replace(qualities[100], correctness_delta_lower=None)
    assert not evaluate_release_gate(loads, qualities, **options)['release_allowed']


def test_incomplete_or_undeclared_experiment_cannot_pass():
    loads, qualities = evidence()
    assert not evaluate_release_gate(loads, qualities)['release_allowed']
    assert not evaluate_release_gate(loads[:-1], qualities, thresholds_predeclared=True,
                                    execution_equivalence_verified=True)['release_allowed']

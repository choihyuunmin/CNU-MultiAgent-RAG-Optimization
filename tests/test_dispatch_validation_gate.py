from dataclasses import replace
import importlib.util
from pathlib import Path

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


def test_plan_balances_position_instance_and_uses_same_200_questions():
    path = Path(__file__).parents[1] / 'scripts/build_dispatch_validation_plan.py'
    spec = importlib.util.spec_from_file_location('plan_dispatch_validation', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    questions = [{'id': f'q{i}'} for i in range(200)]
    plan = module.build_plan(questions)
    assert plan == module.build_plan(questions)
    assert plan['planned_requests'] == 22400
    for load in PerformanceThresholds().loads:
        cells = [c for c in plan['cells'] if c['concurrency'] == load]
        for arm in {c['arm'] for c in cells}:
            rows = [c for c in cells if c['arm'] == arm]
            assert {c['position'] for c in rows} == {1, 2, 3, 4}
            assert {c['physical_slot'] for c in rows} == {0, 1, 2, 3}
        for block in range(1, 5):
            batch = [c for c in cells if c['block'] == block]
            assert all(c['question_ids'] == batch[0]['question_ids'] for c in batch)

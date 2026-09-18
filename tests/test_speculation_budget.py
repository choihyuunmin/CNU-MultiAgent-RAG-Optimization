from dataclasses import replace

import pytest

from cnu_rag_optimization.speculation_budget import (
    BudgetChoice, DecodeContext, DecodeMeasurement, DraftBudgetPolicy, calibrate, measure_decode_window,
)


def context(batch=8):
    return DecodeContext("engine-v1", "stage-v1", batch, 4)


def row(i, k, cost, batch=8, tokens=100):
    return DecodeMeasurement(context(batch), str(i), k, tokens, cost, "engine_decode_window")


def test_measured_cost_beats_acceptance_or_longest_draft_heuristic():
    rows = [row(i, k, cost) for i in range(8) for k, cost in [(0, 100), (4, 60), (16, 130)]]
    choice, = calibrate(rows, bootstrap_samples=100)
    assert choice.draft_tokens == 4 and choice.gain_fraction == pytest.approx(.4)


def test_high_load_can_select_plain_decode_without_removing_model_request():
    rows = [row(i, k, cost, 64) for i in range(8) for k, cost in [(0, 100), (4, 120)]]
    policy = DraftBudgetPolicy(calibrate(rows, bootstrap_samples=100))
    invoked = []
    choice, draft = policy.propose(context(64), lambda k: invoked.append(k),
                                  supported_budgets=frozenset({0, 4}), coordinated_dynamic_k=True)
    assert choice.draft_tokens == 0 and draft == [] and invoked == []


def test_actual_emitted_tokens_are_used_instead_of_assuming_k_plus_one():
    rows = [row(i, 0, 100, tokens=100) for i in range(8)]
    rows += [row(i, 4, 150, tokens=100) for i in range(8)]
    assert calibrate(rows, bootstrap_samples=100)[0].draft_tokens == 0


@pytest.mark.parametrize("source", ["client_latency", "oracle", "synthetic"])
def test_non_engine_timing_cannot_calibrate_live_policy(source):
    with pytest.raises(ValueError, match="engine decode"):
        replace(row(0, 0, 10), measurement_source=source)


def test_missing_candidate_trial_does_not_cherry_pick_fast_subset():
    rows = [row(i, 0, 100) for i in range(8)] + [row(i, 4, 50) for i in range(7)]
    choice, = calibrate(rows, bootstrap_samples=100)
    assert choice.draft_tokens == 0 and choice.reason == "incomplete_paired_trials"


def test_duplicate_measurements_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        calibrate([row(0, 0, 100), row(0, 0, 100)])


@pytest.mark.parametrize("change", [dict(batch_size=9), dict(kv_bucket=5),
    dict(engine_fingerprint="engine-v2"), dict(workload_fingerprint="another-agent")])
def test_unseen_state_never_inherits_calibration(change):
    policy = DraftBudgetPolicy([BudgetChoice(context(), 4, "calibrated_cost_gain", 8)])
    assert policy.select(replace(context(), **change), supported_budgets=frozenset({0, 4})).draft_tokens == 0


def test_unverified_engine_cannot_change_proposer_k():
    policy = DraftBudgetPolicy([])
    with pytest.raises(RuntimeError, match="unverified"):
        policy.propose(context(), lambda k: [1], supported_budgets=frozenset({0, 4}), coordinated_dynamic_k=False)


@pytest.mark.parametrize("draft", [[1, 2, 3, 4, 5], [-1], [True], "tokens"])
def test_invalid_proposal_is_never_returned_to_verifier(draft):
    policy = DraftBudgetPolicy([BudgetChoice(context(), 4, "calibrated_cost_gain", 8)])
    with pytest.raises(ValueError, match="proposer"):
        policy.propose(context(), lambda k: draft, supported_budgets=frozenset({0, 4}), coordinated_dynamic_k=True)


def test_valid_proposal_returns_original_tokens_and_budget():
    policy = DraftBudgetPolicy([BudgetChoice(context(), 4, "calibrated_cost_gain", 8)])
    tokens = [3, 4, 5]
    choice, draft = policy.propose(context(), lambda k: tokens,
                                 supported_budgets=frozenset({0, 4}), coordinated_dynamic_k=True)
    assert choice.draft_tokens == 4 and draft is tokens


def test_uncertain_speed_does_not_pass_gate():
    rows = [row(i, 0, 100) for i in range(8)] + [row(i, 4, 1 if i % 2 else 180) for i in range(8)]
    choice, = calibrate(rows)
    assert choice.draft_tokens == 0


def test_engine_hook_waits_for_device_and_returns_unmodified_output():
    events, result = [], {"committed": [7, 8, 9]}
    clock = iter([1., 1.015])
    def execute():
        events.append("execute")
        return result
    output, observed = measure_decode_window(context(), "trial-1", 4, execute,
        lambda: events.append("synchronize"), lambda out: len(out["committed"]),
        decode_only=True, clock=lambda: next(clock))
    assert output is result
    assert events == ["synchronize", "execute", "synchronize"]
    assert observed.emitted_tokens == 3 and observed.step_wall_ms == pytest.approx(15)


def test_engine_hook_rejects_prefill_before_execution():
    with pytest.raises(ValueError, match="prefill"):
        measure_decode_window(context(), "trial", 4, lambda: pytest.fail("executed"),
                              lambda: None, len, decode_only=False)

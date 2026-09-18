"""Calibrate draft length using measured decode cost, not acceptance alone.

This is an engine-side policy interface, not an HTTP parameter or vLLM patch.
The engine must coordinate its scheduler, proposer and verifier with the same K.
Client latency and oracle step counts are deliberately not accepted as costs.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import asdict, dataclass
import math
import random
import time
from typing import Callable, Iterable


def _integer(value, low, name):
    if type(value) is not int or value < low:
        raise ValueError(f"invalid {name}")


@dataclass(frozen=True)
class DecodeContext:
    engine_fingerprint: str
    workload_fingerprint: str
    batch_size: int
    kv_bucket: int

    def __post_init__(self):
        if not self.engine_fingerprint or not self.workload_fingerprint:
            raise ValueError("engine and workload fingerprints are required")
        _integer(self.batch_size, 1, "batch size")
        _integer(self.kv_bucket, 0, "KV bucket")
        if self.kv_bucket > 9:
            raise ValueError("KV bucket must be floor(usage_fraction * 10), capped at 9")


@dataclass(frozen=True)
class DecodeMeasurement:
    context: DecodeContext
    trial_id: str
    draft_tokens: int
    emitted_tokens: int
    step_wall_ms: float
    measurement_source: str

    def __post_init__(self):
        if not self.trial_id:
            raise ValueError("paired trial ID required")
        _integer(self.draft_tokens, 0, "draft tokens")
        _integer(self.emitted_tokens, 1, "emitted tokens")
        if (isinstance(self.step_wall_ms, bool) or not math.isfinite(self.step_wall_ms)
                or self.step_wall_ms <= 0):
            raise ValueError("positive finite decode time required")
        if self.measurement_source != "engine_decode_window":
            raise ValueError("engine decode measurements required; client/oracle costs are invalid")


def measure_decode_window(context: DecodeContext, trial_id: str, draft_tokens: int,
                          execute: Callable, synchronize: Callable[[], None],
                          count_emitted: Callable, *, decode_only: bool,
                          clock: Callable[[], float] = time.perf_counter):
    """Engine hook for a decode-only window, including drafting and verification.

    synchronize must wait for the actual participating device work (all ranks
    for distributed inference). execute must include the proposer and target
    verification, not just a CPU enqueue. Engine wall time includes host work;
    it is NOT a GPU-kernel-time measurement. Never use this around an HTTP call.
    Caller keeps the declared context stable through the window. Prefill/mixed
    windows are ineligible. No retry, token change or acceptance approximation.
    """
    if decode_only is not True:
        raise ValueError("prefill or mixed windows cannot calibrate decode cost")
    _integer(draft_tokens, 0, "draft tokens")
    synchronize()
    started = clock()
    output = execute()
    synchronize()
    elapsed_ms = (clock() - started) * 1000
    measurement = DecodeMeasurement(context, trial_id, draft_tokens, count_emitted(output),
                                    elapsed_ms, "engine_decode_window")
    return output, measurement


@dataclass(frozen=True)
class BudgetChoice:
    context: DecodeContext
    draft_tokens: int
    reason: str
    paired_trials: int
    gain_fraction: float | None = None
    gain_ci95: tuple[float, float] | None = None


def calibrate(measurements: Iterable[DecodeMeasurement], *, min_pairs=8,
              minimum_gain=.05, bootstrap_samples=2000, seed=90216) -> list[BudgetChoice]:
    """Freeze a per-exact-context choice from complete paired calibration trials.

    Trials must represent matched workload windows, including proposer and
    verifier completion time on the engine. No extrapolation to another batch,
    workload, memory bucket or engine. This gate is NOT a held-out speed claim.
    An incomplete arm invalidates the context instead of dropping slow trials.
    """
    _integer(min_pairs, 2, "minimum pairs")
    _integer(bootstrap_samples, 100, "bootstrap samples")
    if not math.isfinite(minimum_gain) or not 0 <= minimum_gain < 1:
        raise ValueError("minimum gain must be in [0, 1)")
    grouped = defaultdict(lambda: defaultdict(dict))
    for row in measurements:
        arm = grouped[row.context][row.draft_tokens]
        if row.trial_id in arm:
            raise ValueError("duplicate context/K/trial observation")
        arm[row.trial_id] = row
    choices = []
    for context, arms in grouped.items():
        baseline = arms.get(0, {})
        n = len(baseline)
        if n < min_pairs:
            choices.append(BudgetChoice(context, 0, "insufficient_baseline", n))
            continue
        if any(set(arm) != set(baseline) for arm in arms.values()):
            choices.append(BudgetChoice(context, 0, "incomplete_paired_trials", n))
            continue
        ids = sorted(baseline)
        base_cost = [baseline[i].step_wall_ms / baseline[i].emitted_tokens for i in ids]
        candidates = []
        for k, arm in sorted(arms.items()):
            if k == 0:
                continue
            cost = [arm[i].step_wall_ms / arm[i].emitted_tokens for i in ids]
            gain = 1 - sum(cost) / sum(base_cost)
            rng = random.Random(seed)
            samples = []
            for _ in range(bootstrap_samples):
                positions = [rng.randrange(n) for _ in range(n)]
                samples.append(1 - sum(cost[i] for i in positions) / sum(base_cost[i] for i in positions))
            samples.sort()
            ci = (samples[int(.025 * bootstrap_samples)], samples[int(.975 * bootstrap_samples)])
            if gain >= minimum_gain and ci[0] > 0:
                candidates.append(BudgetChoice(context, k, "calibrated_cost_gain", n, gain, ci))
        # Prefer the smallest K on a tie. Selection bias requires held-out testing.
        choices.append(max(candidates, key=lambda x: (x.gain_fraction, -x.draft_tokens))
                       if candidates else BudgetChoice(context, 0, "no_calibrated_gain", n))
    return choices


class DraftBudgetPolicy:
    def __init__(self, choices: Iterable[BudgetChoice]):
        self._choices = {}
        for choice in choices:
            if choice.context in self._choices:
                raise ValueError("duplicate policy context")
            _integer(choice.draft_tokens, 0, "draft tokens")
            self._choices[choice.context] = choice

    def select(self, context: DecodeContext, *, supported_budgets: frozenset[int]) -> BudgetChoice:
        if 0 not in supported_budgets or any(type(k) is not int or k < 0 for k in supported_budgets):
            raise ValueError("backend must support ordinary decode (K=0)")
        choice = self._choices.get(context)
        if choice is None:
            return BudgetChoice(context, 0, "unmeasured_context", 0)
        if choice.draft_tokens not in supported_budgets:
            return BudgetChoice(context, 0, "unsupported_budget", choice.paired_trials)
        return choice

    def propose(self, context: DecodeContext, proposer: Callable[[int], list[int]], *,
                supported_budgets: frozenset[int], coordinated_dynamic_k: bool):
        """Engine seam: K=0 skips the drafter, never the target model decode.

        A backend must explicitly attest that scheduler and verification buffers
        use this same K. Changing an n-gram proposer's field alone is unsafe.
        The returned draft always goes through the original target verifier.
        """
        if coordinated_dynamic_k is not True:
            raise RuntimeError("coordinated scheduler/proposer/verifier support is unverified")
        choice = self.select(context, supported_budgets=supported_budgets)
        if choice.draft_tokens == 0:
            return choice, []
        draft = proposer(choice.draft_tokens)
        if (not isinstance(draft, list) or len(draft) > choice.draft_tokens
                or any(type(token) is not int or token < 0 for token in draft)):
            raise ValueError("proposer violated selected draft budget")
        return choice, draft

    def to_dict(self):
        return {"schema": 1, "scope": "engine_decode_budget_only", "choices": [
            asdict(c) for c in self._choices.values()]}

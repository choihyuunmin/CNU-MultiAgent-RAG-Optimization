"""Framework-independent, call-local reference virtualization.

The integration declares reference fields, rather than the harness guessing from
agent roles or prompt words. JSON outside those string values stays byte-exact.
Reversible representation does NOT imply equivalent model decisions. Calibration
and a task-specific validator remain necessary. No DAG or provider is required.
"""
from __future__ import annotations

from collections.abc import Awaitable, Callable, Sequence
from copy import deepcopy
from dataclasses import dataclass
import json
import math
import random
import statistics
import time

from .structured_harness import strict_json

Path = tuple[str | int, ...]


def _get(value, path: Path):
    for part in path:
        value = value[part]
    return value


def _spans(raw: str) -> dict[Path, tuple[int, int]]:
    """Locate JSON values with the standard decoder; reject ambiguous JSON first."""
    strict_json(raw)
    decoder = json.JSONDecoder()
    result = {}

    def skip(pos):
        while pos < len(raw) and raw[pos].isspace():
            pos += 1
        return pos

    def walk(pos, path):
        pos = skip(pos)
        start = pos
        if raw[pos] == '{':
            pos = skip(pos + 1)
            while raw[pos] != '}':
                key, pos = decoder.raw_decode(raw, pos)
                pos = skip(pos)
                pos = walk(pos + 1, path + (key,))  # validated colon
                pos = skip(pos)
                if raw[pos] == ',':
                    pos = skip(pos + 1)
                else:
                    break
            pos += 1
        elif raw[pos] == '[':
            pos = skip(pos + 1)
            index = 0
            while raw[pos] != ']':
                pos = skip(walk(pos, path + (index,)))
                index += 1
                if raw[pos] == ',':
                    pos = skip(pos + 1)
                else:
                    break
            pos += 1
        else:
            _, pos = decoder.raw_decode(raw, pos)
        result[path] = (start, pos)
        return pos

    walk(0, ())
    return result


def _patch(raw: str, replacements: dict[Path, str]) -> str:
    spans = _spans(raw)
    patches = sorted((spans[path], json.dumps(value, ensure_ascii=False))
                     for path, value in replacements.items())
    for ((left, right), _), ((next_left, _), _) in zip(patches, patches[1:]):
        if right > next_left:
            raise ValueError('overlapping reference paths')
    for (left, right), replacement in reversed(patches):
        raw = raw[:left] + replacement + raw[right:]
    return raw


@dataclass(frozen=True)
class ReferenceContract:
    """Canonical object list and output reference list, supplied by the caller.

    identity must change with the model revision, instructions, schema, validator
    or tool contract. This prevents reusing a calibration across those changes.
    read_only permits one original-input model retry, never a repeated side effect.
    """
    identity: str
    collection: Path
    identifier: str
    selected: Path
    read_only: bool = True

    def __post_init__(self):
        if not self.identity or not self.identifier or not self.selected:
            raise ValueError('identity, identifier and selected path are required')


@dataclass(frozen=True)
class Representation:
    original: str
    encoded: str
    contract: ReferenceContract
    originals: tuple[str, ...]
    reason: str

    @property
    def enabled(self):
        return bool(self.originals)

    def restore(self, output: str) -> str:
        if not self.enabled:
            return output
        data = strict_json(output)
        selected = _get(data, self.contract.selected)
        mapping = {str(i): value for i, value in enumerate(self.originals)}
        if not isinstance(selected, list) or any(
            not isinstance(value, str) or value not in mapping for value in selected
        ):
            raise ValueError('unknown reference in model output')
        # Handles in reasoning/prose cannot safely be interpreted or restored.
        # The output contract must keep all references in the declared list.
        return _patch(output, {self.contract.selected + (i,): mapping[value]
                               for i, value in enumerate(selected)})


def compile_references(raw: str, contract: ReferenceContract, *,
                       protected: Sequence[str] = (), min_saving_chars: int = 1) -> Representation:
    def bypass(reason):
        return Representation(raw, raw, contract, (), reason)
    if not contract.read_only:
        return bypass('side_effecting_call')
    if type(min_saving_chars) is not int or min_saving_chars < 1:
        raise ValueError('min_saving_chars must be a positive integer')
    try:
        data = strict_json(raw)
        objects = _get(data, contract.collection)
        if not isinstance(objects, list) or not objects:
            return bypass('empty_or_unknown_collection')
        ids = [item[contract.identifier] for item in objects]
        if (any(not isinstance(x, str) or not x for x in ids)
                or len(set(ids)) != len(ids)):
            return bypass('ambiguous_identifiers')
        expected = deepcopy(data)
        expected_objects = _get(expected, contract.collection)
        replacements = {}
        for i, item in enumerate(expected_objects):
            item[contract.identifier] = str(i)
            replacements[contract.collection + (i, contract.identifier)] = str(i)
        # Escape analysis: an opaque ID may not also carry meaning in any other
        # string value, key, user instruction, history, or external prompt field.
        masked = deepcopy(data)
        for item in _get(masked, contract.collection):
            item[contract.identifier] = None
        def strings(value):
            if isinstance(value, str):
                yield value
            elif isinstance(value, dict):
                for key, child in value.items():
                    yield key
                    yield from strings(child)
            elif isinstance(value, list):
                for child in value:
                    yield from strings(child)
        external = tuple(strings(masked)) + tuple(protected)
        if any(identifier in field for identifier in ids for field in external):
            return bypass('reference_escapes_contract')
        encoded = _patch(raw, replacements)
        if strict_json(encoded) != expected:
            return bypass('roundtrip_mismatch')
        if len(raw) - len(encoded) < min_saving_chars:
            return bypass('insufficient_saving')
        return Representation(raw, encoded, contract, tuple(ids), 'compiled')
    except (ValueError, TypeError, KeyError, IndexError, RecursionError):
        return bypass('unknown_input_contract')


@dataclass(frozen=True)
class CalibrationPair:
    case_id: str
    original_s: float
    candidate_s: float
    original_loss: float
    candidate_loss: float


@dataclass(frozen=True)
class Calibration:
    identity: str
    enabled: bool
    cases: int
    reduction_lower: float
    excess_loss_upper: float
    reason: str


def calibrate(identity: str, pairs: Sequence[CalibrationPair], *,
              min_cases: int = 12, min_reduction: float = .02,
              max_excess_loss: float = .01, resamples: int = 2000) -> Calibration:
    """Frozen empirical gate; paired case bootstrap, not a correctness certificate.

    Use independent calibration cases and include all failures and fallback time.
    Loss must be computed by the caller's fixed evaluator on [0, 1]. Repeated
    measurements of one case must be averaged before calling, not counted as new
    independent cases. An all-zero sampled loss has unobserved-error uncertainty;
    a passing gate does not prove noninferiority or authorize production release.
    """
    if (type(min_cases) is not int or min_cases < 2 or resamples < 100
            or not math.isfinite(min_reduction) or not 0 <= min_reduction < 1
            or not math.isfinite(max_excess_loss) or not 0 <= max_excess_loss <= 1):
        raise ValueError('invalid calibration bounds')
    if len({p.case_id for p in pairs}) != len(pairs):
        raise ValueError('average repeated cases before calibration')
    for p in pairs:
        if (not all(math.isfinite(x) for x in (p.original_s, p.candidate_s,
                                             p.original_loss, p.candidate_loss))
                or min(p.original_s, p.candidate_s) <= 0
                or not 0 <= p.original_loss <= 1 or not 0 <= p.candidate_loss <= 1):
            raise ValueError('invalid calibration observation')
    if len(pairs) < min_cases:
        return Calibration(identity, False, len(pairs), 0, 1, 'insufficient_cases')
    rng = random.Random(20260916)
    speed, loss = [], []
    for _ in range(resamples):
        sample = rng.choices(pairs, k=len(pairs))
        speed.append(1 - sum(p.candidate_s for p in sample) / sum(p.original_s for p in sample))
        loss.append(statistics.fmean(p.candidate_loss - p.original_loss for p in sample))
    lower = sorted(speed)[int(.025 * resamples)]
    upper = sorted(loss)[int(.975 * resamples)]
    enabled = lower >= min_reduction and upper <= max_excess_loss
    return Calibration(identity, enabled, len(pairs), lower, upper,
                       'empirical_gate_passed' if enabled else 'benefit_or_quality_rejected')


@dataclass(frozen=True)
class BoundaryResult:
    output: str
    transformed: bool
    fallback: bool
    saved_input_chars: int
    elapsed_s: float
    reason: str


class ReferenceHarness:
    """Async callback boundary usable in a chain, fanout, hierarchy or loop.

    Agent routing and side effects stay with the application. A contract miss
    uses the original callback once. A transformed invalid result causes at most
    one read-only retry. Cancellation/network failures propagate without retry.
    """
    def __init__(self, calibration: Calibration, *, min_saving_chars: int = 1):
        self.calibration = calibration
        self.min_saving_chars = min_saving_chars

    async def run(self, raw: str, contract: ReferenceContract, *,
                  invoke: Callable[[str], Awaitable[str]],
                  validate: Callable[[object, object], bool],
                  protected: Sequence[str] = ()) -> BoundaryResult:
        started = time.perf_counter()
        if self.calibration.enabled and self.calibration.identity == contract.identity:
            representation = compile_references(raw, contract, protected=protected,
                                                 min_saving_chars=self.min_saving_chars)
        else:
            representation = Representation(raw, raw, contract, (), 'calibration_miss')
        result = await invoke(representation.encoded)
        fallback = False
        try:
            output = representation.restore(result)
            valid = validate(strict_json(output), strict_json(raw))
        except (ValueError, TypeError, KeyError, IndexError):
            valid = False
        if not valid and representation.enabled:
            fallback = True
            output = await invoke(raw)
            valid = validate(strict_json(output), strict_json(raw))
        if not valid:
            raise ValueError('output violates the original application contract')
        return BoundaryResult(output, representation.enabled, fallback,
                              len(raw) - len(representation.encoded),
                              time.perf_counter() - started, representation.reason)

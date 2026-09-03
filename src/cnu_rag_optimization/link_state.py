"""Contract-constrained link-state routing for equivalent execution paths."""

from __future__ import annotations

import heapq
import math
import threading
import time
from dataclasses import dataclass, field
from itertools import count
from typing import Iterable, Mapping


@dataclass(frozen=True)
class LinkSpec:
    link_id: str
    source: str
    target: str
    initial_delay_ms: float
    capacity: int = 1
    required_contracts: frozenset[str] = frozenset()


@dataclass
class _LinkState:
    spec: LinkSpec
    ewma_delay_ms: float
    successes: int = 0
    failures: int = 0
    inflight: int = 0
    circuit_open_until: float = 0.0

    @property
    def samples(self) -> int:
        return self.successes + self.failures

    @property
    def reliability(self) -> float:
        return (self.successes + 2.0) / (self.samples + 2.0)


@dataclass(frozen=True)
class LinkRouteDecision:
    reachable: bool
    source: str
    target: str
    links: tuple[str, ...]
    nodes: tuple[str, ...]
    cost_ms: float
    reason: str

    @property
    def first_link(self) -> str:
        return self.links[0] if self.links else ""


@dataclass(frozen=True)
class LinkRouteToken:
    links: tuple[str, ...]
    started_at: float


@dataclass
class ContractLinkStateRouter:
    """Choose the lowest-cost contract-safe path with Dijkstra's algorithm."""

    ewma_alpha: float = 0.2
    reliability_floor: float = 0.98
    reliability_min_samples: int = 2
    failure_penalty_ms: float = 5_000.0
    failure_cooldown_seconds: float = 30.0
    _links: dict[str, _LinkState] = field(default_factory=dict, init=False)
    _adjacency: dict[str, list[str]] = field(default_factory=dict, init=False)
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)

    def add_link(self, spec: LinkSpec) -> None:
        if not spec.link_id or not spec.source or not spec.target:
            raise ValueError("link_id, source, and target are required")
        if spec.capacity < 1:
            raise ValueError("capacity must be at least 1")
        if not math.isfinite(spec.initial_delay_ms) or spec.initial_delay_ms < 0:
            raise ValueError("initial_delay_ms must be finite and non-negative")
        with self._lock:
            if spec.link_id in self._links:
                raise ValueError(f"duplicate link_id: {spec.link_id}")
            self._links[spec.link_id] = _LinkState(
                spec=spec,
                ewma_delay_ms=max(0.001, float(spec.initial_delay_ms)),
            )
            self._adjacency.setdefault(spec.source, []).append(spec.link_id)

    def _eligible(
        self,
        state: _LinkState,
        contracts: frozenset[str],
        now: float,
    ) -> bool:
        if not state.spec.required_contracts.issubset(contracts):
            return False
        if state.circuit_open_until > now:
            return False
        return not (
            state.samples >= self.reliability_min_samples
            and state.reliability < self.reliability_floor
        )

    def _cost(self, state: _LinkState) -> float:
        load = 1.0 + (state.inflight / max(1, state.spec.capacity))
        failure = (1.0 - state.reliability) * self.failure_penalty_ms
        return max(0.001, state.ewma_delay_ms) * load + failure

    def choose_path(
        self,
        source: str,
        target: str,
        *,
        contracts: Iterable[str] = (),
    ) -> LinkRouteDecision:
        contract_set = frozenset(str(item) for item in contracts)
        now = time.monotonic()
        with self._lock:
            serial = count()
            queue: list[tuple[float, int, str, tuple[str, ...], tuple[str, ...]]] = [
                (0.0, next(serial), source, (), (source,))
            ]
            best: dict[str, float] = {source: 0.0}
            while queue:
                cost, _, node, links, nodes = heapq.heappop(queue)
                if cost > best.get(node, math.inf):
                    continue
                if node == target:
                    return LinkRouteDecision(
                        True,
                        source,
                        target,
                        links,
                        nodes,
                        round(cost, 3),
                        "shortest_contract_path",
                    )
                for link_id in self._adjacency.get(node, ()):
                    state = self._links[link_id]
                    if not self._eligible(state, contract_set, now):
                        continue
                    next_cost = cost + self._cost(state)
                    next_node = state.spec.target
                    if next_cost >= best.get(next_node, math.inf):
                        continue
                    best[next_node] = next_cost
                    heapq.heappush(
                        queue,
                        (
                            next_cost,
                            next(serial),
                            next_node,
                            links + (link_id,),
                            nodes + (next_node,),
                        ),
                    )
        return LinkRouteDecision(
            False,
            source,
            target,
            (),
            (source,),
            math.inf,
            "no_contract_safe_path",
        )

    def begin(self, decision: LinkRouteDecision) -> LinkRouteToken:
        if not decision.reachable or not decision.links:
            raise ValueError("cannot begin unreachable route")
        with self._lock:
            for link_id in decision.links:
                self._links[link_id].inflight += 1
        return LinkRouteToken(decision.links, time.perf_counter())

    def finish(
        self,
        token: LinkRouteToken,
        *,
        success: bool,
        elapsed_ms: float | None = None,
        failed_link_id: str | None = None,
    ) -> None:
        elapsed = (
            max(0.001, float(elapsed_ms))
            if elapsed_ms is not None
            else max(0.001, (time.perf_counter() - token.started_at) * 1000.0)
        )
        with self._lock:
            states = [self._links[link_id] for link_id in token.links]
            total = sum(max(0.001, item.ewma_delay_ms) for item in states)
            for state in states:
                state.inflight = max(0, state.inflight - 1)
                if not success and failed_link_id and state.spec.link_id != failed_link_id:
                    continue
                sample = elapsed * max(0.001, state.ewma_delay_ms) / total
                alpha = min(1.0, max(0.001, self.ewma_alpha))
                state.ewma_delay_ms = (
                    alpha * sample + (1.0 - alpha) * state.ewma_delay_ms
                )
                if success:
                    state.successes += 1
                else:
                    state.failures += 1
                    state.circuit_open_until = max(
                        state.circuit_open_until,
                        time.monotonic() + max(0.0, self.failure_cooldown_seconds),
                    )

    def link_snapshot(self, link_id: str) -> Mapping[str, float | int | str]:
        with self._lock:
            state = self._links[link_id]
            return {
                "link_id": link_id,
                "ewma_delay_ms": round(state.ewma_delay_ms, 3),
                "reliability": round(state.reliability, 6),
                "samples": state.samples,
                "inflight": state.inflight,
            }

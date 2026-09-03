"""Fail-closed reuse of trace-verified execution procedures."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from threading import RLock
from typing import Iterable, Mapping


@dataclass(frozen=True)
class CompiledProcedure:
    procedure_id: str
    stage: str
    action: str
    required_contracts: frozenset[str]
    fallback_action: str
    priority: int = 0


@dataclass(frozen=True)
class ProcedureResolution:
    selected: bool
    stage: str
    procedure_id: str
    action: str
    fallback_action: str
    required_contracts: tuple[str, ...]
    reason: str

    def trace_attributes(self) -> dict[str, object]:
        return asdict(self)


class CompiledProcedureHarness:
    """Load a manifest once and authorize only exact contract subsets.

    The harness never stores or reuses answers. A contract miss returns the
    declared legacy action so the integrating application can fail closed.
    """

    def __init__(self, manifest_path: str | Path):
        self._path = Path(manifest_path)
        self._lock = RLock()
        self._loaded = False
        self._procedures_by_stage: dict[str, tuple[CompiledProcedure, ...]] = {}
        self._metadata: dict[str, object] = {}

    def _load(self) -> None:
        with self._lock:
            if self._loaded:
                return
            payload = json.loads(self._path.read_text(encoding="utf-8"))
            if payload.get("schema_version") != 1:
                raise ValueError("unsupported compiled procedure schema")

            grouped: dict[str, list[CompiledProcedure]] = {}
            seen: set[str] = set()
            for raw in payload.get("procedures") or []:
                procedure_id = str(raw.get("procedure_id") or "").strip()
                stage = str(raw.get("stage") or "").strip()
                action = str(raw.get("action") or "").strip()
                fallback = str(raw.get("fallback_action") or "").strip()
                if not procedure_id or not stage or not action or not fallback:
                    raise ValueError("compiled procedure fields are required")
                if procedure_id in seen:
                    raise ValueError(f"duplicate compiled procedure: {procedure_id}")
                seen.add(procedure_id)
                procedure = CompiledProcedure(
                    procedure_id=procedure_id,
                    stage=stage,
                    action=action,
                    required_contracts=frozenset(
                        str(item).strip()
                        for item in (raw.get("required_contracts") or [])
                        if str(item).strip()
                    ),
                    fallback_action=fallback,
                    priority=int(raw.get("priority") or 0),
                )
                grouped.setdefault(stage, []).append(procedure)

            self._procedures_by_stage = {
                stage: tuple(
                    sorted(items, key=lambda item: (-item.priority, item.procedure_id))
                )
                for stage, items in grouped.items()
            }
            self._metadata = {
                "compiler": str(payload.get("compiler") or ""),
                "source_trace_sha256": str(payload.get("source_trace_sha256") or ""),
                "procedure_count": len(seen),
            }
            self._loaded = True

    def resolve(
        self,
        stage: str,
        *,
        available_contracts: Iterable[str],
    ) -> ProcedureResolution:
        self._load()
        available = frozenset(str(item) for item in available_contracts)
        candidates = self._procedures_by_stage.get(stage, ())
        for procedure in candidates:
            if procedure.required_contracts.issubset(available):
                return ProcedureResolution(
                    selected=True,
                    stage=stage,
                    procedure_id=procedure.procedure_id,
                    action=procedure.action,
                    fallback_action=procedure.fallback_action,
                    required_contracts=tuple(sorted(procedure.required_contracts)),
                    reason="typed_contract_match",
                )
        return ProcedureResolution(
            selected=False,
            stage=stage,
            procedure_id="",
            action="",
            fallback_action=(
                candidates[0].fallback_action if candidates else "legacy_agent"
            ),
            required_contracts=(),
            reason="contract_miss" if candidates else "stage_not_compiled",
        )

    def metadata(self) -> Mapping[str, object]:
        self._load()
        return dict(self._metadata)

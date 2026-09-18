"""In-process program-dependency overlay for the isolated MOLEG law-search app.

This layers the generalize-branch `ProgramHarness` onto the archived application's
`execute_search` node so that the ontology read starts as soon as its predecessor
finishes, instead of waiting on the original `asyncio.gather` barrier that also
waits for unrelated search branches. The join points still await the same results
and run the original merge code unchanged.

Nothing about the model, prompts, retrieval arguments, return shape, vLLM
settings, or existing timeouts changes. No admission limit, answer cache, call
elision, or question rewriting is introduced. Only the *start time* of already
declared, dependency-ready work moves earlier.

The overlay refuses to attach unless the archived `law_search/nodes.py` matches a
pinned SHA-256, so an unknown application revision fails closed rather than being
transformed blindly. It must run after the app package is importable and before
the LangGraph singleton is compiled.
"""
from __future__ import annotations

import hashlib
from dataclasses import asdict, is_dataclass
from functools import wraps
import importlib
import inspect
import json
from pathlib import Path

from program_harness_attachment import transform


def _emit(sink, prefix, value):
    line = prefix + " " + json.dumps(value, ensure_ascii=False)
    if sink is not None:
        sink.write(line + "\n")
        sink.flush()


def apply_program_overlay(*, fingerprint: str | None = None, event_sink=None,
                          harness_sink=None, start_when_ready=True):
    """Attach ProgramHarness to core.query_loop.law_search.nodes.execute_search.

    fingerprint: required SHA-256 of the reviewed nodes source file.
    event_sink: optional text sink for a one-line attachment record.
    harness_sink: optional text sink receiving per-request workflow-node metadata
        rows (node run time, overlap before the join, join wait). Never receives
        prompts, questions, or answers.
    start_when_ready: False preserves the original barrier as a measured control.

    The pinned source must establish that the moved read is required, read-only,
    independent of the other branches, and uses a rule-based keyword extractor.
    Fingerprinting proves revision identity, not those semantic properties.
    """
    from cnu_rag_optimization.program_harness import ProgramHarness

    nodes = importlib.import_module("core.query_loop.law_search.nodes")
    graph_module = importlib.import_module("core.query_loop.law_search.graph")
    builder = importlib.import_module("core.query_loop.builder")

    source_path = Path(nodes.__file__)
    observed = hashlib.sha256(source_path.read_bytes()).hexdigest()
    if fingerprint is not None and observed != fingerprint:
        raise RuntimeError("law_search nodes source does not match the pinned fingerprint")

    if getattr(builder, "_moleg_graph", None) is not None:
        raise RuntimeError("graph already compiled before program overlay attachment")
    if fingerprint is None:
        raise RuntimeError("a reviewed source fingerprint is required")

    original = nodes.execute_search
    if getattr(original, "__cnu_program_overlay__", False):
        raise RuntimeError("program overlay already attached")
    transformed = transform(inspect.getsource(original))
    # Verify the transform compiles against a real Python module before binding.
    compile(transformed, source_path.name, "exec")

    def sink_emit(row):
        _emit(harness_sink, "CNU_PROGRAM_HARNESS_V1", row)

    harness = ProgramHarness(sink_emit, start_when_ready=start_when_ready)

    def observe(kind, state, **values):
        workflow = harness.scope.get()
        identity = {"request_id": state.get("request_id"),
                    "workflow_id": workflow.workflow_id if workflow else None,
                    "node": "ontology_search" if kind == "program_read_result" else "execute_search"}
        def encode(value):
            if is_dataclass(value) and not isinstance(value, type):
                return asdict(value)
            raise TypeError("unsupported trace value")

        def digest(value):
            return hashlib.sha256(json.dumps(value, default=encode, sort_keys=True,
                ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()).hexdigest()

        try:
            sink_emit({"event": kind, **identity, "status": "ok",
                       **{key + "_hash": digest(value) for key, value in values.items()}})
        except Exception:
            # Do not fabricate a digest or change search results if tracing fails.
            try:
                sink_emit({"event": kind, **identity, "status": "trace_error"})
            except Exception:
                pass

    namespace = dict(original.__globals__)
    namespace["_pipeline"] = harness
    namespace["_program_read_observed"] = lambda state, question, result: observe(
        "program_read_result", state,
        input={"question": question, "limit": 10, "min_match": 3}, output=result)
    exec(compile(transformed, source_path.name, "exec"), namespace)

    @wraps(original)
    async def observed_search(state):
        result = await namespace["execute_search"](state)
        # The actual object handed to the next graph node, not just read inputs.
        observe("program_search_result", state, output=result)
        return result

    wrapped = harness.wrap(observed_search, request_key=lambda state: state.get("request_id"))
    wrapped.__cnu_program_overlay__ = True

    # Rebind both the module attribute and the name captured in the graph module,
    # so the not-yet-compiled subgraph registers the overlaid node.
    nodes.execute_search = wrapped
    graph_module.execute_search = wrapped

    record = {"enabled": True, "nodes": ["ontology_search"],
              "schedule": "ready" if start_when_ready else "barrier",
              "conditional_scenario_work_moved": False,
              "concurrency_limit": None, "model_or_payload_changed": False,
              "source_sha256": observed, "fingerprint_enforced": fingerprint is not None}
    _emit(event_sink, "CNU_PROGRAM_ATTACHMENT", record)
    return harness, record

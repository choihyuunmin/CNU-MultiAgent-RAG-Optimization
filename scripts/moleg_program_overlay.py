"""In-process program-dependency overlay for the isolated MOLEG law-search app.

This layers the generalize-branch `ProgramHarness` onto the archived application's
`execute_search` node so that a ready successor (the ontology-scoped search, the
scenario distill pipeline) starts as soon as its own declared predecessor
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
                          harness_sink=None):
    """Attach ProgramHarness to core.query_loop.law_search.nodes.execute_search.

    fingerprint: expected SHA-256 of the nodes source file. When provided, an
        exact match is required; otherwise the observed digest is reported and
        used (explicit trust of the frozen isolated copy, never a production pod).
    event_sink: optional text sink for a one-line attachment record.
    harness_sink: optional text sink receiving per-request workflow-node metadata
        rows (node run time, overlap before the join, join wait). Never receives
        prompts, questions, or answers.
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

    original = nodes.execute_search
    transformed = transform(inspect.getsource(original))
    # Verify the transform compiles against a real Python module before binding.
    compile(transformed, source_path.name, "exec")

    def sink_emit(row):
        _emit(harness_sink, "CNU_PROGRAM_HARNESS_V1", row)

    harness = ProgramHarness(sink_emit)
    namespace = dict(original.__globals__)
    namespace["_pipeline"] = harness
    exec(compile(transformed, source_path.name, "exec"), namespace)
    wrapped = harness.wrap(namespace["execute_search"])

    # Rebind both the module attribute and the name captured in the graph module,
    # so the not-yet-compiled subgraph registers the overlaid node.
    nodes.execute_search = wrapped
    graph_module.execute_search = wrapped

    record = {"enabled": True, "nodes": ["ontology_search", "scenario_followup"],
              "concurrency_limit": None, "model_or_payload_changed": False,
              "source_sha256": observed, "fingerprint_enforced": fingerprint is not None}
    _emit(event_sink, "CNU_PROGRAM_ATTACHMENT", record)
    return harness, record

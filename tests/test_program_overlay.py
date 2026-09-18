"""Offline guards for the in-process program-dependency overlay.

Uses a fixture source that mirrors the real search-node anchors so the transform
and rebind are exercised without importing the private application.
"""
import hashlib
import asyncio
import importlib
import io
import json
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

FIXTURE = '''\
import asyncio

async def execute_search(state):
    country = state.get("country", "")
    search_terms = state.get("search_terms", "")

    async def _ontology_prefetch():
        return state.get("searcher")

    async def _scenario_decompose():
        await state["unrelated"].wait()
        return state.get("scenarios", ["one"])

    ont_task = asyncio.create_task(_ontology_prefetch())
    scenario_task = asyncio.create_task(_scenario_decompose())
    ont_loader, scenarios = await asyncio.gather(ont_task, scenario_task)

    ont_searcher = ont_loader
    ont_results = []
    if ont_searcher is not None:
        try:
            _ont_base = search_terms or (state.get("user_prompt") or "").strip()
            ont_question = f"{country} {_ont_base}".strip() if country else _ont_base
            ont_kw, ont_cc, ont_results, ont_debug = await asyncio.to_thread(
                ont_searcher.search, question=ont_question, limit=10, min_match=3,
            )
        except Exception:
            ont_searcher = None

    if ont_results:
        return {"laws_list": ont_results, "source": "ontology"}

    if isinstance(scenarios, list) and scenarios:
        try:
            state["scenario_calls"].append(scenarios)
        except Exception:
            pass
    return {"laws_list": [], "source": "scenario"}
'''


def _install_fake_app(tmp_path):
    pkg_names = ["core", "core.query_loop", "core.query_loop.law_search"]
    for name in pkg_names:
        module = types.ModuleType(name)
        module.__path__ = []
        sys.modules[name] = module
    nodes_path = tmp_path / "nodes.py"
    nodes_path.write_text(FIXTURE)
    nodes = types.ModuleType("core.query_loop.law_search.nodes")
    nodes.__file__ = str(nodes_path)
    exec(compile(FIXTURE, str(nodes_path), "exec"), nodes.__dict__)
    sys.modules["core.query_loop.law_search.nodes"] = nodes
    graph = types.ModuleType("core.query_loop.law_search.graph")
    graph.execute_search = nodes.execute_search
    sys.modules["core.query_loop.law_search.graph"] = graph
    builder = types.ModuleType("core.query_loop.builder")
    builder._moleg_graph = None
    sys.modules["core.query_loop.builder"] = builder
    return nodes, graph, builder, nodes_path


def _cleanup():
    for name in ["core.query_loop.builder", "core.query_loop.law_search.graph",
                 "core.query_loop.law_search.nodes", "core.query_loop.law_search",
                 "core.query_loop", "core"]:
        sys.modules.pop(name, None)


def test_overlay_rebinds_search_node(tmp_path):
    nodes, graph, builder, nodes_path = _install_fake_app(tmp_path)
    try:
        overlay = importlib.import_module("moleg_program_overlay")
        original = nodes.execute_search
        sha = hashlib.sha256(nodes_path.read_bytes()).hexdigest()
        harness, record = overlay.apply_program_overlay(fingerprint=sha)
        assert record["enabled"] is True
        assert record["model_or_payload_changed"] is False
        assert record["source_sha256"] == sha
        assert record["nodes"] == ["ontology_search"]
        assert record["conditional_scenario_work_moved"] is False
        # both the module attribute and the graph-captured name are rebound
        assert nodes.execute_search is not original
        assert graph.execute_search is nodes.execute_search
        with pytest.raises(RuntimeError, match="already attached"):
            overlay.apply_program_overlay(fingerprint=sha)
    finally:
        _cleanup()


def test_overlay_fingerprint_mismatch_fails_closed(tmp_path):
    _install_fake_app(tmp_path)
    try:
        overlay = importlib.import_module("moleg_program_overlay")
        with pytest.raises(RuntimeError, match="pinned fingerprint"):
            overlay.apply_program_overlay(fingerprint="0" * 64)
    finally:
        _cleanup()


def test_overlay_refuses_after_graph_compiled(tmp_path):
    _, _, builder, _ = _install_fake_app(tmp_path)
    builder._moleg_graph = object()
    try:
        overlay = importlib.import_module("moleg_program_overlay")
        with pytest.raises(RuntimeError, match="already compiled"):
            overlay.apply_program_overlay(fingerprint=None)
    finally:
        _cleanup()


def test_overlay_requires_reviewed_fingerprint(tmp_path):
    _install_fake_app(tmp_path)
    try:
        overlay = importlib.import_module("moleg_program_overlay")
        with pytest.raises(RuntimeError, match="fingerprint is required"):
            overlay.apply_program_overlay()
    finally:
        _cleanup()


@pytest.mark.parametrize("ready", [False, True])
@pytest.mark.parametrize("outcome", ["hit", "empty", "error", "disabled"])
def test_runtime_matches_original_and_preserves_branch_priority(tmp_path, ready, outcome):
    nodes, _, _, path = _install_fake_app(tmp_path)
    original = nodes.execute_search
    sink = io.StringIO()
    calls = []

    class Searcher:
        def search(self, **kwargs):
            calls.append(kwargs)
            if outcome == "error":
                raise RuntimeError("retrieval failure")
            return ["term"], ["country"], ([{"id": "law-1"}] if outcome == "hit" else []), {}

    async def run():
        gate = asyncio.Event()
        gate.set()
        def state():
            return {"request_id": "q1", "country": "Japan", "search_terms": "private-query",
                    "searcher": None if outcome == "disabled" else Searcher(),
                    "unrelated": gate, "scenario_calls": []}
        before, after = state(), state()
        expected = await original(before)
        before_calls = list(calls)
        calls.clear()
        _, record = importlib.import_module("moleg_program_overlay").apply_program_overlay(
            fingerprint=hashlib.sha256(path.read_bytes()).hexdigest(), harness_sink=sink,
            start_when_ready=ready)
        actual = await nodes.execute_search(after)
        assert actual == expected
        assert before_calls == calls
        assert before["scenario_calls"] == after["scenario_calls"]
        assert len(after["scenario_calls"]) == (0 if outcome == "hit" else 1)
        assert record["schedule"] == ("ready" if ready else "barrier")
        assert not [t for t in asyncio.all_tasks() if t.get_name().startswith("cnu-program-")]
    try:
        asyncio.run(run())
        assert "private-query" not in sink.getvalue()
        rows = [json.loads(line.split(" ", 1)[1]) for line in sink.getvalue().splitlines()]
        workflow = next(r for r in rows if r["event"] == "workflow_closed")
        assert workflow["request_id"] == "q1"
        status = "error" if outcome == "error" else "ok"
        if outcome == "disabled" and not ready:
            status = "cancelled"  # Original branch never joins or runs this read.
        assert workflow["nodes"][0]["status"] == status
        results = [r for r in rows if r["event"] == "program_read_result"]
        assert len(results) == (1 if outcome in {"hit", "empty"} else 0)
        if results:
            assert len(results[0]["input_hash"]) == len(results[0]["output_hash"]) == 64
            assert results[0]["workflow_id"] == workflow["workflow_id"]
        handoff = next(r for r in rows if r["event"] == "program_search_result")
        assert handoff["status"] == "ok" and len(handoff["output_hash"]) == 64
        assert handoff["workflow_id"] == workflow["workflow_id"]
    finally:
        _cleanup()


@pytest.mark.parametrize("ready", [False, True])
def test_search_can_start_before_unrelated_branch_only_in_ready_mode(tmp_path, ready):
    nodes, _, _, path = _install_fake_app(tmp_path)

    async def run():
        unrelated, read_started = asyncio.Event(), asyncio.Event()
        # Event-controlled I/O stub: test dependency order, not simulated speedup.
        async def to_thread(function, **kwargs):
            read_started.set()
            return function(**kwargs)
        nodes.asyncio = types.SimpleNamespace(create_task=asyncio.create_task,
            gather=asyncio.gather, to_thread=to_thread)
        searcher = types.SimpleNamespace(search=lambda **kwargs: ([], [], [{"id": "1"}], {}))
        importlib.import_module("moleg_program_overlay").apply_program_overlay(
            fingerprint=hashlib.sha256(path.read_bytes()).hexdigest(), start_when_ready=ready)
        state = {"request_id": "q", "unrelated": unrelated, "searcher": searcher,
                 "scenario_calls": []}
        task = asyncio.create_task(nodes.execute_search(state))
        if ready:
            await asyncio.wait_for(read_started.wait(), timeout=1)
            assert not unrelated.is_set() and not task.done()
        else:
            # Drain a bounded number of loop turns; no wall-clock benchmark.
            for _ in range(10):
                await asyncio.sleep(0)
            assert not read_started.is_set() and not task.done()
        unrelated.set()
        assert (await asyncio.wait_for(task, timeout=1))["source"] == "ontology"
        assert not state["scenario_calls"]
    try:
        asyncio.run(run())
    finally:
        _cleanup()

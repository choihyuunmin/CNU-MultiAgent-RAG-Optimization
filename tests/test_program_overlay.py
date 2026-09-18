"""Offline guards for the in-process program-dependency overlay.

Uses a fixture source that mirrors the real search-node anchors so the transform
and rebind are exercised without importing the private application.
"""
import hashlib
import importlib
import sys
import types
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

FIXTURE = '''\
import asyncio

ENABLE_SCENARIO_DECOMPOSITION = True


async def execute_search(state):
    country = state.get("country", "")
    search_terms = state.get("search_terms", "")

    async def _ontology_prefetch():
        return object()

    async def _scenario_decompose():
        return []

    ont_task = asyncio.create_task(_ontology_prefetch())
    scenario_task = asyncio.create_task(_scenario_decompose())
    ont_loader, scenarios = await asyncio.gather(ont_task, scenario_task)

    ont_searcher = ont_loader
    if ont_searcher is not None:
        try:
            _ont_base = search_terms or (state.get("user_prompt") or "").strip()
            ont_question = f"{country} {_ont_base}".strip() if country else _ont_base
            ont_kw, ont_cc, ont_results, ont_debug = await asyncio.to_thread(
                ont_searcher.search, question=ont_question, limit=10, min_match=3,
            )
        except Exception:
            ont_searcher = None

    if ENABLE_SCENARIO_DECOMPOSITION and isinstance(scenarios, list) and scenarios:
        try:
            hits_by_label, search_source_from_search = await scenario_search.parallel_scenario_search(
                scenarios=scenarios, country=country, request_id="-",
                merged_keywords=[],
            )
            distill_results = await asyncio.gather(*[])
            distilled = {s.label: picked for s, picked in zip(scenarios, distill_results)}
        except Exception:
            pass
    return {"laws_list": []}
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
        # both the module attribute and the graph-captured name are rebound
        assert nodes.execute_search is not original
        assert graph.execute_search is nodes.execute_search
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

import ast
import sys
from pathlib import Path

scripts = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(scripts))

from program_harness_attachment import transform


def test_transform_has_named_ready_successors():
    source = '''async def execute_search():
    async def _ontology_prefetch():
        return None
    async def _scenario_decompose():
        return None
    ont_task = asyncio.create_task(_ontology_prefetch())
    scenario_task = asyncio.create_task(_scenario_decompose())
    scenarios = await scenario_task
    ont_searcher = await ont_task
    if ont_searcher:
        try:
            _ont_base = search_terms or (state.get("user_prompt") or "").strip()
            ont_kw, ont_cc, ont_results, ont_debug = await asyncio.to_thread(
                ont_searcher.search, question=ont_question, limit=10, min_match=3,
            )
        except Exception:
            pass
    if scenarios:
        try:
            hits_by_label, search_source_from_search = await scenario_search.parallel_scenario_search(
                scenarios=scenarios)
            distill_results = await asyncio.gather(*distill_tasks)
            distilled = {s.label: picked for s, picked in zip(scenarios, distill_results)}
        except Exception:
            pass
'''
    transformed = transform(source)
    ast.parse(transformed)
    assert transformed.count('_pipeline.after("ontology_search"') == 1
    assert transformed.count('_pipeline.after("scenario_followup"') == 1
    assert transformed.count("await _pipeline.join") == 2

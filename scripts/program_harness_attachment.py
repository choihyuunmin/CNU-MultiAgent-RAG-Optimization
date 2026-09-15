"""Attach ready-successor execution to one fingerprinted search function.

Private application source stays outside this repository. Exact anchors and a
whole-file fingerprint make an unknown application revision fail closed.
"""


def transform(source):
    scenario_start = '            hits_by_label, search_source_from_search = await scenario_search.parallel_scenario_search('
    scenario_end = '            distilled = {s.label: picked for s, picked in zip(scenarios, distill_results)}'
    ontology_anchor = '    ont_task = asyncio.create_task(_ontology_prefetch())'
    scenario_task = '    scenario_task = asyncio.create_task(_scenario_decompose())'
    for token in (scenario_start, scenario_end, ontology_anchor, scenario_task):
        if source.count(token) != 1:
            raise ValueError("unrecognized program integration source")

    left, right = source.index(scenario_start), source.index(scenario_end) + len(scenario_end)
    block = source[left:right]
    helper = (
        '    async def _scenario_pipeline(scenarios):\n'
        '        if not ENABLE_SCENARIO_DECOMPOSITION or not isinstance(scenarios, list) or not scenarios:\n'
        '            return None\n'
        + '\n'.join(line[4:] if line.startswith('    ') else line for line in block.splitlines())
        + '\n        return hits_by_label, search_source_from_search, distilled\n\n'
    )
    source = source[:left] + (
        '            hits_by_label, search_source_from_search, distilled = '
        'await _pipeline.join(scenario_followup)'
    ) + source[right:]
    source = source.replace(ontology_anchor, helper + ontology_anchor)
    source = source.replace(
        scenario_task,
        scenario_task + '\n    scenario_followup = _pipeline.after('
        '"scenario_followup", scenario_task, _scenario_pipeline)',
    )

    ontology_start = '            _ont_base = search_terms or (state.get("user_prompt") or "").strip()'
    ontology_end = '                ont_searcher.search, question=ont_question, limit=10, min_match=3,\n            )'
    if source.count(ontology_start) != 1 or source.count(ontology_end) != 1:
        raise ValueError("unrecognized ontology integration source")
    left, right = source.index(ontology_start), source.index(ontology_end) + len(ontology_end)
    block = source[left:right]
    helper = (
        '    async def _ontology_pipeline(ont_searcher):\n'
        '        if ont_searcher is None or isinstance(ont_searcher, Exception):\n'
        '            return None\n'
        + '\n'.join(line[4:] if line.startswith('    ') else line for line in block.splitlines())
        + '\n        return ont_kw, ont_cc, ont_results, ont_debug\n\n'
    )
    source = source[:left] + (
        '            ont_kw, ont_cc, ont_results, ont_debug = '
        'await _pipeline.join(ontology_followup)'
    ) + source[right:]
    source = source.replace(
        ontology_anchor,
        helper + ontology_anchor + '\n    ontology_followup = _pipeline.after('
        '"ontology_search", ont_task, _ontology_pipeline)',
    )
    return source

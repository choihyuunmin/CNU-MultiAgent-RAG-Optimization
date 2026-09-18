"""Attach ready-successor execution to one fingerprinted search function.

Private application source stays outside this repository. The caller must pin
the reviewed source. Only the unconditional ontology read moves; the conditional
scenario pipeline stays in place, after the ontology-result decision.
"""


def transform(source):
    ontology_anchor = '    ont_task = asyncio.create_task(_ontology_prefetch())'
    if source.count(ontology_anchor) != 1 or '_pipeline.' in source:
        raise ValueError("unrecognized program integration source")

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
        + '\n        _program_read_observed(state, ont_question, '
        '(ont_kw, ont_cc, ont_results, ont_debug))'
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

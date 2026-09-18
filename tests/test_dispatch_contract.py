import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
from cnu_rag_optimization.typed_dispatch import try_typed_single_tool_dispatch


def dispatch(handler, arguments, validator=None):
    return asyncio.run(try_typed_single_tool_dispatch(
        available_tools={'search': handler}, candidate_tool='search',
        arguments=arguments, argument_validator=validator))


def test_validator_required_and_no_tool_execution():
    async def forbidden(args):
        raise AssertionError('must not execute')
    assert dispatch(forbidden, {'x': 1}).reason == 'validator_missing'


def test_nested_inputs_isolated_and_result_identity_preserved():
    arguments = {'keywords': ['a']}
    output = object()
    async def tool(args):
        args['keywords'].append('b')
        return output
    result = dispatch(tool, arguments, lambda args: True)
    assert arguments == {'keywords': ['a']}
    assert result.value is output


@pytest.mark.parametrize('args', [{'x': (1, 2)}, {'x': float('nan')}, {1: 'a'}])
def test_lossy_arguments_rejected(args):
    async def forbidden(args):
        raise AssertionError('must not execute')
    assert not dispatch(forbidden, args, lambda args: True).dispatched


def test_mutating_validator_rejected_before_execution():
    args = {'x': [1]}
    def validator(copy):
        copy['x'].append(2)
        return True
    async def forbidden(copy):
        raise AssertionError('must not execute')
    assert not dispatch(forbidden, args, validator).dispatched
    assert args == {'x': [1]}


def test_handler_failure_propagates_without_retry():
    calls = []
    async def handler(args):
        calls.append(args)
        raise RuntimeError('handler failure')
    with pytest.raises(RuntimeError, match='handler failure'):
        dispatch(handler, {'x': 1}, lambda args: True)
    assert len(calls) == 1


def test_attachment_original_and_direct_preserve_next_stage_object():
    path = Path(__file__).parents[1] / 'docs/paper-draft-20260917/followup/review_dispatch_attachment.py'
    spec = importlib.util.spec_from_file_location('dispatch_attachment_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    async def run(mode):
        app, events = SimpleNamespace(), []
        async def handler(args, request_id):
            return '{"ids":["a"]}'
        app.tool_search_laws = handler
        app._parse_search_payload = lambda value: ([{'id': 'a'}], ['a'], 'test')
        async def original(**kw):
            value = await app.tool_search_laws({'country': kw['country'],
                'search_terms': kw['search_terms'], 'keywords': kw['merged_keywords']}, kw['request_id'])
            laws, ids, source = app._parse_search_payload(value)
            return {'search_tool_result': value, 'laws_list': laws, 'search_law_ids': ids,
                    'search_source': source, 'search_params': {'search_terms': kw['search_terms'],
                    'keywords': kw['merged_keywords'], 'country': kw['country']}}
        app.run_search = original
        module.install(app, mode, emit=events.append)
        result = await app.run_search(country='A', merged_keywords=['law'], search_terms='law', request_id='req')
        assert module._branch.get() is None
        return result, events[-1]['result_hash']
    assert asyncio.run(run('original')) == asyncio.run(run('direct'))

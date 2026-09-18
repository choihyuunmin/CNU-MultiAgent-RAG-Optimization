import asyncio

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

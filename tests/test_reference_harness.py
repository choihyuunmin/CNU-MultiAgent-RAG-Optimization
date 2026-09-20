import asyncio
from dataclasses import replace
import json

import pytest

from cnu_rag_optimization.reference_harness import (
    Calibration, CalibrationPair, ReferenceContract, ReferenceHarness,
    calibrate, compile_references,
)


CONTRACT = ReferenceContract('model/prompt/schema-v1', ('payload', 'records'), 'key', ('answer', 'refs'))
READY = Calibration(CONTRACT.identity, True, 12, .1, 0, 'test_fixture')


def source(prefix='a'):
    return json.dumps({'payload': {'records': [
        {'key': prefix + '-reference-100', 'text': 'literal  spaces\n"quote"'},
        {'key': prefix + '-reference-200', 'text': '문서 ⚖', 'nested': {'key': 'untouched'}},
    ]}, 'unknown': [False, None, 1e-5]}, ensure_ascii=True, indent=3)


def valid(output, original):
    return all(x in {r['key'] for r in original['payload']['records']}
               for x in output['answer']['refs'])


def test_generic_paths_preserve_bytes_values_unicode_and_restore_output_layout():
    raw = source()
    plan = compile_references(raw, CONTRACT)
    assert plan.enabled
    assert plan.encoded == raw.replace('"a-reference-100"', '"0"').replace('"a-reference-200"', '"1"')
    output = ' { "answer" : { "refs" : ["1", "0"] }, "ok":true } '
    assert plan.restore(output) == output.replace('"1"', '"a-reference-200"').replace('"0"', '"a-reference-100"')
    assert plan.restore('{"answer":{"refs":[]}}') == '{"answer":{"refs":[]}}'
    with pytest.raises(ValueError):
        plan.restore('{"answer":{"refs":["unknown"]}}')


@pytest.mark.parametrize('raw', ['{"payload":0}', '{"x":1,"x":2}', '{"x":NaN}', 'null', '[]'])
def test_unknown_inputs_bypass_without_modification(raw):
    plan = compile_references(raw, CONTRACT)
    assert not plan.enabled and plan.encoded == raw


def test_escape_analysis_prevents_cross_field_inconsistent_references():
    assert not compile_references(source(), CONTRACT, protected=('user cites a-reference-100',)).enabled
    raw = source().replace('untouched', 'a-reference-100')
    assert compile_references(raw, CONTRACT).reason == 'reference_escapes_contract'
    raw = json.dumps({'payload': {'records': [{'key': 'long\nvalue', 'text': 'uses long\nvalue'}]}})
    assert compile_references(raw, CONTRACT).reason == 'reference_escapes_contract'
    raw = source().replace('a-reference-200', 'a-reference-100')
    assert compile_references(raw, CONTRACT).reason == 'ambiguous_identifiers'
    assert not compile_references(source(), replace(CONTRACT, read_only=False)).enabled


def test_request_isolation_parallel_and_iterative_callbacks():
    async def scenario():
        harness = ReferenceHarness(READY)
        async def invoke(raw):
            await asyncio.sleep(0)
            ids = [r['key'] for r in json.loads(raw)['payload']['records']]
            return json.dumps({'answer': {'refs': ids}})
        async def one(i):
            # Reusing one harness across concurrent workflows and loop iterations
            # cannot mix the mappings of different invocation scopes.
            for step in range(3):
                prefix = f'case-{i}-step-{step}'
                result = await harness.run(source(prefix), CONTRACT, invoke=invoke, validate=valid)
                assert json.loads(result.output)['answer']['refs'] == [prefix+'-reference-100', prefix+'-reference-200']
        await asyncio.gather(*(one(i) for i in range(24)))
    asyncio.run(scenario())


def test_one_fallback_preserves_original_arguments_and_counts_whole_latency():
    async def scenario():
        calls = []
        async def invoke(raw):
            calls.append(raw)
            return ('{"answer":{"refs":["bad"]}}' if len(calls) == 1
                    else '{"answer":{"refs":["a-reference-100"]}}')
        result = await ReferenceHarness(READY).run(source(), CONTRACT, invoke=invoke, validate=valid)
        assert result.fallback and calls[1] == source() and len(calls) == 2
        assert result.elapsed_s >= 0
    asyncio.run(scenario())


def test_changed_contract_and_failed_calibration_use_original_once():
    async def scenario():
        for contract, calibration in [(replace(CONTRACT, identity='v2'), READY),
                                      (CONTRACT, replace(READY, enabled=False))]:
            calls = []
            async def invoke(raw):
                calls.append(raw)
                return '{"answer":{"refs":["a-reference-100"]}}'
            result = await ReferenceHarness(calibration).run(source(), contract, invoke=invoke, validate=valid)
            assert calls == [source()] and not result.transformed
    asyncio.run(scenario())


def test_cancellation_and_network_errors_are_not_retried():
    async def scenario():
        for error in [asyncio.CancelledError, ConnectionError]:
            calls = []
            async def invoke(raw):
                calls.append(raw)
                raise error()
            with pytest.raises(error):
                await ReferenceHarness(READY).run(source(), CONTRACT, invoke=invoke, validate=valid)
            assert len(calls) == 1
    asyncio.run(scenario())


def test_empirical_gate_rejects_quality_loss_and_no_latency_gain():
    good = [CalibrationPair(str(i), 10+i/10, 7+i/10, 0, 0) for i in range(12)]
    assert calibrate('v1', good).enabled
    assert not calibrate('v1', good[:3]).enabled
    assert not calibrate('v1', [replace(x, candidate_loss=.2) for x in good]).enabled
    assert not calibrate('v1', [replace(x, candidate_s=x.original_s) for x in good]).enabled
    with pytest.raises(ValueError, match='repeated'):
        calibrate('v1', good+good)
    with pytest.raises(ValueError):
        calibrate('v1', [replace(good[0], candidate_s=float('nan'))])

import asyncio
import copy
from pathlib import Path
import sys

import httpx

from cnu_rag_optimization import WorkflowAdapter

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from moleg_adapter_serving import ServingMeter


def test_tokenization_preserves_model_inputs_and_metrics_require_a_delta():
    async def run():
        adapter = WorkflowAdapter()
        meter = ServingMeter(adapter, {'roles':['orchestrator'], 'output_reserve_tokens':512},
            routes={'orchestrator':{'base':'http://fixture', 'model':'actual-model', 'key':'fixture-key'}})
        bodies = []
        async def transport(request):
            if request.url.path == '/tokenize':
                import json
                bodies.append(json.loads(request.content))
                return httpx.Response(200, json={'count':123})
            return httpx.Response(200, text='vllm:kv_cache_usage_perc 0.2\n'
                'vllm:num_requests_waiting 0\nvllm:num_preemptions_total 10\n')
        meter.client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        payload = {'messages':[{'role':'user','content':'private'}], 'tools':[{'name':'tool'}],
                   'max_tokens':999, 'extra_body':{'chat_template_kwargs':{'enable_thinking':False}}}
        unchanged = copy.deepcopy(payload)
        with adapter.request() as trace:
            estimate = await meter.estimate('orchestrator', payload)
        assert estimate.input_tokens == 123 and estimate.output_tokens == 999
        assert payload == unchanged
        assert bodies[0]['messages'] == payload['messages']
        assert bodies[0]['chat_template_kwargs'] == {'enable_thinking':False}
        assert 'max_tokens' not in bodies[0]
        assert 'orchestrator' not in adapter.pressure
        meter.last_metrics.clear()
        await meter.refresh_pressure('orchestrator')
        assert adapter.pressure['orchestrator'].preemptions_delta == 0
        assert 'private' not in str(trace.spans) and 'fixture-key' not in str(trace.spans)
        await meter.close()
    asyncio.run(run())


def test_tokenizer_failure_returns_unknown_cost_and_never_rewrites_request():
    async def run():
        a = WorkflowAdapter()
        meter = ServingMeter(a, {}, routes={'orchestrator':{'base':'http://fixture','model':'m','key':'x'}})
        async def transport(request):
            return httpx.Response(503)
        meter.client = httpx.AsyncClient(transport=httpx.MockTransport(transport))
        assert await meter.estimate('orchestrator', {'messages':[]}) is None
        assert meter.counts['tokenize_errors'] == 1 and meter.counts['metrics_errors'] == 1
        assert not a.pressure
        await meter.close()
    asyncio.run(run())

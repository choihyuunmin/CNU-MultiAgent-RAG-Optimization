"""Isolated application process whose orchestrator calls can be routed to a
different serving instance of the *same* model (direct OpenAI-compatible
endpoint) while everything else keeps the fixed 'baseline' behaviour of the
2026-09-05 paper harness. Set MOLEG_ORCH_BASE to a base URL to enable; unset
keeps the original proxy path. Never changes deployed code or model servers.
"""
from __future__ import annotations
import hashlib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import moleg_paper_runtime as base


def install():
    base.install('baseline')
    orch_base = os.environ.get('MOLEG_ORCH_BASE')
    if not orch_base:
        return
    from dotenv import dotenv_values
    from openai import AsyncOpenAI
    import infra.llm.client as llm
    serving = dotenv_values(os.environ['MOLEG_SERVING_ENV'])
    model_name = os.environ.get('MOLEG_ORCH_MODEL', 'google/gemma-4-31B-it')
    client = AsyncOpenAI(base_url=orch_base, api_key=serving['VLLM_API_KEY'], timeout=120, max_retries=0)
    installed_chat = llm.acompletion_via_proxy

    async def chat(**kwargs):
        model = str(kwargs.get('model') or llm.MASTER_MODEL)
        if model != 'orchestrator':
            return await installed_chat(**kwargs)
        payload = {'model': model_name, 'messages': kwargs.get('messages', []),
                   'temperature': kwargs.get('temperature', 0)}
        for k in ('tools', 'tool_choice', 'extra_body', 'response_format'):
            if kwargs.get(k) is not None:
                payload[k] = kwargs[k]
        if kwargs.get('max_tokens') is not None:
            payload['max_tokens'] = int(kwargs['max_tokens'])
        start = time.perf_counter()
        try:
            response = await client.chat.completions.create(**payload)
        except Exception as e:
            base.trace_add('errors', component='llm', error=type(e).__name__)
            raise
        usage = response.usage.model_dump() if response.usage else {}
        base.trace_add('llm', model=model, elapsed_s=time.perf_counter() - start, stream=False, usage=usage,
                       route='direct:' + orch_base,
                       input_sha256=hashlib.sha256(json.dumps(kwargs, sort_keys=True, default=str).encode()).hexdigest(),
                       output_sha256=hashlib.sha256(json.dumps(response.choices[0].message.model_dump(), sort_keys=True, default=str).encode()).hexdigest(),
                       finish_reason=response.choices[0].finish_reason)
        return response

    llm.acompletion_via_proxy = chat
    for module in list(sys.modules.values()):
        if module is None or not getattr(module, '__name__', '').startswith(('agent.', 'infra.', 'core.')):
            continue
        if getattr(module, 'acompletion_via_proxy', None) is installed_chat:
            module.acompletion_via_proxy = chat


if __name__ == '__main__':
    install()
    import uvicorn
    from main_server import app
    uvicorn.run(app, host='127.0.0.1', port=int(os.environ['MOLEG_STUDY_PORT']), log_level='error')

import asyncio
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import moleg_model_transport as transport


def test_error_chain_keeps_network_reason_but_never_exception_messages():
    network = OSError(104, 'private URL and credential in message')
    wrapped = RuntimeError('private SDK request')
    wrapped.__cause__ = network
    network.__context__ = wrapped  # Defensive cycle guard.
    chain = transport.transport_error_chain(wrapped)
    assert chain == [{'type': 'RuntimeError', 'module': 'builtins'},
                     {'type': 'ConnectionResetError', 'module': 'builtins', 'errno': 104}]
    assert 'private' not in str(chain)


@pytest.mark.parametrize('endpoint', [
    'https://127.0.0.1:28160/v1', 'http://example.com:28160/v1',
    'http://user:secret@127.0.0.1:28160/v1', 'http://127.0.0.1/v1',
    'http://127.0.0.1:28160/v1?token=value', 'http://127.0.0.1:28160/other',
])
def test_replica_route_requires_explicit_loopback(endpoint, monkeypatch):
    monkeypatch.setattr(transport, '_worker_experiment_base', None)
    with pytest.raises(ValueError):
        transport.configure_worker_experiment(endpoint)
    assert transport._worker_experiment_base is None


def test_worker_only_route_retains_generation_options_and_stream_close(monkeypatch):
    calls, closed, client_options = [], [], []

    class Stream:
        def __aiter__(self):
            return self

        async def __anext__(self):
            raise StopAsyncIteration

        async def close(self):
            closed.append(True)

    def client(label):
        async def create(*, model, messages, temperature, stream, tools=None,
                         tool_choice=None, max_tokens=None, reasoning_effort=None,
                         extra_body=None, seed=None, stop=None):
            calls.append((label, dict(model=model, messages=messages, temperature=temperature,
                stream=stream, tools=tools, tool_choice=tool_choice, max_tokens=max_tokens,
                reasoning_effort=reasoning_effort, extra_body=extra_body, seed=seed, stop=stop)))
            return Stream() if stream else 'response'
        return SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))

    infra, package, llm = ModuleType('infra'), ModuleType('infra.llm'), ModuleType('infra.llm.client')
    infra.llm, package.client = package, llm
    llm.MASTER_MODEL = 'master'
    llm.get_model_for_role = lambda role: {'tool': 'retrieval', 'tool_synthesis': 'worker'}[role]
    llm._get_openai_client = lambda: client('proxy')
    llm.acompletion_via_proxy = llm.acompletion_stream_via_proxy = lambda: None
    sdk = ModuleType('openai')
    def constructor(**kwargs):
        client_options.append(kwargs)
        return client('replica')
    sdk.AsyncOpenAI = constructor
    for module in (infra, package, llm, sdk):
        monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(transport, '_worker_experiment_base', None)
    transport.configure_worker_experiment('http://127.0.0.1:28160/v1/')
    transport.install_standard_transport()

    async def run():
        for model in ['retrieval', 'worker', 'master', 'comparison']:
            await llm.acompletion_via_proxy(model=model, messages=[{'role': 'user', 'content': 'test'}],
                temperature=0.4, tools=[{'type': 'function'}], tool_choice='auto', max_tokens=4321,
                reasoning_effort='low', seed=19, stop=['END'])
        async for _ in llm.acompletion_stream_via_proxy(model='worker', messages=[]):
            pass
    asyncio.run(run())
    assert [label for label, _ in calls] == ['replica', 'replica', 'proxy', 'proxy', 'replica']
    assert [payload['model'] for _, payload in calls] == [
        'openai/gpt-oss-20b', 'openai/gpt-oss-20b', 'master', 'comparison', 'openai/gpt-oss-20b']
    for _, payload in calls[:4]:
        assert payload['max_tokens'] == 4321 and payload['seed'] == 19
        assert payload['stop'] == ['END'] and payload['reasoning_effort'] == 'low'
        assert payload['temperature'] == .4 and payload['tool_choice'] == 'auto'
        assert payload['tools'] == [{'type': 'function'}]
    assert calls[0][1]['extra_body'] is None
    assert calls[2][1]['extra_body'] == {'allowed_openai_params': ['reasoning_effort']}
    assert client_options[0]['max_retries'] == 0
    assert closed == [True]

"""Isolated integration: reranker auth, explicit stage failures, model dispatch."""
from contextlib import ExitStack, nullcontext
from contextvars import ContextVar
from functools import wraps
import hashlib
import json
import math
from pathlib import Path
from urllib.parse import urlsplit

from cnu_rag_optimization.continuation import ContinuationWindow


def origin(url):
    parsed = urlsplit(url)
    if parsed.scheme not in {'http', 'https'} or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError('an explicit HTTP origin is required')
    return parsed.scheme, parsed.hostname.lower(), parsed.port or (443 if parsed.scheme == 'https' else 80)


def rerank_scores_valid(response, count):
    try:
        values = response.json()['results']
        indices = [item['index'] for item in values]
        return (len(indices) == count and all(type(i) is int for i in indices)
                and set(indices) == set(range(count))
                and all(math.isfinite(float(item['relevance_score'])) for item in values))
    except (KeyError, TypeError, ValueError):
        return False


def install(config_path):
    import infra.llm.client as llm
    import tools.search_tool.engine as engine
    import agent.validation_agent as preparation
    import api.controller.generate_controller as controller
    from moleg_paper_runtime import bootstrap, trace_add
    from moleg_workflow_adapter import replace_aliases
    settings = json.loads(Path(config_path).read_text())
    if set(settings)-{'fingerprints', 'rerank_serving_auth', 'rerank_origin', 'model_slots', 'ordering', 'aging_s'}:
        raise ValueError('unknown continuation configuration')
    for module in [engine, preparation, llm, controller]:
        if settings['fingerprints'].get(module.__name__) != hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest():
            raise ValueError('dependency integration source mismatch')
    if settings.get('rerank_serving_auth'):
        if origin(engine.RERANK_API_URL) != origin(settings['rerank_origin']):
            raise ValueError('reranker origin differs from operator binding')
        _, serving, _, _ = bootstrap()
        # Restore the app's intended reranking, without storing the key in config.
        engine.RERANK_API_KEY = serving['VLLM_API_KEY']
        before_post = engine.requests.post

        def post(url, *args, **kwargs):
            if origin(url) != origin(settings['rerank_origin']):
                raise ValueError('reranker credential cannot be sent to another origin')
            kwargs['allow_redirects'] = False
            try:
                response = before_post(url, *args, **kwargs)
            except Exception as exc:
                trace_add('dependency_failure', component='rerank', error_type=type(exc).__name__)
                raise
            if response.status_code != 200:
                trace_add('dependency_failure', component='rerank', status=response.status_code)
            elif not rerank_scores_valid(response, len(kwargs['json']['documents'])):
                trace_add('dependency_failure', component='rerank', reason='invalid_score_response')
            return response
        engine.requests.post = post
        probe = post(engine.RERANK_API_URL,
                     headers={'Authorization': 'Bearer '+engine.RERANK_API_KEY},
                     json={'model': engine.RERANK_MODEL_NAME, 'query': '교육',
                           'documents': ['교육 관련 자료', '날씨 관련 자료']}, timeout=10)
        if probe.status_code != 200 or not rerank_scores_valid(probe, 2):
            raise RuntimeError('reranker preflight did not return two scores')

    before_prep = preparation.run_preparation

    @wraps(before_prep)
    async def prepare(**kwargs):
        result = await before_prep(**kwargs)
        if result.get('early_exit'):
            # In this fingerprinted function early_exit is the infrastructure
            # exception branch, before the app rewrites it as a friendly answer.
            trace_add('dependency_failure', component='preparation', reason='backend_failure')
        return result
    preparation.run_preparation = prepare
    replace_aliases(before_prep, prepare, 'run_preparation')

    windows = {model: ContinuationWindow(capacity, ordering=settings.get('ordering','continuation'),
                   aging_s=settings.get('aging_s',10),
                   on_event=lambda event, model=model: trace_add('dispatch', model=model, **event))
               for model, capacity in settings.get('model_slots',{}).items()}
    before_execute = controller.execute_generate

    @wraps(before_execute)
    async def execute(**kwargs):
        with ExitStack() as stack:
            for window in windows.values():
                stack.enter_context(window.flow())
            return await before_execute(**kwargs)
    controller.execute_generate = execute

    before_call = llm.call_llm
    outer_credit = ContextVar('moleg_outer_model_credit', default=None)

    @wraps(before_call)
    async def call(*args, **kwargs):
        model = llm.get_model_for_role(kwargs.get('role', args[4] if len(args)>4 else 'master'))
        window = windows.get(model)
        # Queue before call_llm starts its per-call 120-second timer.
        async with window.slot() if window else nullcontext():
            token = outer_credit.set(model if window else None)
            try:
                return await before_call(*args, **kwargs)
            except Exception as exc:
                trace_add('model_attempt_failure', model=model, error_type=type(exc).__name__)
                raise
            finally:
                outer_credit.reset(token)
    llm.call_llm = call
    replace_aliases(before_call, call, 'call_llm')

    before_chat = llm.acompletion_via_proxy

    @wraps(before_chat)
    async def chat(**kwargs):
        model = str(kwargs.get('model') or llm.MASTER_MODEL)
        window = windows.get(model)
        # call_llm's wait_for creates a child task. It already owns the outer
        # slot, so the outer wrapper marks that invocation explicitly below.
        if outer_credit.get() == model:
            return await before_chat(**kwargs)
        async with window.slot() if window else nullcontext():
            return await before_chat(**kwargs)

    llm.acompletion_via_proxy = chat
    replace_aliases(before_chat, chat, 'acompletion_via_proxy')
    before_stream = llm.acompletion_stream_via_proxy

    @wraps(before_stream)
    async def stream(**kwargs):
        model = str(kwargs.get('model') or llm.MASTER_MODEL)
        window = windows.get(model)
        async with window.slot() if window and outer_credit.get()!=model else nullcontext():
            source = before_stream(**kwargs)
            try:
                async for chunk in source:
                    yield chunk
            finally:
                await source.aclose()
    llm.acompletion_stream_via_proxy = stream
    replace_aliases(before_stream, stream, 'acompletion_stream_via_proxy')
    return {'windows': windows, 'rerank_auth_repaired': bool(settings.get('rerank_serving_auth'))}

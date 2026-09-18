"""Ordinary SDK calls with complete generation options and explicit stream close."""
import inspect
import time
from urllib.parse import urlsplit

_worker_experiment_base = None


def transport_error_chain(exc):
    """Keep exception types/errno for diagnosis without messages, URLs or keys."""
    chain, seen = [], set()
    while exc is not None and id(exc) not in seen and len(chain) < 8:
        seen.add(id(exc))
        item = {'type': type(exc).__name__, 'module': type(exc).__module__}
        number = getattr(exc, 'errno', None)
        if type(number) is int:
            item['errno'] = number
        chain.append(item)
        exc = exc.__cause__ if exc.__cause__ is not None else exc.__context__
    return chain


def configure_worker_experiment(base):
    """Opt-in route for an owned loopback replica of the same worker model."""
    global _worker_experiment_base
    parsed = urlsplit(base)
    if (parsed.scheme != 'http' or parsed.hostname != '127.0.0.1' or not parsed.port
            or parsed.username or parsed.password or parsed.query or parsed.fragment
            or parsed.path.rstrip('/') != '/v1'):
        raise ValueError('worker experiment requires an explicit loopback /v1 endpoint')
    _worker_experiment_base = base.rstrip('/')


def chat_payload(kwargs, *, model, streaming, create, proxy=False):
    # Derive supported parameters from the installed SDK, rather than keeping an
    # incomplete allowlist that silently drops effort, seed, stop or token limits.
    accepted = inspect.signature(create).parameters
    payload = {k: v for k, v in kwargs.items() if k in accepted}
    payload.update(model=model, messages=kwargs.get("messages", []),
                   temperature=kwargs.get("temperature", 0), stream=streaming)
    if proxy:
        extra = dict(payload.get("extra_body") or {})
        if "reasoning_effort" in payload or "reasoning_effort" in extra:
            # The installed LiteLLM version does not recognize gpt-oss as a
            # reasoning model. Explicitly allow this supported upstream option
            # per request, rather than dropping it or changing the proxy config.
            allowed = extra.get("allowed_openai_params", [])
            if not isinstance(allowed, list):
                raise ValueError("allowed_openai_params must be a list")
            extra["allowed_openai_params"] = list(dict.fromkeys([*allowed, "reasoning_effort"]))
            payload["extra_body"] = extra
    return payload


def install_standard_transport():
    import infra.llm.client as llm
    from moleg_workflow_adapter import replace_aliases
    worker = None
    worker_aliases = set()
    if _worker_experiment_base:
        from openai import AsyncOpenAI
        worker = AsyncOpenAI(base_url=_worker_experiment_base, api_key='isolated-experiment',
                             timeout=120, max_retries=0)
        worker_aliases = {llm.get_model_for_role(role) for role in ('tool', 'tool_synthesis')}

    def destination(kwargs):
        model = str(kwargs.get('model') or llm.MASTER_MODEL).strip()
        routed = worker is not None and model in worker_aliases
        client = worker if routed else llm._get_openai_client()
        return client.chat.completions.create, 'openai/gpt-oss-20b' if routed else model, not routed

    async def chat(**kwargs):
        create, model, proxy = destination(kwargs)
        payload = chat_payload(kwargs, model=model, streaming=False, create=create, proxy=proxy)
        started = time.perf_counter()
        try:
            return await create(**payload)
        finally:
            record_time(started)

    async def stream(**kwargs):
        create, model, proxy = destination(kwargs)
        payload = chat_payload(kwargs, model=model, streaming=True, create=create, proxy=proxy)
        source = None
        started = time.perf_counter()
        try:
            source = await create(**payload)
            async for chunk in source:
                yield chunk
        finally:
            try:
                if source is not None:
                    await source.close()
            finally:
                record_time(started)

    def record_time(started):
        # Preserve the app's request-local cumulative model timing, when present.
        callback = getattr(llm, "add_llm_time", None)
        if callback:
            callback((time.perf_counter() - started) * 1000)

    for name, replacement in [("acompletion_via_proxy", chat),
                               ("acompletion_stream_via_proxy", stream)]:
        original = getattr(llm, name)
        setattr(llm, name, replacement)
        replace_aliases(original, replacement, name)

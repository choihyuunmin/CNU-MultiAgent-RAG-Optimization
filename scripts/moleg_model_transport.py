"""Ordinary SDK calls with complete generation options and explicit stream close."""
import inspect
import time


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

    async def chat(**kwargs):
        create = llm._get_openai_client().chat.completions.create
        payload = chat_payload(kwargs, model=str(kwargs.get("model") or llm.MASTER_MODEL).strip(),
                               streaming=False, create=create, proxy=True)
        started = time.perf_counter()
        try:
            return await create(**payload)
        finally:
            record_time(started)

    async def stream(**kwargs):
        create = llm._get_openai_client().chat.completions.create
        payload = chat_payload(kwargs, model=str(kwargs.get("model") or llm.MASTER_MODEL).strip(),
                               streaming=True, create=create, proxy=True)
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

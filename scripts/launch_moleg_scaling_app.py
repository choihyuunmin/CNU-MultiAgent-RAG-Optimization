"""Launch an isolated app; opt-in harnesses may control model dispatch.

Run in the application's existing Python environment. All credentials come
from operator files. Loopback binding is mandatory. Never edits deployed code,
restarts a model, or changes the production application's limit.
"""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def install_immediate_emission():
    from core.query_loop import _helpers as helpers
    original = helpers.emit_streaming_delta

    async def emit(queue, delta, request_id="-", **kwargs):
        # Keep the existing emitter's marking, errors and cancellation behavior.
        # Only pacing and subdivision change. Text and event order are retained.
        return await original(queue, delta, request_id, chunk_chars=1000, delay_s=0)

    helpers.emit_streaming_delta = emit
    for module in list(sys.modules.values()):
        if module is None or not getattr(module, "__name__", "").startswith(("core.", "api.", "agent.")):
            continue
        if getattr(module, "emit_streaming_delta", None) is original:
            module.emit_streaming_delta = emit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--app-root", type=Path, required=True)
    parser.add_argument("--app-env", type=Path, required=True)
    parser.add_argument("--serving-env", type=Path, required=True)
    parser.add_argument("--proxy-config", type=Path, required=True)
    parser.add_argument("--trace", type=Path, required=True, help="private output: may contain visible text")
    parser.add_argument("--port", type=int, default=28220)
    parser.add_argument("--emission", choices=["original", "immediate"], default="immediate")
    parser.add_argument("--adapter-config", type=Path,
                        default=Path(__file__).resolve().parents[1] / "integrations/2025-moleg-search/workflow-reasoning.json",
                        help="observable workflow and reasoning policy configuration")
    parser.add_argument("--adapter-trace", type=Path, help="new prompt-free workflow trace file")
    parser.add_argument("--base-profile", choices=["baseline", "speed"], default="baseline",
                        help="speed combines earlier application methods; requires separate quality evaluation")
    parser.add_argument("--program-overlap", action="store_true",
                        help="attach the ProgramHarness ready-successor overlay to execute_search; "
                             "moves start time of dependency-ready search work earlier without "
                             "changing models, prompts, retrieval arguments, or the return shape")
    parser.add_argument("--program-fingerprint",
                        help="expected SHA-256 of law_search/nodes.py; overlay fails closed on mismatch")
    parser.add_argument("--no-ontology", action="store_true",
                        help="skip the ontology-scoped search branch (loader returns None); "
                             "retrieval falls back to the existing hybrid path, unchanged otherwise")
    parser.add_argument("--structured-harness", type=Path,
                        help="opt-in fingerprinted context/structured decoding harness configuration")
    parser.add_argument("--continuation-harness", type=Path,
                        help="opt-in dependency health and continuation dispatch configuration")
    parser.add_argument("--worker-experiment-base",
                        help="owned loopback /v1 replica of the same gpt-oss worker, for serving A/B only")
    args = parser.parse_args()
    if args.adapter_trace is None:
        args.adapter_trace = args.trace.with_suffix(".workflow.jsonl")
    if args.adapter_trace.resolve() == args.trace.resolve():
        parser.error("model and workflow traces must use different paths")
    from moleg_workflow_adapter import load_adapter
    adapter, estimates, hooks = load_adapter(args.adapter_config)
    if args.adapter_trace.exists():
        parser.error("use a new adapter trace path")
    for path in (args.app_env, args.serving_env, args.proxy_config):
        if not path.is_file():
            parser.error("operator configuration file missing")
    if not (args.app_root / "src/main_server.py").is_file():
        parser.error("application source root missing")
    if args.trace.exists():
        parser.error("use a new private trace path")
    args.trace.parent.mkdir(parents=True, exist_ok=True)
    args.trace.touch(mode=0o600)
    from dotenv import load_dotenv
    load_dotenv(args.app_env, override=True)
    os.environ.update({"MOLEG_RAG_ROOT": str(args.app_root.resolve()),
        "MOLEG_SERVING_ENV": str(args.serving_env.resolve()),
        "MOLEG_PROXY_CONFIG": str(args.proxy_config.resolve()),
        "MOLEG_TRACE_PATH": str(args.trace.resolve()),
        "MOLEG_STUDY_PROFILE": args.base_profile})
    import moleg_paper_runtime
    if args.worker_experiment_base:
        from moleg_model_transport import configure_worker_experiment
        configure_worker_experiment(args.worker_experiment_base)
    # Optional speculative calls remain off in the normal execution path.
    moleg_paper_runtime.install(args.base_profile, allow_preparation_overlap=False)
    harness_close = None
    if args.structured_harness:
        from moleg_structured_harness import install as install_structured
        harness_close = install_structured(args.structured_harness,
                                          args.trace.with_suffix(".capture.jsonl"))
    from moleg_workflow_adapter import install, TraceASGI
    install(adapter, estimates, hooks)
    continuation = None
    if args.continuation_harness:
        from moleg_continuation_harness import install as install_continuation
        continuation = install_continuation(args.continuation_harness)
    args.adapter_trace.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(args.adapter_trace, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    adapter_sink = os.fdopen(fd, "w", buffering=1)
    if args.emission == "immediate":
        install_immediate_emission()
    from main_server import app
    if harness_close:
        from contextlib import asynccontextmanager
        original_lifespan = app.router.lifespan_context
        @asynccontextmanager
        async def lifespan(application):
            try:
                async with original_lifespan(application) as state:
                    yield state
            finally:
                await harness_close()
        app.router.lifespan_context = lifespan
    from api.concurrency import generate_queue
    from moleg_unrestricted import remove_admission_limits
    remove_admission_limits(app, generate_queue)
    if args.no_ontology:
        # Disable only the ontology-scoped search branch: the loader returns None,
        # so execute_search takes its existing "no ontology" fallback (hybrid
        # results). Models, prompts, hybrid retrieval, and return shape unchanged.
        import core.query_loop.law_search.nodes as _ls_nodes
        _ls_nodes._get_ontology_searcher = lambda *a, **k: None
        print({"no_ontology": True}, flush=True)
    program_harness = None
    program_harness_sink = None
    if args.program_overlap:
        from moleg_program_overlay import apply_program_overlay
        program_harness_sink = open(args.adapter_trace.with_suffix(".harness.jsonl"), "w", buffering=1)
        program_harness, _overlay = apply_program_overlay(
            fingerprint=args.program_fingerprint, event_sink=program_harness_sink,
            harness_sink=program_harness_sink)
    @app.get("/__scaling_state", include_in_schema=False)
    async def scaling_state():
        # This process is loopback-only. No request IDs or application data.
        import resource
        usage = resource.getrusage(resource.RUSAGE_SELF)
        return {"max_pipelines": None, "max_http": None,
                "active": generate_queue._global.active,
                "queued": generate_queue._global.queued, "emission": args.emission,
                "process_maxrss_kib": usage.ru_maxrss,
                "process_cpu_s": usage.ru_utime + usage.ru_stime,
                "base_profile": args.base_profile,
                "legacy_preparation_overlap": False,
                "program_overlap": args.program_overlap,
                "no_ontology": args.no_ontology,
                "worker_experiment_base": args.worker_experiment_base,
                "continuation": {name: window.snapshot() for name, window in continuation['windows'].items()}
                    if continuation else None,
                "rerank_auth_repaired": continuation['rerank_auth_repaired'] if continuation else False,
                "adapter": adapter.snapshot(),
                "serving_meter": adapter.serving_meter.snapshot()
                    if hasattr(adapter, "serving_meter") else None}
    print({"scope": "isolated loopback application", "max_pipelines": None,
           "max_http": None, "emission": args.emission, "port": args.port}, flush=True)
    import uvicorn
    try:
        uvicorn.run(TraceASGI(app, adapter, adapter_sink),
                    host="127.0.0.1", port=args.port, log_level="error")
    finally:
        adapter_sink.close()
        if program_harness_sink is not None:
            program_harness_sink.close()


if __name__ == "__main__":
    main()

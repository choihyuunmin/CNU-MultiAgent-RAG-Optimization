"""Build isolated program-harness trial from frozen application bundle."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

from build_capability_scale import build
from program_harness_attachment import transform


def build_trial(previous, remote_source, root, name):
    source = remote_source.read_text()
    function = next(node for node in ast.parse(source).body
                    if isinstance(node, ast.AsyncFunctionDef) and node.name == "execute_search")
    transformed = transform(ast.get_source_segment(source, function))
    compile(transformed, "program_harness_check", "exec")
    fingerprint = hashlib.sha256(remote_source.read_bytes()).hexdigest()
    artifact = build(previous, (root / "scripts/run_capability_scale.py").read_text(), name)
    data = artifact["items"][0]["data"]
    data["program_harness.py"] = (root / "src/cnu_rag_optimization/program_harness.py").read_text()
    data["program_harness_attachment.py"] = (root / "scripts/program_harness_attachment.py").read_text()
    anchor = '    if args.mode == "capability":\n        search.run_search = wrapped_search\n'
    if data["serve_trial.py"].count(anchor) != 1:
        raise ValueError("unexpected frozen application glue")
    injection = '''    import inspect
    from program_harness import ProgramHarness
    from program_harness_attachment import transform
    nodes = importlib.import_module("core.query_loop.law_search.nodes")
    if hashlib.sha256(Path(nodes.__file__).read_bytes()).hexdigest() != FINGERPRINT:
        raise RuntimeError("remote program source mismatch")
    if args.mode == "capability":
        harness = ProgramHarness(lambda row: print("CNU_PROGRAM_HARNESS_V1 " + json.dumps(row), flush=True))
        namespace = dict(nodes.execute_search.__globals__)
        namespace["_pipeline"] = harness
        exec(compile(transform(inspect.getsource(nodes.execute_search)), nodes.__file__, "exec"), namespace)
        nodes.execute_search = harness.wrap(namespace["execute_search"])
        graph_module = importlib.import_module("core.query_loop.law_search.graph")
        builder = importlib.import_module("core.query_loop.builder")
        if builder._moleg_graph is not None:
            raise RuntimeError("graph already compiled before program harness attachment")
        graph_module.execute_search = nodes.execute_search
        print("CNU_PROGRAM_ATTACHMENT " + json.dumps({"enabled": True,
            "nodes": ["ontology_search", "scenario_followup"], "concurrency_limit": None,
            "model_or_payload_changed": False}), flush=True)
'''.replace("FINGERPRINT", repr(fingerprint))
    data["serve_trial.py"] = data["serve_trial.py"].replace(anchor, injection).replace(
        '"scope": "search dispatch only", "call_elision": args.mode == "capability"',
        '"scope": "program dependency harness only", "call_elision": False')
    compile(data["serve_trial.py"], "serve_trial.py", "exec")
    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--previous", type=Path, required=True)
    parser.add_argument("--remote-source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    artifact = build_trial(json.loads(args.previous.read_text()), args.remote_source, root, args.name)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(artifact, stream)

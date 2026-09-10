"""CPU contract check of the combined adapter; NOT a GPU speed experiment.

Synthetic complete payloads exercise sustained users 1/2/4/8/16/32. Stored live
results are referenced separately; their improvements are never pooled with
this test or attributed to the new budget scheduler.
"""
import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from cnu_rag_optimization import (
    ModelBudget, TokenReservation, WorkflowAdapter, diagnose_trace,
    QualityEvidence, evaluate_quality_gate,
)


async def exercise(users, mode):
    adapter = WorkflowAdapter(mode=mode, budgets={"orchestrator": ModelBudget(8, 640, 512)})
    pending = iter(range(64))
    completed = []

    async def worker():
        for index in pending:
            original = {"model": "fixture-revision", "messages": [{"content": f"synthetic-{index}"}],
                        "max_tokens": 32, "tools": [], "temperature": 0,
                        "evidence": [{"id": f"fixture-{index}", "content": "unaltered fixture"}]}
            before = json.dumps(original, sort_keys=True)
            expected = hashlib.sha256(before.encode()).hexdigest()
            async def model():
                await asyncio.sleep(0)  # yield to peers; not a model-time simulation
                return {"answer": expected, "evidence": original["evidence"]}
            with adapter.request() as trace:
                with adapter.stage("preparation"):
                    result = await adapter.call("orchestrator", model,
                                                reservation=TokenReservation(48, 32, 16))
                with adapter.stage("answer"):
                    async def source():
                        yield {"delta": result["answer"][:32]}
                        await asyncio.sleep(0)
                        yield {"delta": result["answer"][32:]}
                    chunks = [c async for c in adapter.stream("orchestrator", source,
                                                   reservation=TokenReservation(48, 32, 16))]
            unchanged = (json.dumps(original, sort_keys=True) == before
                         and "".join(c["delta"] for c in chunks) == expected)
            completed.append({"fixture": index, "exact": unchanged,
                              "trace_complete": diagnose_trace(trace)["dropped_spans"] == 0})
    await asyncio.gather(*(worker() for _ in range(users)))
    idle = all(g.active == 0 and not g.pending for g in adapter.gates.values())
    assert len(completed) == 64 and all(r["exact"] and r["trace_complete"] for r in completed) and idle
    return {"users": users, "mode": mode, "requests": 64,
            "exact_fixture_outputs": sum(r["exact"] for r in completed), "credits_released": idle}


async def run():
    rows = []
    for users in [1, 2, 4, 8, 16, 32]:
        for mode in ["observe", "budget"]:
            rows.append(await exercise(users, mode))
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("choose a new output file")
    root = Path(__file__).resolve().parents[1]
    live_path = root / "experiments/results/moleg-scaling-isolated-20260907/validation-counterbalanced/aggregate.json"
    live = json.loads(live_path.read_text())
    historical = [{"users": g["users"], "policy": g["policy"], "n": g["n"],
                   "success": g["success"], "mean_s": g["metrics"]["elapsed_s"]["mean"]}
                  for g in live["groups"]]
    evidence = QualityEvidence(64, True, 1, 1)
    result = {"scope": "CPU synthetic payload/stream contracts; no GPU inference, no speed or accuracy claim",
              "synthetic_checks": asyncio.run(run()),
              "historical_live_results_not_new_adapter_measurements": historical,
              "historical_aggregate_sha256": hashlib.sha256(live_path.read_bytes()).hexdigest(),
              "quality_evidence_available_for_release": asdict(evidence),
              "release_gate": evaluate_quality_gate(evidence),
              "source_sha256": {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
                  for p in [Path(__file__), root / "src/cnu_rag_optimization/adaptive.py",
                            root / "src/cnu_rag_optimization/quality_gate.py",
                            root / "scripts/moleg_workflow_adapter.py",
                            root / "scripts/moleg_paper_runtime.py",
                            root / "scripts/launch_moleg_scaling_app.py"]}}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as sink:
        json.dump(result, sink, indent=2)
        sink.write("\n")
    print(json.dumps({"synthetic_requests": sum(r["requests"] for r in result["synthetic_checks"]),
                      "all_contracts_passed": True, "quality_release_allowed": result["release_gate"]["release_allowed"]}))


if __name__ == "__main__":
    main()

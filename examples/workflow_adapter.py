import asyncio
import json
import time

from cnu_rag_optimization import (
    ServingPressure, TokenReservation, WorkflowAdapter, diagnose_trace,
)


async def main():
    adapter = WorkflowAdapter(speculation=True)
    # Synthetic fresh telemetry for this example ONLY. Production integrations
    # must supply real samples and preemption counter deltas, not these values.
    adapter.pressure["orchestrator"] = ServingPressure(time.monotonic(), .20, 0, 0)
    full_input = {"model_revision": "fixture-v1", "messages": [{"role": "user", "content": "fixture"}],
                  "options": {"max_tokens": 100}, "tenant_scope": "example", "evidence": []}

    async def authoritative_input():
        # In an application this is the normal classifier/preparation dependency.
        await asyncio.sleep(0)
        return full_input

    async def invoke(payload):
        # Replace with the SAME model RPC, unchanged payload, no nested gate.
        return {"fixture_answer": "unchanged", "max_tokens": payload["options"]["max_tokens"]}

    with adapter.request() as trace:
        with adapter.stage("preparation"):
            answer, reused = await adapter.verified_overlap(
                model="orchestrator", predicted_input=full_input,
                authoritative_input=authoritative_input, invoke=invoke,
                reservation=TokenReservation(1000, 100), read_only=True)
    print(json.dumps({"answer": answer, "reused": reused, "diagnosis": diagnose_trace(trace)}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

"""Synthetic mechanism demo, not a GPU benchmark or legal-quality evaluation."""
import asyncio
import json
import statistics
import time

from cnu_rag_optimization.inference_overlap import InferenceOverlapAdapter, ReadContract


async def main():
    adapter = InferenceOverlapAdapter()
    contract = ReadContract("metadata", "fixture-v1", "test-scope", "immutable-fixture-v1",
                            b'{"ids":[1,2]}', read_only=True, deterministic=True)
    calls = {"llm": 0, "read": 0}

    async def original_llm():
        calls["llm"] += 1
        await asyncio.sleep(.15)
        return {"answer": "fixture"}

    async def original_read():
        calls["read"] += 1
        await asyncio.sleep(.06)
        return {"metadata": [1, 2]}

    observed_llm = adapter.completion(original_llm)

    async def run(overlap):
        if not overlap:
            return await observed_llm(), await original_read()
        async with adapter.prepare_read(contract, original_read) as prepared:
            answer = await observed_llm()
            # In an application, recompute from authoritative inputs/ACL/revision.
            # Do not reuse a guessed contract if selection or snapshot has changed.
            metadata = await prepared.consume(contract, original_read)
            return answer, metadata

    times = {False: [], True: []}
    reference = ({"answer": "fixture"}, {"metadata": [1, 2]})
    for repeat in range(6):
        for enabled in ((False, True) if repeat % 2 == 0 else (True, False)):
            started = time.perf_counter()
            assert await run(enabled) == reference
            times[enabled].append((time.perf_counter() - started) * 1000)
    print(json.dumps({"synthetic_only": True, "runs_per_method": 6,
        "baseline_mean_ms": statistics.mean(times[False]),
        "overlap_mean_ms": statistics.mean(times[True]),
        "fixture_outputs_equal": True, "calls": calls,
        "gpu_speedup_measured": False, "reasoning_shortened": False}, indent=2))


if __name__ == "__main__":
    asyncio.run(main())

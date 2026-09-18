"""Synthetic mechanism check, NOT a GPU or retrieval performance experiment."""
from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import statistics
import time

from cnu_rag_optimization.stream_tools import CallReady, RoundEnd, ToolBinding, execute_tool_stream


async def measure(case, mode):
    invoked, events = [], []
    async def tool(args):
        invoked.append(args["i"])
        await asyncio.sleep(case["tool_s"])
        return json.dumps(args, sort_keys=True)

    async def source():
        if case["buffered"]:
            await asyncio.sleep(case["gap_s"] * case["count"])
        for i in range(case["count"]):
            if not case["buffered"]:
                await asyncio.sleep(case["gap_s"])
            yield CallReady(i, str(i), "read", json.dumps({"i": i}))
        yield RoundEnd(case["count"])

    started = time.perf_counter()
    values = await execute_tool_stream(source(), {"read": ToolBinding(tool, True)},
                                      mode=mode, on_event=events.append)
    elapsed = time.perf_counter() - started
    end = next(e["elapsed_ms"] for e in events if e["event"] == "round_end")
    assert sorted(invoked) == list(range(case["count"]))
    assert [v.content for v in values] == [json.dumps({"i": i}) for i in range(case["count"])]
    digest = hashlib.sha256(json.dumps([asdict(v) for v in values], sort_keys=True).encode()).hexdigest()
    return {"mode": mode, "elapsed_s": elapsed, "tool_calls": len(invoked),
            "early_calls": sum(e["event"] == "tool_start" and e["elapsed_ms"] < end for e in events),
            "output_sha256": digest}


async def main(output):
    cases = {
        "four_incremental": dict(count=4, gap_s=.025, tool_s=.04, buffered=False),
        "single_call_negative_control": dict(count=1, gap_s=.025, tool_s=.04, buffered=False),
        "buffered_negative_control": dict(count=4, gap_s=.025, tool_s=.04, buffered=True),
    }
    rows = []
    modes = ["serial", "parallel", "stream"]
    for name, case in cases.items():
        for repeat in range(3):
            for mode in modes[repeat:] + modes[:repeat]:
                rows.append({"case": name, "repeat": repeat, **await measure(case, mode)})
        assert len({r["output_sha256"] for r in rows if r["case"] == name}) == 1
    means = {name: {mode: statistics.mean(r["elapsed_s"] for r in rows
               if r["case"] == name and r["mode"] == mode) for mode in modes} for name in cases}
    result = {
        "kind": "synthetic_asyncio_mechanism_check", "real_model_calls": 0,
        "network_io": False, "latencies_are_injected": True,
        "promote_to_full_experiment": False,
        "reason": "Requires real provider commit support and a live paired pilot first",
        "cases": cases, "repeats": 3, "mean_seconds": means, "rows": rows,
        "source_sha256": {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
            Path(__file__).resolve(), Path("src/cnu_rag_optimization/stream_tools.py")]},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"kind": result["kind"], "rounds": len(rows), "mean_seconds": means,
                      "promote_to_full_experiment": False}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    asyncio.run(main(args.output))

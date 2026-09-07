#!/usr/bin/env python3
"""Replay one stored real answer through the application's original SSE helper.

This is a delivery-only test, NOT a new LLM inference or legal accuracy score.
Both arms consume exactly the same previously generated answer. Nothing is sent
to the model. Preserve every token event and its ordering, not only final text.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import importlib
import json
import os
import sys
import time
from pathlib import Path


async def run(args):
    app = args.application_root.resolve()
    sys.path[:0] = [str(app / "src"), str(app / "scripts/experiment")]
    from run_experiment_bundle import _load_server_env
    os.environ.update(_load_server_env(app / ".env"))
    helpers = importlib.import_module("core.query_loop._helpers")
    matched = [json.loads(line) for line in args.responses.read_text().splitlines()
               if line.strip() and json.loads(line).get("question_id") == args.question_id]
    if len(matched) != 1:
        raise ValueError("exactly one stored real response required")
    row = matched[0]
    text = (row.get("response") or {}).get("comment")
    if row.get("status") != "ok" or not isinstance(text, str) or not text:
        raise ValueError("successful nonempty stored answer required")
    results, events = {}, {}
    for mode in ("original", "no_display_wait"):
        queue = asyncio.Queue()
        started = time.perf_counter()
        kwargs = {} if mode == "original" else {"delay_s": 0.0}
        await helpers.emit_fake_stream(queue, text, "delivery-replay", **kwargs)
        elapsed = time.perf_counter() - started
        values = []
        while not queue.empty():
            values.append(queue.get_nowait())
        events[mode] = values
        canonical = json.dumps(values, ensure_ascii=False, separators=(",", ":"))
        results[mode] = {"elapsed_seconds": elapsed, "events": len(values),
                         "event_sequence_sha256": hashlib.sha256(canonical.encode()).hexdigest(),
                         "original_text_preserved": "".join(v["delta"] for v in values) == text}
    payload = {"kind": "delivery_replay_not_llm_inference", "question_id": args.question_id,
               "source_file_sha256": hashlib.sha256(args.responses.read_bytes()).hexdigest(),
               "helper_source_sha256": hashlib.sha256(Path(helpers.__file__).read_bytes()).hexdigest(),
               "text_characters": len(text), "original_defaults": helpers.emit_fake_stream.__kwdefaults__,
               "all_events_identical": events["original"] == events["no_display_wait"], "results": results}
    if not payload["all_events_identical"] or not all(v["original_text_preserved"] for v in results.values()):
        raise RuntimeError("delivery contents changed")
    with args.output.open("x", encoding="utf-8") as output:
        output.write(json.dumps(payload, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--application-root", required=True, type=Path)
    parser.add_argument("--responses", required=True, type=Path)
    parser.add_argument("--question-id", required=True)
    parser.add_argument("--output", required=True, type=Path)
    asyncio.run(run(parser.parse_args()))


if __name__ == "__main__":
    main()

"""Isolated CPU experiment on the archived emitter, with synthetic ready text.

This measures already-complete text, NOT live model deltas. It cannot be used
to claim the same speedup for all streamed LLM responses or the deployed app.
Only three reviewed helper functions are loaded; no application/bootstrap or
credentials are imported. The source hash and exact synthetic output hashes
are retained for reproducibility.
"""
import argparse
import ast
import asyncio
import hashlib
import json
import logging
from pathlib import Path
import time
from typing import Optional


def emitter(path):
    source = path.read_text()
    tree = ast.parse(source)
    wanted = {"mark_tokens_emitted", "emit_streaming_delta", "emit_fake_stream"}
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
             and node.name in wanted]
    if {node.name for node in nodes} != wanted:
        raise ValueError("reviewed emitter helpers missing")
    scope = {"asyncio": asyncio, "Optional": Optional, "logger": logging.getLogger("replay"),
             "STREAM_CHUNK_CHARS": 5, "STREAM_DELAY_S": .02}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return scope["emit_streaming_delta"], hashlib.sha256(path.read_bytes()).hexdigest()


async def run(args):
    emit, source_sha = emitter(args.helpers)
    rows = []
    async def trial(users, length, immediate):
        text = ("가나다 ABC 123\n" * (length // 12 + 1))[:length]
        assert len(text) == length
        expected = hashlib.sha256(text.encode()).hexdigest()
        async def one():
            queue = asyncio.Queue()
            start = time.perf_counter()
            await emit(queue, text, chunk_chars=1000 if immediate else 5,
                       delay_s=0 if immediate else .02)
            elapsed = time.perf_counter() - start
            pieces = []
            while not queue.empty():
                pieces.append(queue.get_nowait()["delta"])
            return {"elapsed_s": elapsed, "chunks": len(pieces),
                    "exact": "".join(pieces) == text,
                    "sha256": hashlib.sha256("".join(pieces).encode()).hexdigest()}
        start = time.perf_counter()
        responses = await asyncio.gather(*(one() for _ in range(users)))
        row = {"users": users, "characters": length, "policy": "immediate" if immediate else "paced",
               "scope": "simultaneous CPU ready-text emission; synthetic text; no GPU/network/browser",
               "expected_sha256": expected, "wall_s": time.perf_counter() - start,
               "intentional_sleep_per_response_s": 0 if immediate else ((length + 4) // 5 - 1) * .02,
               "responses": responses}
        rows.append(row)
        print(users, length, row["policy"], round(row["wall_s"], 3), "all_exact", all(r["exact"] for r in responses), flush=True)
    for users in args.users:
        for length in args.lengths:
            for immediate in (False, True):
                await trial(users, length, immediate)
    args.output.write_text(json.dumps({"scope": "archived helper CPU replay, not live app optimization",
        "source_sha256": source_sha, "chunk_chars": 5, "delay_s": .02, "rows": rows}, indent=2) + "\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--helpers", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--users", type=int, nargs="+", default=[1, 8, 32])
    parser.add_argument("--lengths", type=int, nargs="+", default=[16749])
    args = parser.parse_args()
    if args.output.exists() or min(args.users + args.lengths) < 1:
        parser.error("use a new output path and positive user/text lengths")
    asyncio.run(run(args))


if __name__ == "__main__":
    main()

"""Counterbalanced stage replay: identical evidence, fresh model outputs.

Captures are private. Public rows contain counts, hashes, and overlap metrics.
Original capture is pseudo-reference, never expert ground truth.
"""
from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import os
from pathlib import Path
import random
import statistics
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from cnu_rag_optimization.structured_harness import (
    compact_selection_messages, compact_structured_payload,
    encode_selection_ids, decode_selection_ids, strict_json,
)


async def run(args):
    from moleg_paper_runtime import bootstrap
    from moleg_model_transport import chat_payload
    from openai import AsyncOpenAI
    from moleg_paper_metrics import id_recall
    _, serving, config, resolve = bootstrap()
    data = [json.loads(x) for x in args.capture.read_text().splitlines()]
    data = [x for x in data if x['schema'] == 'law_selection_response']
    p = next(x['litellm_params'] for x in config['model_list']
             if x['model_name'] == data[0]['kwargs']['model'])
    args.output.mkdir(mode=0o700, exist_ok=False)
    public = (args.output / 'requests.jsonl').open('x', buffering=1)
    private = (args.output / 'private-outputs.jsonl').open('x', buffering=1)
    rows = []
    async with AsyncOpenAI(base_url=resolve(p['api_base']),
            api_key=resolve(p.get('api_key')) or serving['VLLM_API_KEY'],
            max_retries=0, timeout=120) as client:
        for repeat in range(args.repeats):
            modes = ['baseline', 'minify', 'short_ids']
            if repeat % 2:
                modes.reverse()
            order = list(range(len(data)))
            random.Random(20260915 + repeat).shuffle(order)
            for mode in modes:
                sem = asyncio.Semaphore(args.concurrency)
                async def one(index):
                    async with sem:
                        case = data[index]
                        payload = chat_payload(case['kwargs'], model=p['model'][7:],
                            streaming=False, create=client.chat.completions.create)
                        mapping = {}
                        if mode == 'minify':
                            payload['messages'] = compact_selection_messages(payload['messages'])
                        elif mode == 'short_ids':
                            payload['messages'], mapping = encode_selection_ids(payload['messages'], min_candidates=args.min_candidates)
                        started = time.perf_counter()
                        row = {'case_index': index, 'mode': mode, 'repeat': repeat,
                               'short_ids': len(mapping), 'ok': False}
                        try:
                            out = await client.chat.completions.create(**payload)
                            text = decode_selection_ids(out.choices[0].message.content or '', mapping)
                            parsed, reference = strict_json(text), strict_json(case['output'])
                            ids, ref = parsed.get('selected_ids', []), reference.get('selected_ids', [])
                            row.update(ok=True, usage=out.usage.model_dump() if out.usage else {},
                                exact=parsed == reference, recall=id_recall(ref, ids),
                                precision=id_recall(ids, ref), selected_count=len(ids),
                                reference_count=len(ref), output_sha256=hashlib.sha256(text.encode()).hexdigest())
                            private.write(json.dumps(dict(row, output=text), ensure_ascii=False) + '\n')
                        except Exception as exc:
                            row['error_type'] = type(exc).__name__
                        row['elapsed_s'] = time.perf_counter() - started
                        rows.append(row)
                        public.write(json.dumps(row) + '\n')
                await asyncio.gather(*(one(i) for i in order))
                subset = [r for r in rows if r['mode'] == mode and r['repeat'] == repeat]
                print(json.dumps({'mode': mode, 'repeat': repeat, 'n': len(subset),
                    'success': sum(r['ok'] for r in subset),
                    'mean_s': statistics.fmean(r['elapsed_s'] for r in subset),
                    'recall': statistics.fmean(r['recall'] for r in subset if r.get('recall') is not None)}), flush=True)
    public.close()
    private.close()


def main():
    os.umask(0o077)
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--capture', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--repeats', type=int, default=2)
    p.add_argument('--concurrency', type=int, default=4)
    p.add_argument('--min-candidates', type=int, default=1)
    asyncio.run(run(p.parse_args()))


if __name__ == '__main__':
    main()

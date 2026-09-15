"""Steady-concurrency replay of frozen orchestrator requests against several
model endpoints (production instance, dedicated plain instance, dedicated
speculative instance). Per-request latency includes the whole HTTP round trip.

No prompt is rewritten. Variant order rotates per block and reverses on odd
repeats. Speculative-decoding counters are read from the server's /metrics
before and after each block; telemetry never aborts a trial.
"""
from __future__ import annotations
import argparse, asyncio, hashlib, json, random, re, time
from datetime import datetime, timezone
from pathlib import Path


def normalized(answer):
    try:
        parsed = json.loads(answer)
        return json.dumps(parsed, sort_keys=True, ensure_ascii=False), isinstance(parsed, dict), parsed
    except (TypeError, ValueError):
        return answer, False, None


async def main():
    import httpx
    from dotenv import dotenv_values
    ap = argparse.ArgumentParser()
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--serving-env', type=Path, required=True)
    ap.add_argument('--variant', action='append', required=True, help='name=http://host:port')
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--concurrency', type=int, default=4)
    ap.add_argument('--block', type=int, default=16)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--stages', default='classify,prepare')
    ap.add_argument('--seed', type=int, default=20260906)
    ap.add_argument('--tag', default='')
    ap.add_argument('--fold', choices=['even', 'odd'], help='replay only questions of this fold (case id parity)')
    args = ap.parse_args()
    key = dotenv_values(args.serving_env)['VLLM_API_KEY']
    variants = dict(v.split('=', 1) for v in args.variant)
    inputs = [r for r in json.loads(args.requests.read_text()) if r['stage'] in args.stages.split(',')]
    if args.fold:
        inputs = [r for r in inputs if ('even' if int(re.sub(r'\D', '', r['case_id'])) % 2 == 0 else 'odd') == args.fold]
    if args.limit:
        inputs = inputs[:args.limit]
    manifest = {'scope': 'frozen classify/preparation requests replayed independently; not full application',
                'unique_questions': len({r['case_id'] for r in inputs}), 'requests_per_variant_repeat': len(inputs),
                'variants': variants, 'repeats': args.repeats, 'concurrency': args.concurrency, 'block': args.block,
                'input_file_sha256': hashlib.sha256(args.requests.read_bytes()).hexdigest(), 'fold': args.fold,
                'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), 'tag': args.tag,
                'started_utc': datetime.now(timezone.utc).isoformat()}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    mpath = args.output.with_suffix('.manifest.json')
    if mpath.exists():
        old = json.loads(mpath.read_text())
        for f in ['variants', 'repeats', 'concurrency', 'block', 'input_file_sha256']:
            if old[f] != manifest[f]:
                raise RuntimeError('resume would change ' + f)
        manifest['resumed_from'] = old.get('started_utc')
    mpath.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + '\n')
    done = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            r = json.loads(line)
            done.add((r['case_id'], r['stage'], r['variant'], r['repeat']))
    sink = args.output.open('a', buffering=1)
    bsink = args.output.with_suffix('.blocks.jsonl').open('a', buffering=1)
    names = {'vllm:num_requests_running', 'vllm:num_requests_waiting', 'vllm:num_preemptions_total',
             'vllm:prefix_cache_queries_total', 'vllm:prefix_cache_hits_total', 'vllm:prompt_tokens_total',
             'vllm:generation_tokens_total', 'vllm:request_success_total', 'vllm:spec_decode_num_drafts_total',
             'vllm:spec_decode_num_draft_tokens_total', 'vllm:spec_decode_num_accepted_tokens_total'}

    async def counters(base):
        try:
            async with httpx.AsyncClient(timeout=5., trust_env=False) as t:
                r = await t.get(base.rstrip('/') + '/metrics')
                r.raise_for_status()
        except httpx.HTTPError as exc:
            return {'telemetry_error_type': type(exc).__name__}
        values = {}
        for line in r.text.splitlines():
            name = re.split(r'[ {]', line, 1)[0]
            if name in names:
                values[name] = values.get(name, 0.) + float(line.rsplit(' ', 1)[1])
            elif name == 'vllm:spec_decode_num_accepted_tokens_per_pos':
                m = re.search(r'position="(\d+)"', line)
                if m:
                    values[f'accepted_pos_{m.group(1)}'] = float(line.rsplit(' ', 1)[1])
        return values

    async with httpx.AsyncClient(timeout=httpx.Timeout(180., connect=5.), trust_env=False,
                                 limits=httpx.Limits(max_connections=32, max_keepalive_connections=32)) as client:
        async def one(item, variant, repeat, block, sem, warmup=False):
            identity = (item['case_id'], item['stage'], variant, repeat)
            if identity in done and not warmup:
                return None
            raw = json.dumps(item['payload'], ensure_ascii=False).encode()
            url = variants[variant].rstrip('/') + '/v1/chat/completions'
            rid = f'{args.output.stem}-{repeat}-{block}-{variant}-{item["case_id"]}-{item["stage"]}'
            async with sem:
                start = time.perf_counter()
                row = {k: item[k] for k in ['case_id', 'kind', 'stage', 'question_sha256', 'payload_sha256']}
                row.update(variant=variant, repeat=repeat, block=block, request_id=rid,
                           started_utc=datetime.now(timezone.utc).isoformat())
                try:
                    resp = await client.post(url, content=raw, headers={'Authorization': 'Bearer ' + key,
                                             'Content-Type': 'application/json', 'X-Experiment-ID': rid})
                    row['status'] = resp.status_code
                    resp.raise_for_status()
                    result = resp.json()
                    choice = result['choices'][0]
                    answer = choice['message'].get('content') or ''
                    reason = choice['message'].get('reasoning_content') or choice['message'].get('reasoning') or ''
                    norm, valid, parsed = normalized(answer)
                    row.update(ok=True, answer=answer, answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                               normalized_sha256=hashlib.sha256(norm.encode()).hexdigest(), valid_json=valid,
                               parsed=parsed, reasoning_chars=len(reason), finish_reason=choice.get('finish_reason'),
                               usage=result.get('usage'))
                except Exception as exc:
                    row.update(ok=False, error_type=type(exc).__name__, error=str(exc)[:200])
                row['elapsed_s'] = time.perf_counter() - start
            if not warmup:
                sink.write(json.dumps(row, ensure_ascii=False) + '\n')
            return row

        for variant in variants:
            sem = asyncio.Semaphore(args.concurrency)
            rows = await asyncio.gather(*(one(item, variant, -1, -1, sem, True) for item in inputs[:2]))
            if not all(r['ok'] for r in rows):
                raise RuntimeError(f'warmup failed for {variant}: ' + str([r.get('error') for r in rows]))
            print('warmup', variant, [round(r['elapsed_s'], 2) for r in rows], flush=True)
        vnames = list(variants)
        for repeat in range(args.repeats):
            ordered = list(inputs)
            random.Random(args.seed + repeat).shuffle(ordered)
            for block, offset in enumerate(range(0, len(ordered), args.block)):
                batch = ordered[offset:offset + args.block]
                shift = (block + repeat) % len(vnames)
                order = vnames[shift:] + vnames[:shift]
                if repeat % 2:
                    order.reverse()
                for variant in order:
                    if all((x['case_id'], x['stage'], variant, repeat) in done for x in batch):
                        continue
                    before = await counters(variants[variant])
                    sem = asyncio.Semaphore(args.concurrency)
                    started = time.perf_counter()
                    rows = await asyncio.gather(*(one(x, variant, repeat, block, sem) for x in batch))
                    elapsed = time.perf_counter() - started
                    after = await counters(variants[variant])
                    actual = [r for r in rows if r is not None]
                    bsink.write(json.dumps({'repeat': repeat, 'block': block, 'variant': variant, 'n': len(actual),
                                            'wall_s': elapsed, 'ok': sum(r['ok'] for r in actual),
                                            'counters_before': before, 'counters_after': after}) + '\n')
                    if actual:
                        print(json.dumps({'repeat': repeat, 'block': block, 'variant': variant, 'n': len(actual),
                                          'ok': sum(r['ok'] for r in actual), 'wall_s': round(elapsed, 2),
                                          'mean_s': round(sum(r['elapsed_s'] for r in actual) / len(actual), 3)}), flush=True)
                        if sum(not r['ok'] for r in actual) >= 2:
                            raise RuntimeError('two or more failures in one block; stopping for safety review')
    sink.close(); bsink.close()
    print('complete', flush=True)


if __name__ == '__main__':
    asyncio.run(main())

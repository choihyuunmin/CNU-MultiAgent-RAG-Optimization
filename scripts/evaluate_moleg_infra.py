"""Counterbalanced, bounded live-GPU request replay through an external gateway.

All gateway and upstream waiting is included. No live application mutations.
Prompts/visible outputs stay private; hidden reasoning is counted, never saved.
"""
from __future__ import annotations
import argparse
import asyncio
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import re
import statistics
import time


async def main():
    import httpx
    from dotenv import dotenv_values
    ap = argparse.ArgumentParser()
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--serving-env', type=Path, required=True)
    ap.add_argument('--gateway', required=True)
    ap.add_argument('--variants', default='pass,fifo8,fair8')
    ap.add_argument('--repeats', type=int, default=2)
    ap.add_argument('--burst', type=int, default=16)
    ap.add_argument('--limit', type=int)
    ap.add_argument('--telemetry-url', default='http://127.0.0.1:8000/metrics')
    ap.add_argument('--proxy-config', type=Path)
    ap.add_argument('--proxy-url', default='http://127.0.0.1:4000')
    ap.add_argument('--proxy-model-alias', default='orchestrator')
    ap.add_argument('--mixed-single-stage', action='store_true')
    args = ap.parse_args()
    serving = dotenv_values(args.serving_env)
    key = serving['VLLM_API_KEY']
    inputs = json.loads(args.requests.read_text())
    if args.limit:
        inputs = inputs[:args.limit]
    variants = args.variants.split(',')
    if args.mixed_single_stage:
        questions = sorted(set(r['case_id'] for r in inputs))
        random.Random(91006).shuffle(questions)
        selected = {q:('classify' if i%2 == 0 else 'prepare') for i,q in enumerate(questions)}
        inputs = [r for r in inputs if r['stage'] == selected[r['case_id']]]
    proxy_key = None
    if 'proxy' in variants:
        import yaml
        config = yaml.safe_load(args.proxy_config.read_text())
        value = config['general_settings']['master_key']
        proxy_key = (os.environ.get(value.split('/',1)[1]) or serving.get(value.split('/',1)[1])) if value.startswith('os.environ/') else value
        route = next(r['litellm_params'] for r in config['model_list'] if r['model_name'] == args.proxy_model_alias)
        assert all(route['model'].removeprefix('openai/') == r['payload']['model'] for r in inputs)
        if not proxy_key:
            raise RuntimeError('missing proxy authentication')
    manifest = {'scope':'independent frozen analysis requests, not full application',
                'unique_questions':len(set(r['case_id'] for r in inputs)), 'requests_per_variant_repeat':len(inputs),
                'variants':variants,'repeats':args.repeats,'burst':args.burst,'mixed_single_stage':args.mixed_single_stage,
                'input_file_sha256':hashlib.sha256(args.requests.read_bytes()).hexdigest(),
                'script_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                'inputs':[{k:v for k,v in r.items() if k!='payload'} for r in inputs]}
    manifest_path=args.output.with_suffix('.manifest.json')
    if manifest_path.exists():
        original=json.loads(manifest_path.read_text())
        for field in ['inputs','variants','repeats','burst','mixed_single_stage','input_file_sha256']:
            if original[field]!=manifest[field]:
                raise RuntimeError('resume would change '+field)
        versions=list(original.get('collector_versions',[original['script_sha256']]))
        if manifest['script_sha256'] not in versions:
            versions.append(manifest['script_sha256'])
        manifest.update(collector_versions=versions,resume_utc=datetime.now(timezone.utc).isoformat())
    manifest_path.write_text(json.dumps(manifest,indent=2)+'\n')
    done = set()
    previous=[]
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            r = json.loads(line)
            previous.append(r)
            done.add((r['case_id'], r['stage'], r['variant'], r['repeat']))
    sink = args.output.open('a', buffering=1)
    block_sink = args.output.with_suffix('.blocks.jsonl').open('a', buffering=1)
    existing_blocks={}
    if args.output.with_suffix('.blocks.jsonl').exists():
        for line in args.output.with_suffix('.blocks.jsonl').read_text().splitlines():
            b=json.loads(line);existing_blocks[(b['variant'],b['repeat'],b['block'])]=b
    # Preserve completed requests after a telemetry-only failure. Do not invent
    # the lost wall clock of that burst, and do not rerun successful inferences.
    from collections import defaultdict
    grouped=defaultdict(list)
    for r in previous:grouped[(r['variant'],r['repeat'],r['block'])].append(r)
    for k,values in grouped.items():
        if k not in existing_blocks and len(values)==args.burst:
            block_sink.write(json.dumps({'variant':k[0],'repeat':k[1],'block':k[2],
                'n':len(values),'ok':sum(r['ok'] for r in values),'wall_s':None,
                'measurement_note':'request rows complete; burst wall lost after telemetry failure; no fabricated time'})+'\n')
    async with httpx.AsyncClient(timeout=httpx.Timeout(120., connect=5.),
                                 limits=httpx.Limits(max_connections=32, max_keepalive_connections=32), trust_env=False) as client:
        async def counters():
            try:
                # Telemetry is outside timing and must not abort valid trials.
                # A separate fresh connection avoids the model server's idle
                # keepalive timeout racing the five-second polling interval.
                async with httpx.AsyncClient(timeout=5.,trust_env=False,
                    limits=httpx.Limits(max_connections=1,max_keepalive_connections=0)) as telemetry:
                    response = await telemetry.get(args.telemetry_url)
                    response.raise_for_status()
            except httpx.HTTPError as exc:
                return {'telemetry_error_type':type(exc).__name__}
            values = {}
            names = {'vllm:num_requests_running','vllm:num_requests_waiting','vllm:num_preemptions_total',
                     'vllm:prefix_cache_queries_total','vllm:prefix_cache_hits_total',
                     'vllm:prompt_tokens_total','vllm:generation_tokens_total','vllm:request_success_total'}
            for line in response.text.splitlines():
                name = re.split(r'[ {]',line,1)[0]
                if name in names:
                    values[name] = values.get(name, 0.)+float(line.rsplit(' ',1)[1])
            return values
        async def one(item, variant, repeat, block, barrier, warmup=False):
            identity = (item['case_id'], item['stage'], variant, repeat)
            if identity in done and not warmup:
                return None
            raw = json.dumps(item['payload'], ensure_ascii=False).encode()
            url = args.gateway.rstrip('/')+'/'+variant+'/v1/chat/completions'
            auth_key = key
            if variant == 'proxy':
                payload = dict(item['payload'], model=args.proxy_model_alias)
                raw = json.dumps(payload, ensure_ascii=False).encode()
                url = args.proxy_url.rstrip('/')+'/v1/chat/completions'
                auth_key = proxy_key
            rid = f'{args.output.stem}-{repeat}-{block}-{variant}-{item["case_id"]}-{item["stage"]}'
            await barrier.wait()
            start = time.perf_counter()
            row = {k:item[k] for k in ['case_id','kind','stage','question_sha256','payload_sha256']}
            row.update(variant=variant, repeat=repeat, block=block, request_id=rid,
                       wire_sha256=hashlib.sha256(raw).hexdigest(), started_utc=datetime.now(timezone.utc).isoformat())
            try:
                response = await client.post(url, content=raw,
                                             headers={'Authorization':'Bearer '+auth_key, 'Content-Type':'application/json', 'X-Experiment-ID':rid})
                row['status'] = response.status_code
                row['response_sha256'] = hashlib.sha256(response.content).hexdigest()
                response.raise_for_status()
                result = response.json()
                choice = result['choices'][0]
                message = choice['message']
                answer = message.get('content') or ''
                reason = message.get('reasoning_content') or message.get('reasoning') or ''
                try:
                    parsed = json.loads(answer)
                    normalized = json.dumps(parsed, sort_keys=True, ensure_ascii=False)
                    valid = isinstance(parsed, dict)
                except (TypeError, ValueError):
                    parsed, normalized, valid = None, answer, False
                row.update(ok=True, answer=answer, answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                           normalized_sha256=hashlib.sha256(normalized.encode()).hexdigest(), valid_json=valid,
                           parsed=parsed, reasoning_chars=len(reason), finish_reason=choice.get('finish_reason'),
                           usage=result.get('usage'), gateway_wait_s=float(response.headers.get('x-experiment-queue-s', 0)),
                           upstream_s=float(response.headers.get('x-experiment-upstream-s', 0)))
            except Exception as exc:
                row.update(ok=False, error_type=type(exc).__name__)
            row['elapsed_s'] = time.perf_counter()-start
            if not warmup:
                sink.write(json.dumps(row, ensure_ascii=False)+'\n')
            return row

        event = asyncio.Event(); event.set()
        for variant in variants:
            for item in inputs[:2]:
                row = await one(item, variant, -1, -1, event, True)
                if not row['ok']:
                    raise RuntimeError('warmup failed: '+row.get('error_type','unknown'))
        for repeat in range(args.repeats):
            ordered = list(inputs)
            random.Random(20260906+repeat).shuffle(ordered)
            for block, offset in enumerate(range(0, len(ordered), args.burst)):
                batch = ordered[offset:offset+args.burst]
                shift = (block+repeat) % len(variants)
                order = variants[shift:]+variants[:shift]
                if repeat % 2:
                    order.reverse()
                for variant in order:
                    if all((x['case_id'], x['stage'], variant, repeat) in done for x in batch):
                        continue
                    barrier = asyncio.Event()
                    before = await counters()
                    tasks = [asyncio.create_task(one(x, variant, repeat, block, barrier)) for x in batch]
                    await asyncio.sleep(0)
                    started = time.perf_counter()
                    barrier.set()
                    rows = await asyncio.gather(*tasks)
                    elapsed = time.perf_counter()-started
                    after = await counters()
                    actual = [r for r in rows if r is not None]
                    block_sink.write(json.dumps({'repeat':repeat,'block':block,'variant':variant,'n':len(actual),
                                                  'wall_s':elapsed,'ok':sum(r['ok'] for r in actual),
                                                  'counters_before':before,'counters_after':after})+'\n')
                    if actual:
                        print(json.dumps({'repeat':repeat, 'block':block, 'variant':variant, 'n':len(actual),
                                          'ok':sum(r['ok'] for r in actual), 'mean_s':round(statistics.fmean(r['elapsed_s'] for r in actual),3),
                                          'wall_s':round(elapsed,3)}), flush=True)
                        if sum(not r['ok'] for r in actual) > 1:
                            raise RuntimeError('multiple request errors: stop to avoid production disturbance')
    sink.close(); block_sink.close()
    print('COMPLETE', flush=True)


if __name__ == '__main__':
    asyncio.run(main())

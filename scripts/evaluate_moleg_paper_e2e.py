"""Block-interleaved, repeated HTTP/SSE benchmark with resumable JSONL rows."""
from __future__ import annotations
import argparse
import asyncio
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
import time
import uuid


async def run(args):
    import httpx
    cases = json.loads(args.cases.read_text())[:args.limit]
    variants = dict(v.split("=", 1) for v in args.variant)
    assert len({x['question'] for x in cases}) == len(cases)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    done = set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            r = json.loads(line)
            done.add((r['case_id'],r['variant'],r['repeat']))
    clients = {k:httpx.AsyncClient(timeout=httpx.Timeout(args.timeout,connect=15),
        limits=httpx.Limits(max_connections=args.concurrency,max_keepalive_connections=args.concurrency)) for k in variants}
    sem = asyncio.Semaphore(args.concurrency)
    sink = args.output.open('a',encoding='utf-8',buffering=1)
    start_all = time.perf_counter()

    async def request(case, variant, repeat, warmup=False):
        async with sem:
            session = f"paper-{case['case_id']}-{variant}-{repeat}-{uuid.uuid4().hex[:10]}"
            row = {'case_id':case['case_id'],'kind':case['kind'],'variant':variant,'repeat':repeat,
                   'session_id':session,'route':args.route,'utc':datetime.now(timezone.utc).isoformat(),
                   'question_sha256':hashlib.sha256(case['question'].encode()).hexdigest()}
            start = time.perf_counter()
            pieces, events = [], []
            first_event = first_token = done_time = None
            result = None
            try:
                async with clients[variant].stream('POST',variants[variant].rstrip('/')+args.route,
                        json={'prompt':case['question'],'session_id':session}) as response:
                    row['status'] = response.status_code
                    response.raise_for_status()
                    if args.route.endswith('/stream'):
                        async for line in response.aiter_lines():
                            if not line.startswith('data:'): continue
                            event = json.loads(line[5:].strip())
                            now = time.perf_counter()-start
                            if first_event is None: first_event = now
                            stage = event.get('stage','')
                            if stage == 'token' and event.get('delta'):
                                if first_token is None: first_token = now
                                pieces.append(event['delta'])
                            elif stage == 'done':
                                result = event.get('result')
                                done_time = now
                            elif stage == 'error':
                                row['pipeline_error'] = event.get('message','')[:300]
                            if not events or events[-1]['stage'] != stage:
                                events.append({'stage':stage,'s':now})
                    else:
                        result = json.loads(await response.aread())
                        done_time = time.perf_counter()-start
                row['ok'] = isinstance(result,dict) and not row.get('pipeline_error')
            except Exception as e:
                row['ok'] = False
                row['error_type'] = type(e).__name__
            row.update(elapsed_s=time.perf_counter()-start,first_event_s=first_event,
                       ttft_s=first_token,done_s=done_time,events=events,
                       streamed_chars=sum(map(len,pieces)),streamed_sha256=hashlib.sha256(''.join(pieces).encode()).hexdigest(),
                       result=result)
            if not warmup:
                sink.write(json.dumps(row,ensure_ascii=False)+'\n')
            return row

    # Every warmup completes before measured work. Fresh session each time.
    for variant in variants:
        for case in cases[:2]:
            r = await request(case,variant,-1,True)
            print('warmup',variant,case['case_id'],r['ok'],round(r['elapsed_s'],2),flush=True)
            if not r['ok']: raise RuntimeError('warmup failed; refusing main run')
    names = list(variants)
    for repeat in range(args.repeats):
        ordered = list(cases)
        random.Random(args.seed+repeat).shuffle(ordered)
        for block,offset in enumerate(range(0,len(ordered),args.block_size)):
            batch = ordered[offset:offset+args.block_size]
            shift = (block+repeat)%len(names)
            order = names[shift:]+names[:shift]
            if (block//len(names))%2: order.reverse()
            for variant in order:
                missing = [c for c in batch if (c['case_id'],variant,repeat) not in done]
                rows = await asyncio.gather(*(request(c,variant,repeat) for c in missing))
                if rows:
                    print(json.dumps({'repeat':repeat,'block':block,'variant':variant,
                        'finished':offset+len(batch),'n':len(cases),'ok':sum(r['ok'] for r in rows),
                        'mean_s':sum(r['elapsed_s'] for r in rows)/len(rows),
                        'wall_s':round(time.perf_counter()-start_all)},ensure_ascii=False),flush=True)
    sink.close()
    for c in clients.values(): await c.aclose()


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cases',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--variant',action='append',required=True)
    ap.add_argument('--route',default='/api/generate/stream')
    ap.add_argument('--limit',type=int,default=400)
    ap.add_argument('--repeats',type=int,default=2)
    ap.add_argument('--concurrency',type=int,default=4)
    ap.add_argument('--block-size',type=int,default=8)
    ap.add_argument('--seed',type=int,default=20260905)
    ap.add_argument('--timeout',type=float,default=300)
    asyncio.run(run(ap.parse_args()))


if __name__=='__main__': main()

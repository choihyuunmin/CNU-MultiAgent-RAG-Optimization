"""Exploratory frozen-prompt generation control after primary GPU timings.

Separates proxy routing and low reasoning from retrieval changes, on a fixed
60-question slice with two repetitions. Raw model TTFT is not API TTFT. Inputs
and visible answers stay private; reasoning text is never saved.
"""
from __future__ import annotations
import argparse
import asyncio
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import statistics
import time
from moleg_paper_runtime import bootstrap
from moleg_paper_metrics import paired_cluster_ci,timing


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,ensure_ascii=False).encode()).hexdigest()


async def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=60)
    args=ap.parse_args();root=args.directory
    _,serving,config,resolve=bootstrap()
    from openai import AsyncOpenAI
    import infra.llm.client as llm
    cases={c['case_id']:c for c in json.loads((root/'cases.json').read_text())}
    streams=[json.loads(x) for x in (root/'e2e_stream.jsonl').read_text().splitlines()]
    sessions={r['session_id']:r['case_id'] for r in streams if r['variant']=='baseline' and r['repeat']==0 and r['ok']}
    captured=defaultdict(list)
    for line in (root/'generation_requests.jsonl').read_text().splitlines():
        row=json.loads(line)
        if row['profile']=='baseline' and row['session_id'] in sessions:
            captured[sessions[row['session_id']]].append(row['kwargs'])
    strata=defaultdict(list);rng=random.Random(20260905)
    for q,requests in sorted(captured.items()):
        if len(requests)==1 and requests[0].get('model')=='worker agent':
            strata[cases[q]['kind']].append({'case_id':q,'kind':cases[q]['kind'],'kwargs':requests[0]})
    for values in strata.values():rng.shuffle(values)
    selected=[]
    while len(selected)<args.limit and any(strata.values()):
        for kind in sorted(strata):
            if strata[kind] and len(selected)<args.limit:selected.append(strata[kind].pop())
    if len(selected)!=args.limit:raise RuntimeError('insufficient single-call worker generation inputs')
    manifest={'n':len(selected),'repeats':2,'scope':'exploratory frozen upstream generation inputs, not full API',
              'inputs':[{'case_id':r['case_id'],'kind':r['kind'],'sha256':fingerprint(r['kwargs'])} for r in selected]}
    (root/'generation_replay_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    p=next(r['litellm_params'] for r in config['model_list'] if r['model_name']=='worker agent')
    client=AsyncOpenAI(base_url=resolve(p['api_base']),api_key=resolve(p.get('api_key')) or serving['VLLM_API_KEY'],
                       timeout=120,max_retries=0)
    names=['proxy_default','direct_default','direct_low'];sem=asyncio.Semaphore(4)
    path=root/'generation_replay.jsonl';rows=[]
    if path.exists():rows=[json.loads(x) for x in path.read_text().splitlines()]
    done={(r['case_id'],r['variant'],r['repeat']) for r in rows}
    sink=path.open('a',encoding='utf-8',buffering=1)
    async def one(request,name,repeat,warmup=False):
        key=(request['case_id'],name,repeat)
        if key in done:return
        async with sem:
            kwargs=request['kwargs'];start=time.perf_counter();ttft=None
            row={'case_id':request['case_id'],'kind':request['kind'],'variant':name,'repeat':repeat,
                 'input_sha256':fingerprint(kwargs)}
            source=None;content=[];reasoning_chars=0;finish_reason=None
            try:
                if name=='proxy_default':source=llm.acompletion_stream_via_proxy(**kwargs)
                else:
                    payload={'model':p['model'][len('openai/'):],'messages':kwargs['messages'],
                             'temperature':kwargs.get('temperature',0),'stream':True}
                    for k in ['tools','tool_choice','extra_body']:
                        if kwargs.get(k):payload[k]=kwargs[k]
                    if name=='direct_low':payload['reasoning_effort']='low'
                    source=await client.chat.completions.create(**payload)
                async for chunk in source:
                    if not chunk.choices:continue
                    finish_reason=chunk.choices[0].finish_reason or finish_reason
                    delta=chunk.choices[0].delta
                    if delta.content:
                        if ttft is None:ttft=time.perf_counter()-start
                        content.append(delta.content)
                    fields=delta.model_dump()
                    reasoning_chars+=len(fields.get('reasoning') or fields.get('reasoning_content') or '')
                row['ok']=True
            except Exception as exc:row.update(ok=False,error_type=type(exc).__name__)
            finally:
                if source is not None:
                    if hasattr(source,'aclose'):await source.aclose()
                    elif hasattr(source,'close'):await source.close()
            answer=''.join(content)
            row.update(elapsed_s=time.perf_counter()-start,ttft_s=ttft,reasoning_chars=reasoning_chars,
                       answer_chars=len(answer),answer_sha256=hashlib.sha256(answer.encode()).hexdigest(),
                       finish_reason=finish_reason,answer=answer)
            if not warmup:rows.append(row);sink.write(json.dumps(row,ensure_ascii=False)+'\n')
            return row
    for name in names:
        for request in selected[:2]:
            result=await one(request,name,-1,True)
            if not result['ok']:raise RuntimeError('generation replay warmup failed')
    for repeat in range(2):
        ordered=list(selected);random.Random(20260905+repeat).shuffle(ordered)
        for block,offset in enumerate(range(0,len(ordered),4)):
            shift=(block+repeat)%3;order=names[shift:]+names[:shift]
            if repeat:order.reverse()
            for name in order:await asyncio.gather(*(one(r,name,repeat) for r in ordered[offset:offset+4]))
            print('generation_replay',repeat,min(offset+4,len(ordered)),len(ordered),flush=True)
    sink.close();await client.close()
    summary={'expected_rows':len(selected)*6,'rows':len(rows),'scope':manifest['scope'],'variants':{},'comparisons':{}}
    groups=defaultdict(lambda:defaultdict(list))
    for row in rows:groups[row['variant']][row['case_id']].append(row)
    for name,queries in groups.items():
        values=[r for rs in queries.values() for r in rs];repeat_exact=[]
        for rs in queries.values():
            byrepeat={r['repeat']:r for r in rs}
            if all(k in byrepeat and byrepeat[k]['ok'] for k in [0,1]):
                repeat_exact.append(byrepeat[0]['answer_sha256']==byrepeat[1]['answer_sha256'])
        summary['variants'][name]={'n':len(values),'success':sum(r['ok'] for r in values),
            'empty_answer_count':sum(not r['answer_chars'] for r in values),
            'length_limited_count':sum(r.get('finish_reason')=='length' for r in values),
            'latency':timing([r['elapsed_s'] for r in values]),'ttft':timing([r['ttft_s'] for r in values]),
            'mean_reasoning_chars':statistics.fmean(r['reasoning_chars'] for r in values),
            'mean_answer_chars':statistics.fmean(r['answer_chars'] for r in values),
            'exact_answer_between_repeats':statistics.fmean(repeat_exact) if repeat_exact else None}
    for baseline,candidate in [('proxy_default','direct_default'),('direct_default','direct_low')]:
        pairs=[]
        for q,rs in groups[baseline].items():
            cs=groups[candidate].get(q,[])
            if cs:pairs.append((statistics.fmean(r['elapsed_s'] for r in rs),statistics.fmean(r['elapsed_s'] for r in cs)))
        summary['comparisons'][baseline+'_vs_'+candidate]=paired_cluster_ci(pairs)
    (root/'generation_replay_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary),flush=True)


if __name__=='__main__':asyncio.run(main())

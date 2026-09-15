"""Exploratory original-query anchoring ablation; run AFTER timed E2E.

Tests whether avoiding generative keyword rewriting preserves known-item quality.
This is a search-only path, not a replacement for the application's guardrails,
intent routing, conversation memory, or generation. It is not a full API speedup.
"""
from __future__ import annotations
import argparse
import asyncio
import contextvars
import json
import logging
from pathlib import Path
import random
import threading
import time
from moleg_paper_runtime import bootstrap


def literal_country(question,countries):
    # Match longest names first and mask nested aliases such as China/Hong Kong.
    remaining=question;matches=[]
    for country in sorted(countries,key=len,reverse=True):
        if country and country in remaining:
            matches.append(country);remaining=remaining.replace(country,' ')
    return matches[0] if len(matches)==1 else None


def reciprocal_rank_fusion(*rankings,k=60):
    scores={};docs={};order={}
    for ranked in rankings:
        seen=set()
        for i,doc in enumerate(ranked):
            key=str(doc.get('id') or '')
            if not key or key in seen:continue
            seen.add(key);order.setdefault(key,len(order))
            if key not in docs or (doc.get('source')=='article' and docs[key].get('source')!='article'):
                docs[key]=doc
            scores[key]=scores.get(key,0)+1/(k+i+1)
    return [docs[key] for key in sorted(scores,key=lambda key:(-scores[key],order[key]))]


async def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--limit',type=int,default=400)
    args=ap.parse_args()
    _,serving,config,resolve=bootstrap()
    import requests
    from openai import AsyncOpenAI
    import agent.validation_agent as validation
    from agent.validation_agent import run_preparation
    from domain.country.service import get_available_countries
    import tools.search_tool.engine as module
    logging.disable(logging.CRITICAL)
    countries=get_available_countries();engine=module.search_engine
    fast_config=next(x['litellm_params'] for x in config['model_list'] if x['model_name']=='worker agent')
    fast_client=AsyncOpenAI(base_url=resolve(fast_config['api_base']),
        api_key=resolve(fast_config.get('api_key')) or serving['VLLM_API_KEY'],timeout=120,max_retries=0)
    use_fast=contextvars.ContextVar('use_fast_extraction',default=False)
    original_llm=validation.call_llm
    async def preparation_llm(**kwargs):
        if not use_fast.get():return await original_llm(**kwargs)
        output=await fast_client.chat.completions.create(model=fast_config['model'][len('openai/'):],
            messages=kwargs['messages'],temperature=0,response_format=kwargs.get('response_format'),
            extra_body={'reasoning_effort':'low'})
        return output.choices[0].message.content or ''
    validation.call_llm=preparation_llm
    engine._rerank_results=lambda query,documents,*a,**kw:documents
    local=threading.local();sem=asyncio.Semaphore(4)
    cases=json.loads((args.directory/'cases.json').read_text())[:args.limit]
    rng=random.Random(20260905);rng.shuffle(cases)
    path=args.directory/'query_anchor.jsonl';done=set()
    if path.exists():
        done={(r['case_id'],r['variant']) for r in (json.loads(x) for x in path.read_text().splitlines())}
    sink=path.open('a',encoding='utf-8',buffering=1)

    def rerank(query,docs):
        if not docs:return []
        if not hasattr(local,'session'):local.session=requests.Session()
        response=local.session.post(module.RERANK_API_URL,
            headers={'Authorization':'Bearer '+serving['VLLM_API_KEY']},json={
            'model':module.RERANK_MODEL_NAME,'query':query[:module.RERANK_QUERY_MAX_CHARS],
            'documents':[str(d.get('content') or d.get('title') or '')[:module.RERANK_DOCUMENT_TEXT_MAX_CHARS] for d in docs]},timeout=30)
        response.raise_for_status();ranked=response.json()['results']
        assert sorted(r['index'] for r in ranked)==list(range(len(docs)))
        ranked.sort(key=lambda r:(-r['relevance_score'],r['index']))
        return [docs[r['index']] for r in ranked[:20]]

    async def original(question):
        country=literal_country(question,countries)
        start=time.perf_counter()
        out=await asyncio.to_thread(engine.search_laws,country,[],question)
        return out.get('laws',[]),question,country,{'query':question,'keywords':[],
            'preparation_s':0.,'search_s':time.perf_counter()-start}

    async def prepared(question):
        start=time.perf_counter()
        p=await run_preparation(history=[{'role':'user','content':question}],request_id='query-anchor')
        prep_s=time.perf_counter()-start
        if p.get('early_exit'):raise RuntimeError('preparation failed')
        kw=list(dict.fromkeys(p.get('keywords_original',[])+p.get('keywords_transformed',[])))
        query=p.get('transformed_query') or ' '.join(kw) or question
        start=time.perf_counter()
        out=await asyncio.to_thread(engine.search_laws,p.get('country') or None,kw,query)
        return out.get('laws',[]),query,p.get('country'),{'query':query,'keywords':kw,
            'preparation_s':prep_s,'search_s':time.perf_counter()-start}

    async def one(case,variant):
        if (case['case_id'],variant) in done:return
        async with sem:
            start=time.perf_counter();row={'case_id':case['case_id'],'variant':variant,'kind':case['kind']}
            try:
                if variant in ['prepared','fast_extraction']:
                    token=use_fast.set(variant=='fast_extraction')
                    try:docs,query,country,details=await prepared(case['question'])
                    finally:use_fast.reset(token)
                elif variant=='original':docs,query,country,details=await original(case['question'])
                else:
                    a,b=await asyncio.gather(original(case['question']),prepared(case['question']))
                    docs=reciprocal_rank_fusion(a[0],b[0]);query=case['question'];country=a[2]
                    details={'branches':[a[3],b[3]]}
                rerank_start=time.perf_counter()
                ranked=await asyncio.to_thread(rerank,query,docs)
                row.update(ok=True,ranked_ids=[str(d['id']) for d in ranked],country=country,
                           candidate_ids=[str(d['id']) for d in docs],
                           candidates=len(docs),rerank_s=time.perf_counter()-rerank_start,preparation=details)
            except Exception as exc:row.update(ok=False,error_type=type(exc).__name__)
            row['elapsed_s']=time.perf_counter()-start
            sink.write(json.dumps(row,ensure_ascii=False)+'\n')
    names=['prepared','original','anchored_union','fast_extraction']
    for block,offset in enumerate(range(0,len(cases),8)):
        order=names[block%len(names):]+names[:block%len(names)]
        for name in order:
            await asyncio.gather(*(one(c,name) for c in cases[offset:offset+8]))
        print('query_anchor',min(offset+8,len(cases)),len(cases),flush=True)
    sink.close()
    await fast_client.close()


if __name__=='__main__':asyncio.run(main())

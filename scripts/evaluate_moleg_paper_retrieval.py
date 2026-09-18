"""Reranker ablation on frozen real LLM extraction and real OpenSearch results.

Compares no rerank, authenticated current content view, metadata view, and query
window view. Three repeats use fresh retrieval/rerank calls, counterbalanced
order. Documents and question text are saved only to the private output file.
"""
from __future__ import annotations
import argparse
import asyncio
import copy
import json
import logging
from pathlib import Path
import random
import threading
import time

from moleg_paper_runtime import bootstrap, evidence_text


async def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--cases',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--repeats',type=int,default=3)
    ap.add_argument('--limit',type=int,default=400)
    ap.add_argument('--extraction-cache',type=Path)
    ap.add_argument('--variants',default='raw,auth_content,metadata,window')
    args=ap.parse_args()
    variant_names=args.variants.split(',')
    if not set(variant_names)<=set(['raw','auth_content','strict_auth_content','metadata','window']):
        raise ValueError('unknown reranking view')
    _,serving,_,_=bootstrap()
    from agent.validation_agent import run_preparation
    import tools.search_tool.engine as module
    import requests
    logging.disable(logging.CRITICAL)
    engine=module.search_engine
    # Disable only the broken reranker for candidate collection, preserving all
    # hybrid candidates and their current order.
    engine._rerank_results=lambda query,documents,*a,**kw: documents
    cases=json.loads(args.cases.read_text())[:args.limit]
    sem=asyncio.Semaphore(4)
    local=threading.local()
    def rank(query,documents,variant):
        if variant=='raw': return list(documents),0.0
        if not hasattr(local,'session'): local.session=requests.Session()
        if variant=='strict_auth_content':
            texts=[engine._clip_rerank_text(str(d.get('content') or d.get('title') or ''),module.RERANK_DOCUMENT_TEXT_MAX_CHARS) for d in documents]
        elif variant=='auth_content':
            texts=[str(d.get('content') or d.get('title') or '')[:module.RERANK_DOCUMENT_TEXT_MAX_CHARS] for d in documents]
        else:
            texts=[evidence_text(query,d,module.RERANK_DOCUMENT_TEXT_MAX_CHARS,variant=='window') for d in documents]
        start=time.perf_counter()
        if not texts:return [],0.0
        response=local.session.post(module.RERANK_API_URL,
            headers={'Authorization':'Bearer '+serving['VLLM_API_KEY']},
            json={'model':module.RERANK_MODEL_NAME,'query':engine._clip_rerank_text(query,module.RERANK_QUERY_MAX_CHARS)
                  if variant=='strict_auth_content' else query[:module.RERANK_QUERY_MAX_CHARS],
                  'documents':texts},timeout=30)
        response.raise_for_status()
        scores=response.json()['results']
        assert sorted(x['index'] for x in scores)==list(range(len(documents)))
        scores.sort(key=lambda x:(-x['relevance_score'],x['index']))
        return [documents[x['index']] for x in scores],time.perf_counter()-start

    extracted={}
    extraction_path=args.extraction_cache or args.output.with_suffix('.extraction.jsonl')
    if extraction_path.exists():
        for line in extraction_path.read_text().splitlines():
            row=json.loads(line);extracted[(row['case_id'],row['repeat'])]=row
    xf=extraction_path.open('a',encoding='utf-8',buffering=1)
    async def extract(case,repeat):
        key=(case['case_id'],repeat)
        if key in extracted:return extracted[key]
        async with sem:
            start=time.perf_counter()
            prep=await run_preparation(history=[{'role':'user','content':case['question']}],request_id='paper-extraction')
            row={'case_id':case['case_id'],'repeat':repeat,'elapsed_s':time.perf_counter()-start,'prep':prep}
            xf.write(json.dumps(row,ensure_ascii=False)+'\n');extracted[key]=row
            return row
    for repeat in range(2):
        for offset in range(0,len(cases),16):
            await asyncio.gather(*(extract(c,repeat) for c in cases[offset:offset+16]))
            print('extraction',repeat,offset+len(cases[offset:offset+16]),flush=True)
    xf.close()
    done=set()
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            r=json.loads(line);done.add((r['case_id'],r['repeat']))
    sink=args.output.open('a',encoding='utf-8',buffering=1)
    async def one(case,repeat,index):
        if (case['case_id'],repeat) in done:return
        async with sem:
            prep=extracted[(case['case_id'],0)]['prep']
            country=prep.get('country') or None
            keywords=list(dict.fromkeys(prep.get('keywords_original',[])+prep.get('keywords_transformed',[])))
            query=prep.get('transformed_query') or ' '.join(keywords) or case['question']
            start=time.perf_counter()
            row={'case_id':case['case_id'],'kind':case['kind'],'repeat':repeat,'variants':{}}
            try:
                out=await asyncio.to_thread(engine.search_laws,country,keywords,query)
                docs=out.get('laws',[])
                row['search_s']=time.perf_counter()-start
                row['candidate_count']=len(docs)
                names=list(variant_names)
                shift=(index+repeat)%len(names);names=names[shift:]+names[:shift]
                for name in names:
                    ranked,dt=await asyncio.to_thread(rank,query,docs,name)
                    row['variants'][name]={'rerank_s':dt,'ranked_ids':[str(d.get('id','')) for d in ranked],
                        'laws':[{k:d.get(k) for k in ['id','title','country','subject']} for d in ranked[:20]]}
                row['ok']=True
            except Exception as e:
                row['ok']=False;row['error_type']=type(e).__name__
            sink.write(json.dumps(row,ensure_ascii=False)+'\n')
    for repeat in range(args.repeats):
        ordered=list(cases);random.Random(20260905+repeat).shuffle(ordered)
        for offset in range(0,len(cases),16):
            await asyncio.gather(*(one(c,repeat,offset+i) for i,c in enumerate(ordered[offset:offset+16])))
            print('retrieval',repeat,min(offset+16,len(cases)),flush=True)
    sink.close()


if __name__=='__main__':asyncio.run(main())

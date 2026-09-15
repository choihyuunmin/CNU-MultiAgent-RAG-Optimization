"""Exploratory paired search with frozen extraction: change country only.

400 questions, two fresh searches each. A unique longest literal jurisdiction
overrides the prepared country; ambiguity leaves it unchanged. No source label
or reference answer enters the intervention. Run after primary GPU timings.
"""
import argparse
import asyncio
import hashlib
import json
import logging
from pathlib import Path
import random
import threading
import time
from moleg_paper_runtime import bootstrap
from evaluate_moleg_query_anchor import literal_country


async def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',type=Path,required=True)
    args=ap.parse_args();root=args.directory
    _,serving,_,_=bootstrap()
    import requests
    import tools.search_tool.engine as module
    logging.disable(logging.CRITICAL)
    engine=module.search_engine
    engine._rerank_results=lambda query,documents,*a,**kw:documents
    extraction={r['case_id']:r['prep'] for line in (root/'retrieval.extraction.jsonl').read_text().splitlines()
                for r in [json.loads(line)] if r['repeat']==0}
    cases=json.loads((root/'cases.json').read_text());random.Random(20260905).shuffle(cases)
    path=root/'country_ablation.jsonl'
    done={r['case_id'] for line in path.read_text().splitlines() for r in [json.loads(line)]} if path.exists() else set()
    sink=path.open('a',encoding='utf-8',buffering=1);local=threading.local();sem=asyncio.Semaphore(4)
    def search(country,keywords,query):
        start=time.perf_counter()
        docs=engine.search_laws(country,keywords,query).get('laws',[])
        search_s=time.perf_counter()-start;rerank_start=time.perf_counter()
        ranked=[]
        if docs:
            if not hasattr(local,'session'):local.session=requests.Session()
            response=local.session.post(module.RERANK_API_URL,headers={'Authorization':'Bearer '+serving['VLLM_API_KEY']},
                json={'model':module.RERANK_MODEL_NAME,'query':query[:module.RERANK_QUERY_MAX_CHARS],
                      'documents':[str(d.get('content') or d.get('title') or '')[:module.RERANK_DOCUMENT_TEXT_MAX_CHARS]
                                   for d in docs]},timeout=30)
            response.raise_for_status();scores=response.json()['results']
            assert sorted(r['index'] for r in scores)==list(range(len(docs)))
            scores.sort(key=lambda r:(-r['relevance_score'],r['index']))
            ranked=[docs[r['index']] for r in scores[:20]]
        return {'country':country,'candidate_ids':[str(d['id']) for d in docs],
                'ranked_ids':[str(d['id']) for d in ranked],
                'search_s':search_s,'rerank_s':time.perf_counter()-rerank_start,
                'elapsed_s':time.perf_counter()-start}
    async def one(case,index):
        if case['case_id'] in done:return
        async with sem:
            prep=extraction[case['case_id']]
            before=prep.get('country') or None
            literal=literal_country(case['question'],prep['available_countries'])
            after=literal or before
            keywords=list(dict.fromkeys(prep.get('keywords_original',[])+prep.get('keywords_transformed',[])))
            query=prep.get('transformed_query') or ' '.join(keywords) or case['question']
            row={'case_id':case['case_id'],'kind':case['kind'],'country_changed':before!=after,
                 'literal_abstained':literal is None,'variants':{},
                 'shared_search_input_sha256':hashlib.sha256(json.dumps(
                     {'keywords':keywords,'query':query},sort_keys=True,ensure_ascii=False).encode()).hexdigest()}
            order=[('prepared_country',before),('literal_country_only',after)]
            if index%2:order.reverse()
            for name,country in order:
                start=time.perf_counter()
                try:row['variants'][name]={'ok':True,**await asyncio.to_thread(search,country,keywords,query)}
                except Exception as exc:row['variants'][name]={'ok':False,'error_type':type(exc).__name__,
                    'country':country,'elapsed_s':time.perf_counter()-start}
            row['ok']=all(v['ok'] for v in row['variants'].values())
            sink.write(json.dumps(row,ensure_ascii=False)+'\n')
    for offset in range(0,len(cases),16):
        await asyncio.gather(*(one(c,offset+i) for i,c in enumerate(cases[offset:offset+16])))
        print('country_ablation',min(offset+16,len(cases)),len(cases),flush=True)
    sink.close()


if __name__=='__main__':asyncio.run(main())

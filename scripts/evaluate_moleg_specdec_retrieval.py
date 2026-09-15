"""Replay logged preparation outputs through the application's own
post-processing, hybrid search and authenticated reranker, then score them
against the source-derived silver labels. Identical outputs share one search
so that only genuine extraction differences can change retrieval.
"""
from __future__ import annotations
import argparse, asyncio, contextvars, hashlib, json, logging, threading, time
from pathlib import Path
from moleg_paper_runtime import bootstrap
from moleg_paper_metrics import known_item_metrics

ANSWER = contextvars.ContextVar('replayed_answer')


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--rows', type=Path, action='append', required=True, help='replay jsonl (prepare rows used)')
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--concurrency', type=int, default=4)
    args = ap.parse_args()
    _, serving, _, _ = bootstrap()
    from agent import validation_agent
    import tools.search_tool.engine as module
    import requests
    logging.disable(logging.CRITICAL)
    engine = module.search_engine
    engine._rerank_results = lambda query, documents, *a, **kw: documents  # candidates only; rerank below

    async def fake_call_llm(**kwargs):
        return ANSWER.get()
    validation_agent.call_llm = fake_call_llm
    cases = {c['case_id']: c for c in json.loads(args.cases.read_text())}
    local = threading.local()

    def rerank(query, documents):
        if not documents:
            return [], 0.
        if not hasattr(local, 'session'):
            local.session = requests.Session()
        texts = [str(d.get('content') or d.get('title') or '')[:module.RERANK_DOCUMENT_TEXT_MAX_CHARS] for d in documents]
        start = time.perf_counter()
        r = local.session.post(module.RERANK_API_URL, headers={'Authorization': 'Bearer ' + serving['VLLM_API_KEY']},
                               json={'model': module.RERANK_MODEL_NAME, 'query': query[:module.RERANK_QUERY_MAX_CHARS],
                                     'documents': texts}, timeout=30)
        r.raise_for_status()
        scores = r.json()['results']
        assert sorted(x['index'] for x in scores) == list(range(len(documents)))
        scores.sort(key=lambda x: (-x['relevance_score'], x['index']))
        return [documents[x['index']] for x in scores], time.perf_counter() - start

    # unique (case, normalized answer) keys across all arms
    work = {}
    origins = []
    for path in args.rows:
        for line in path.read_text().splitlines():
            r = json.loads(line)
            if r.get('stage') != 'prepare' or not r.get('ok'):
                continue
            key = (r['case_id'], r['normalized_sha256'])
            work.setdefault(key, r['answer'])
            origins.append({'file': path.name, 'variant': r['variant'], 'repeat': r['repeat'], 'case_id': r['case_id'],
                            'normalized_sha256': r['normalized_sha256']})
    done = {}
    if args.output.exists():
        for line in args.output.read_text().splitlines():
            r = json.loads(line); done[(r['case_id'], r['normalized_sha256'])] = r
    sink = args.output.open('a', encoding='utf-8', buffering=1)
    sem = asyncio.Semaphore(args.concurrency)

    async def one(key, answer):
        if key in done:
            return
        case = cases[key[0]]
        async with sem:
            token = ANSWER.set(answer)
            row = {'case_id': key[0], 'normalized_sha256': key[1], 'kind': case['kind']}
            try:
                prep = await validation_agent.run_preparation(history=[{'role': 'user', 'content': case['question']}],
                                                              request_id='specdec-eval')
                ANSWER.reset(token)
                if prep.get('early_exit'):
                    row.update(ok=False, error_type='early_exit'); sink.write(json.dumps(row, ensure_ascii=False) + '\n'); return
                country = prep.get('country') or None
                keywords = list(dict.fromkeys(prep.get('keywords_original', []) + prep.get('keywords_transformed', [])))
                query = prep.get('transformed_query') or ' '.join(keywords) or case['question']
                row['prep'] = {k: prep.get(k) for k in ['country', 'transformed_query', 'keywords_original', 'keywords_transformed']}
                start = time.perf_counter()
                out = await asyncio.to_thread(engine.search_laws, country, keywords, query)
                docs = out.get('laws', [])
                row['search_s'] = time.perf_counter() - start
                row['candidate_ids'] = [str(d.get('id', '')) for d in docs]
                ranked, dt = await asyncio.to_thread(rerank, query, docs)
                row['rerank_s'] = dt
                row['ranked_ids'] = [str(d.get('id', '')) for d in ranked][:20]
                row['metrics'] = known_item_metrics(row['ranked_ids'], case)
                if case.get('source_law_id'):
                    row['source_law_in_candidates'] = float(any(x.split('_')[0] == case['source_law_id'] for x in row['candidate_ids']))
                row['ok'] = True
            except Exception as e:
                row.update(ok=False, error_type=type(e).__name__, error=str(e)[:200])
            sink.write(json.dumps(row, ensure_ascii=False) + '\n')
    keys = list(work.items())
    print('unique preparation outputs', len(keys), 'origins', len(origins), flush=True)
    for offset in range(0, len(keys), 32):
        await asyncio.gather(*(one(k, a) for k, a in keys[offset:offset + 32]))
        print('evaluated', min(offset + 32, len(keys)), flush=True)
    args.output.with_suffix('.origins.json').write_text(json.dumps(origins, ensure_ascii=False))
    print('complete', flush=True)


if __name__ == '__main__':
    asyncio.run(main())

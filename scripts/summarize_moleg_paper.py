"""Summarize private raw rows into shareable numeric results."""
from __future__ import annotations
import argparse
from collections import defaultdict,Counter
import hashlib
import json
from pathlib import Path
import statistics
from moleg_paper_metrics import timing,paired_cluster_ci,paired_difference_ci,mcnemar_exact,known_item_metrics,id_recall,means,holm_adjust


def read_rows(path):
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()] if path.exists() else []


def known_error_response(row):
    """Exact fixed error messages in the frozen app, not a quality classifier."""
    messages={
        '응답 생성에 시간이 걸리고 있습니다. 잠시 후 다시 시도해 주세요.':'timeout',
        '일시적인 오류가 발생했습니다. 다시 시도해 주세요.':'generic_error',
        '분석 중 오류가 발생했습니다.':'parse_error',
        '검색 서비스에 일시적 문제가 있습니다.':'search_error',
        '데이터 조회에 실패했습니다.':'database_error',
    }
    return messages.get(((row.get('result') or {}).get('comment') or '').strip())


def summarize_query_anchor(rows, cases):
    groups=defaultdict(dict)
    for row in rows:
        key=row['case_id']
        if key in groups[row['variant']]:raise ValueError('duplicate query-anchor observation')
        groups[row['variant']][key]=row
    result={'rows':len(rows),'expected_rows':len(cases)*4,
        'warning':'Exploratory search-only one-repeat ablation; no intent, guardrail, conversation or answer generation. Not a full API speedup.',
        'variants':{},'comparisons':{}}
    numeric=[]
    for name,records in groups.items():
        values=list(records.values());metrics=[];bykind=defaultdict(list);candidate_coverage=[]
        for r in values:
            case=cases[r['case_id']]
            score=known_item_metrics(r.get('ranked_ids',[]),case)
            coverage=None
            if case.get('source_id') and ('candidate_ids' in r or not r['ok']):
                candidates=r.get('candidate_ids',[])
                coverage={'source_law_in_candidates':float(any(x.split('_')[0]==case['source_law_id'] for x in candidates)),
                    'source_chunk_in_candidates':float(case['source_id'] in candidates) if case.get('reference_answer') else None}
                candidate_coverage.append(coverage)
            if score:
                metrics.append(score);bykind[r['kind']].append(score)
            numeric.append({k:r.get(k) for k in ['case_id','variant','kind','ok','elapsed_s','rerank_s','candidates','error_type']} |
                           {'metrics':score,'candidate_coverage':coverage})
        result['variants'][name]={'n':len(values),'success':sum(r['ok'] for r in values),
            'latency_including_failures':timing([r['elapsed_s'] for r in values]),
            'metrics':means(metrics),'candidate_coverage':means(candidate_coverage),
            'by_kind':{kind:{'n':len(rs),'success':sum(r['ok'] for r in rs),
                'latency':timing([r['elapsed_s'] for r in rs]),'metrics':means(bykind[kind])}
                for kind in sorted({r['kind'] for r in values})
                for rs in [[r for r in values if r['kind']==kind]]}}
    for name,records in groups.items():
        if name=='prepared':continue
        latency=[];quality=[];binary=[];recalls=[];qa_recalls=[];country_matches=[];keyword_jaccards=[];query_exact=[]
        for q,b in groups['prepared'].items():
            c=records.get(q)
            if c is None:continue
            latency.append((b['elapsed_s'],c['elapsed_s']))
            bm=known_item_metrics(b.get('ranked_ids',[]),cases[q])
            cm=known_item_metrics(c.get('ranked_ids',[]),cases[q])
            if bm and bm['chunk_mrr20'] is not None:
                quality.append((bm['chunk_mrr20'],cm['chunk_mrr20']))
                binary.append((bm['chunk_hit1'],cm['chunk_hit1']))
            recall=id_recall(b.get('ranked_ids',[])[:20],c.get('ranked_ids',[])[:20])
            if recall is not None:
                recalls.append(recall)
                if cases[q].get('reference_answer'):qa_recalls.append(recall)
            if b.get('preparation') and c.get('preparation') and 'branches' not in c['preparation']:
                country_matches.append(b.get('country')==c.get('country'))
                bk=set(b['preparation'].get('keywords',[]));ck=set(c['preparation'].get('keywords',[]))
                keyword_jaccards.append(len(bk&ck)/len(bk|ck) if bk|ck else 1.)
                query_exact.append(b['preparation'].get('query')==c['preparation'].get('query'))
        result['comparisons'][name]={'latency':paired_cluster_ci(latency),
            'chunk_mrr20_change':paired_difference_ci(quality),'chunk_top1_mcnemar':mcnemar_exact(binary),
            'document_recall_vs_prepared':statistics.fmean(recalls) if recalls else None,
            'nonempty_reference_pairs':len(recalls),
            'document_recall_vs_prepared_source_qa':statistics.fmean(qa_recalls) if qa_recalls else None,
            'source_qa_nonempty_reference_pairs':len(qa_recalls),
            'country_match_vs_prepared':statistics.fmean(country_matches) if country_matches else None,
            'keyword_jaccard_vs_prepared':statistics.fmean(keyword_jaccards) if keyword_jaccards else None,
            'exact_search_query_vs_prepared':statistics.fmean(query_exact) if query_exact else None}
    adjusted=holm_adjust({k:v['chunk_top1_mcnemar']['p_two_sided'] for k,v in result['comparisons'].items()})
    for name,p in adjusted.items():result['comparisons'][name]['chunk_top1_mcnemar']['p_holm_three_comparisons']=p
    return result,numeric


def summarize_country_ablation(rows,cases):
    if len({r['case_id'] for r in rows})!=len(rows):raise ValueError('duplicate country-ablation observation')
    names=['prepared_country','literal_country_only'];metrics=defaultdict(list);values=defaultdict(list)
    mrr=[];top1=[];latency=[];coverage=defaultdict(list)
    for r in rows:
        case=cases[r['case_id']];pair={}
        for name in names:
            v=r['variants'][name];values[name].append(v)
            score=known_item_metrics(v.get('ranked_ids',[]),case)
            if score:
                metrics[name].append(score)
                coverage[name].append({'source_law_in_candidates':float(any(
                    x.split('_')[0]==case['source_law_id'] for x in v.get('candidate_ids',[]))),
                    'source_chunk_in_candidates':float(case['source_id'] in v.get('candidate_ids',[]))
                        if case.get('reference_answer') else None})
                pair[name]=score
        if case.get('reference_answer'):
            a,b=(pair[n] for n in names)
            mrr.append((a['chunk_mrr20'],b['chunk_mrr20']));top1.append((a['chunk_hit1'],b['chunk_hit1']))
        latency.append(tuple(r['variants'][n]['elapsed_s'] for n in names))
    return {'rows':len(rows),'expected_rows':len(cases),'success':sum(r['ok'] for r in rows),
        'scope':'Exploratory country-only paired fresh search; same frozen query and keywords; no LLM preparation or full API timing.',
        'country_changed_count':sum(r['country_changed'] for r in rows),
        'country_changed_source_questions':sum(r['country_changed'] and bool(cases[r['case_id']].get('source_id')) for r in rows),
        'country_changed_source_qa_questions':sum(r['country_changed'] and bool(cases[r['case_id']].get('reference_answer')) for r in rows),
        'country_changed_case_ids':[r['case_id'] for r in rows if r['country_changed']],
        'variants':{name:{'n':len(values[name]),'success':sum(v['ok'] for v in values[name]),
            'metrics':means(metrics[name]),'candidate_coverage':means(coverage[name]),
            'latency':timing([v['elapsed_s'] for v in values[name]])} for name in names},
        'chunk_mrr20_change':paired_difference_ci(mrr),'chunk_top1_mcnemar':mcnemar_exact(top1),
        'latency':paired_cluster_ci(latency)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',type=Path,required=True)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args();directory=args.directory
    cases={x['case_id']:x for x in json.loads((directory/'cases.json').read_text())}
    summary={'dataset':json.loads((directory/'manifest.json').read_text()),'label_warning':
        'Source-derived silver known-item retrieval; not exhaustive relevance judgments or expert legal accuracy.'}
    extraction=read_rows(directory/'retrieval.extraction.jsonl')
    if extraction:
        groups=defaultdict(dict)
        for r in extraction:groups[r['case_id']][r['repeat']]=r
        pairs=[];exact=[];country=[];keyword=[]
        fields=['country','transformed_query','keywords_original','keywords_transformed']
        for values in groups.values():
            if 0 not in values or 1 not in values:continue
            a,b=values[0]['prep'],values[1]['prep']
            exact.append(all(a.get(k)==b.get(k) for k in fields))
            country.append(a.get('country')==b.get('country'))
            ak=set(a.get('keywords_original',[])+a.get('keywords_transformed',[]))
            bk=set(b.get('keywords_original',[])+b.get('keywords_transformed',[]))
            keyword.append(len(ak&bk)/len(ak|bk) if ak|bk else 1.)
            pairs.append((values[0]['elapsed_s'],values[1]['elapsed_s']))
        summary['extraction_repeat']={'rows':len(extraction),'pairs':len(pairs),
            'failed_preparation':sum(bool(r['prep'].get('early_exit')) for r in extraction),
            'normalized_fields_exact_rate':statistics.fmean(exact) if exact else None,
            'country_match_rate':statistics.fmean(country) if country else None,
            'keyword_jaccard':statistics.fmean(keyword) if keyword else None,
            'latency_repeat_0_vs_1':paired_cluster_ci(pairs)}
    rows=read_rows(directory/'retrieval.jsonl')
    if rows:
        retrieval={};metric_rows=defaultdict(list);query_metrics=defaultdict(lambda:defaultdict(list))
        for name in ['raw','auth_content','metadata','window']:
            times=[];bykind=defaultdict(list)
            for r in rows:
                result=r.get('variants',{}).get(name,{})
                ids=result.get('ranked_ids',[])
                m=known_item_metrics(ids,cases[r['case_id']])
                if m is not None:
                    metric_rows[name].append(m);query_metrics[name][r['case_id']].append(m)
                    bykind[cases[r['case_id']]['kind']].append(m)
                if r['ok']: times.append(r['search_s']+result['rerank_s'])
            retrieval[name]={'metrics':means(metric_rows[name]),'latency':timing(times),
                'by_kind':{k:{'n':len(v),'metrics':means(v)} for k,v in bykind.items()}}
        comparisons={}
        for name in ['auth_content','metadata','window']:
            pairs=[];binary=[]
            for q,base in query_metrics['raw'].items():
                target=query_metrics[name][q]
                if base[0]['chunk_mrr20'] is None:continue
                b=statistics.fmean(x['chunk_mrr20'] for x in base)
                c=statistics.fmean(x['chunk_mrr20'] for x in target)
                pairs.append((b,c))
                # Exact McNemar uses the first repetition once per question.
                binary.append((base[0]['chunk_hit1'],target[0]['chunk_hit1']))
            comparisons[name]={'chunk_mrr20_change':paired_difference_ci(pairs),
                'chunk_top1_mcnemar_first_repeat':mcnemar_exact(binary)}
        adjusted=holm_adjust({n:r['chunk_top1_mcnemar_first_repeat']['p_two_sided'] for n,r in comparisons.items()})
        for name,p in adjusted.items():
            comparisons[name]['chunk_top1_mcnemar_first_repeat']['p_holm_three_reranker_comparisons']=p
        summary['retrieval']={'rows':len(rows),'success':sum(r['ok'] for r in rows),
            'chunk_eligible_questions':sum(bool(c.get('reference_answer')) for c in cases.values()),
            'expected_rows':len(cases)*3,'variants':retrieval,'comparisons':comparisons}
        def top1_relation(ids,case):
            if not ids:return 'empty'
            if ids[0]==case['source_id']:return 'source_chunk'
            if ids[0].split('_')[0]==case['source_law_id']:return 'same_law_different_chunk'
            return 'different_law'
        shapes=defaultdict(Counter);transitions=Counter();transition_cases=defaultdict(list)
        for r in rows:
            case=cases[r['case_id']]
            if r['repeat']!=0 or not case.get('reference_answer'):continue
            relations={name:top1_relation(v.get('ranked_ids',[]),case) for name,v in r['variants'].items()}
            for name,relation in relations.items():shapes[name][relation]+=1
            if 'raw' in relations and 'auth_content' in relations:
                transition=relations['raw']+' -> '+relations['auth_content']
                transitions[transition]+=1;transition_cases[transition].append(r['case_id'])
        summary['retrieval']['first_repeat_top1_relation_to_source']={k:dict(v) for k,v in shapes.items()}
        summary['retrieval']['raw_to_auth_content_first_repeat']={'counts':dict(transitions),'case_ids':dict(transition_cases),
            'warning':'Relation to the one known source, not exhaustive relevance or legal error adjudication.'}
    for filename in ['e2e_stream.jsonl','e2e_json.jsonl']:
        rows=read_rows(directory/filename)
        if not rows:continue
        groups=defaultdict(list)
        for r in rows:groups[r['variant']].append(r)
        result={'rows':len(rows),'variants':{},'comparisons':{}}
        queries=defaultdict(lambda:defaultdict(list))
        numeric=[]
        for name,records in groups.items():
            metrics=[];kinds={}
            for r in records:
                queries[name][r['case_id']].append(r)
                case=cases[r['case_id']]
                data=r.get('result') or {}
                laws=data.get('laws') or []
                ids=[str(x.get('item_id') or x.get('id') or '') for x in laws]
                m=known_item_metrics(ids,case)
                if m:
                    metrics.append({'source_law_in_returned_evidence':float(any(x.split('_')[0]==case['source_law_id'] for x in ids)),
                        'source_chunk_in_returned_evidence':float(case['source_id'] in ids) if case.get('reference_answer') else None})
                numeric.append({k:r.get(k) for k in ['case_id','kind','variant','repeat','ok','status',
                    'elapsed_s','ttft_s','done_s','streamed_chars','error_type']} |
                    {'returned_laws':len(laws),'answer_chars':len(data.get('comment') or ''),
                     'known_error_response':known_error_response(r),
                     'hitl':bool(data.get('hitl'))})
            for kind in sorted({r['kind'] for r in records}):
                selected=[r for r in records if r['kind']==kind]
                kinds[kind]={'n':len(selected),'success':sum(r['ok'] for r in selected),
                    'latency':timing([r['elapsed_s'] for r in selected]),
                    'ttft':timing([r.get('ttft_s') for r in selected])}
            result['variants'][name]={'n':len(records),'success':sum(r['ok'] for r in records),
                'latency_including_failures':timing([r['elapsed_s'] for r in records]),
                'ttft':timing([r.get('ttft_s') for r in records]),
                'known_item_in_returned_evidence':means(metrics),'by_kind':kinds,
                'mean_answer_chars':statistics.fmean(len((r.get('result') or {}).get('comment') or '') for r in records),
                'hitl_count':sum(bool((r.get('result') or {}).get('hitl')) for r in records),
                'empty_answer_count':sum(not (r.get('result') or {}).get('comment') for r in records),
                'known_error_response_counts':dict(Counter(known_error_response(r) for r in records if known_error_response(r))),
                'success_excluding_known_error_response':sum(r['ok'] and not known_error_response(r) for r in records),
                'by_repeat':{str(rep):timing([r['elapsed_s'] for r in records if r['repeat']==rep])
                    for rep in sorted({r['repeat'] for r in records})}}
        for name in groups:
            if name=='baseline':continue
            pairs=[];ttft=[];recalls=[];exact=[];source_pairs=[];source_binary=[];chunk_pairs=[]
            for q,baselines in queries['baseline'].items():
                candidates=queries[name].get(q,[])
                if not candidates:continue
                pairs.append((statistics.fmean(r['elapsed_s'] for r in baselines),
                              statistics.fmean(r['elapsed_s'] for r in candidates)))
                bt=[r['ttft_s'] for r in baselines if r.get('ttft_s') is not None]
                ct=[r['ttft_s'] for r in candidates if r.get('ttft_s') is not None]
                if bt and ct:ttft.append((statistics.fmean(bt),statistics.fmean(ct)))
                case=cases[q]
                if case.get('source_id'):
                    def included(row,chunk=False):
                        ids=[str(x.get('item_id') or x.get('id') or '') for x in (row.get('result') or {}).get('laws',[])]
                        return float(case['source_id'] in ids) if chunk else float(any(x.split('_')[0]==case['source_law_id'] for x in ids))
                    source_pairs.append((statistics.fmean(included(r) for r in baselines),
                                         statistics.fmean(included(r) for r in candidates)))
                    b0=next((r for r in baselines if r['repeat']==0),None)
                    c0=next((r for r in candidates if r['repeat']==0),None)
                    if b0 and c0:source_binary.append((included(b0),included(c0)))
                    if case.get('reference_answer'):
                        chunk_pairs.append((statistics.fmean(included(r,True) for r in baselines),
                                            statistics.fmean(included(r,True) for r in candidates)))
                for b in baselines:
                    c=next((x for x in candidates if x['repeat']==b['repeat']),None)
                    if c is None:continue
                    def ids(row):return [str(x.get('item_id') or x.get('id') or '') for x in (row.get('result') or {}).get('laws',[])]
                    a,z=ids(b),ids(c);rec=id_recall(a,z)
                    if rec is not None:recalls.append(rec);exact.append(a==z)
            result['comparisons'][name]={'latency':paired_cluster_ci(pairs),'ttft':paired_cluster_ci(ttft),
                'source_law_in_returned_evidence_change':paired_difference_ci(source_pairs),
                'source_chunk_in_returned_evidence_change':paired_difference_ci(chunk_pairs),
                'source_law_mcnemar_first_repeat':mcnemar_exact(source_binary),
                'document_recall_vs_baseline':statistics.fmean(recalls) if recalls else None,
                'exact_returned_ids':statistics.fmean(exact) if exact else None,
                'nonempty_reference_pairs':len(recalls)}
        # Repeated-control floor for observable document and answer variability.
        repeat_recalls=[];answer_exact=[]
        for q,records in queries['baseline'].items():
            if len(records)<2:continue
            b,c=sorted(records,key=lambda x:x['repeat'])[:2]
            a=[x.get('item_id') for x in (b.get('result') or {}).get('laws',[])]
            z=[x.get('item_id') for x in (c.get('result') or {}).get('laws',[])]
            rec=id_recall(a,z)
            if rec is not None:repeat_recalls.append(rec)
            answer_exact.append((b.get('result') or {}).get('comment')==(c.get('result') or {}).get('comment'))
        result['baseline_repeat']={'pairs':len(answer_exact),'document_recall':statistics.fmean(repeat_recalls) if repeat_recalls else None,
            'exact_answer':statistics.fmean(answer_exact) if answer_exact else None}
        summary[filename.removesuffix('.jsonl')]=result
        sessions={r['session_id'] for r in rows}
        traces=[t for name in groups for t in read_rows(directory/f'{name}.trace.jsonl') if t['session_id'] in sessions]
        trace_summary={}
        for name in groups:
            selected=[t for t in traces if t['profile']==name]
            models=defaultdict(list)
            for t in selected:
                for call in t.get('llm',[]):models[str(call.get('model'))].append(call)
            trace_summary[name]={'requests_with_trace':len(selected),
                'overlap_consumed':sum(len(t.get('overlap',[])) for t in selected),
                'recorded_nonstream_llm_errors':sum(len(t.get('errors',[])) for t in selected),
                'recorded_nonstream_error_types':dict(Counter(e.get('error','unknown') for t in selected for e in t.get('errors',[]))),
                'requests_with_recorded_nonstream_errors':sum(bool(t.get('errors')) for t in selected),
                'stream_calls_without_visible_output':sum(bool(c.get('stream')) and not c.get('characters')
                    for t in selected for c in t.get('llm',[])),
                'error_instrumentation_scope':'Nonstream exceptions are counted; HTTP/done and answer quality are separate. Stream setup failures are not exhaustively traced.',
                'rerank_statuses':dict(Counter(str(c['status']) for t in selected for c in t.get('rerank',[]))),
                'guardrail_statuses':dict(Counter(str(c['status']) for t in selected for c in t.get('guardrail',[]))),
                'models':{m:{'calls':len(cs),'latency':timing([c['elapsed_s'] for c in cs]),
                    'stream_calls':sum(bool(c.get('stream')) for c in cs),
                    'finish_reasons_nonstream':dict(Counter(str(c.get('finish_reason')) for c in cs if not c.get('stream'))),
                    'summed_call_wall_s_not_critical_path':sum(c['elapsed_s'] for c in cs),
                    'stream_reasoning_characters':sum(c.get('reasoning_characters',0) for c in cs),
                    'nonstream_prompt_tokens':sum((c.get('usage') or {}).get('prompt_tokens',0) or 0 for c in cs),
                    'nonstream_cached_prompt_tokens':sum(((c.get('usage') or {}).get('prompt_tokens_details') or {}).get('cached_tokens',0) or 0 for c in cs),
                    'nonstream_completion_tokens':sum((c.get('usage') or {}).get('completion_tokens',0) or 0 for c in cs)}
                          for m,cs in models.items()}}
        summary[filename.removesuffix('.jsonl')]['trace']=trace_summary
        trace_by_session={t['session_id']:t for t in traces}
        result['slowest_requests']=[
            {k:r.get(k) for k in ['case_id','variant','repeat','elapsed_s','ttft_s','ok']} |
            {'known_error_response':known_error_response(r),
             'calls':[{k:c.get(k) for k in ['model','elapsed_s','stream','characters','reasoning_characters','finish_reason']}
                      for c in trace_by_session.get(r['session_id'],{}).get('llm',[])]}
            for r in sorted(rows,key=lambda r:r['elapsed_s'],reverse=True)[:10]]
        args.output.parent.mkdir(parents=True,exist_ok=True)
        args.output.with_name(filename.replace('.jsonl','.metrics.jsonl')).write_text(
            ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in numeric))
    rows=read_rows(directory/'country_ablation.jsonl')
    if rows:summary['country_ablation']=summarize_country_ablation(rows,cases)
    rows=read_rows(directory/'query_anchor.jsonl')
    if rows:
        summary['query_anchor'],numeric=summarize_query_anchor(rows,cases)
        args.output.with_name('query_anchor.metrics.jsonl').write_text(
            ''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in numeric))
    rows=read_rows(directory/'rerank_clipping.jsonl')
    if rows:
        values=defaultdict(list);pairs=[];exact=[];hit_pairs=[]
        for r in rows:
            case=cases[r['case_id']]
            for name,variant in r.get('variants',{}).items():
                score=known_item_metrics(variant.get('ranked_ids',[]),case)
                if score:values[name].append(score)
            variants=r.get('variants',{})
            if all(n in variants for n in ['auth_content','strict_auth_content']):
                a=variants['auth_content']['ranked_ids'];b=variants['strict_auth_content']['ranked_ids']
                exact.append(a==b)
                am=known_item_metrics(a,case);bm=known_item_metrics(b,case)
                if am and am['chunk_mrr20'] is not None:
                    pairs.append((am['chunk_mrr20'],bm['chunk_mrr20']))
                    hit_pairs.append((am['chunk_hit1'],bm['chunk_hit1']))
        summary['rerank_clipping']={'rows':len(rows),'expected_rows':len(cases),
            'success':sum(r['ok'] for r in rows),'scope':'exploratory one-repeat truncation sensitivity on same fresh candidate sets',
            'metrics':{n:means(ms) for n,ms in values.items()},
            'exact_ranking_fraction':statistics.fmean(exact) if exact else None,
            'strict_minus_prefix_mrr':paired_difference_ci(pairs),'top1_mcnemar':mcnemar_exact(hit_pairs)}
    for filename in ['country_resolution.json','guardrail_mapping.json','answer_judge_summary.json','generation_replay_summary.json','http_application_to_gpu.json','http_loopback.json',
                     'ngram_current_compat.json','ngram_upgrade_compat.json']:
        path=directory/filename
        if path.exists():
            value=json.loads(path.read_text())
            summary[filename.removesuffix('.json')]={k:v for k,v in value.items() if k!='rows' or not isinstance(v,list)}
    path=directory/'stream_replay.json'
    if path.exists():
        replay=json.loads(path.read_text());rows=replay['rows']
        summary['stream_replay']={'n':len(rows),'scope':replay['scope'],
            'exact_both':sum(r['paced']['exact_text'] and r['immediate']['exact_text'] for r in rows),
            'paced':timing([r['paced']['elapsed_s'] for r in rows]),
            'immediate':timing([r['immediate']['elapsed_s'] for r in rows]),
            'paired_latency':paired_cluster_ci([(r['paced']['elapsed_s'],r['immediate']['elapsed_s']) for r in rows]),
            'note':'Emitter-only wall time; not additive attribution or measured full-API speedup.'}
    summary['artifact_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest()
        for p in directory.glob('*.jsonl') if p.name in ['retrieval.jsonl','e2e_stream.jsonl','e2e_json.jsonl',
            'retrieval.extraction.jsonl','country_ablation.jsonl',
            'query_anchor.jsonl','answer_judgments.jsonl','generation_replay.jsonl','rerank_clipping.jsonl']}
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps({k: {'rows':v.get('rows'),'success':v.get('success')} for k,v in summary.items()
        if isinstance(v,dict) and isinstance(v.get('rows'),int)},ensure_ascii=False))


if __name__=='__main__':main()

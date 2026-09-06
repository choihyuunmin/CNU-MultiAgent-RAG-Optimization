"""Audit and summarize request-level infrastructure experiments, without gold claims."""
from __future__ import annotations
import argparse
from collections import defaultdict, Counter
import json
import math
from pathlib import Path
import random
import statistics
from moleg_paper_metrics import timing, percentile, paired_cluster_ci


def clustered_ratio(clusters, resamples=4000):
    """Each triple is (baseline sum, candidate sum, number of requests in block)."""
    if not clusters:
        return {'n_clusters':0}
    def estimate(items):
        b=sum(x[0] for x in items); c=sum(x[1] for x in items); n=sum(x[2] for x in items)
        return b/n, c/n, 100*(1-c/b), (c-b)/n
    b,c,reduction,delta=estimate(clusters)
    rng=random.Random(20260906)
    draws=[estimate(rng.choices(clusters,k=len(clusters))) for _ in range(resamples)]
    return {'n_clusters':len(clusters),'n_requests':sum(x[2] for x in clusters),
            'baseline_mean_s':b,'candidate_mean_s':c,'speedup':b/c,'reduction_pct':reduction,
            'reduction_95ci_pct':[percentile([x[2] for x in draws],p) for p in [.025,.975]],
            'delta_s':delta,'delta_95ci_s':[percentile([x[3] for x in draws],p) for p in [.025,.975]],
            'bootstrap_resamples':resamples,'unit':'paired request burst; request-count weighted'}


def summarize(rows, blocks, manifest, traces=None, allow_partial=False):
    keys=[(r['case_id'],r['stage'],r['variant'],r['repeat']) for r in rows]
    if len(keys)!=len(set(keys)):
        raise ValueError('duplicate request observations')
    expected={(i['case_id'],i['stage'],v,rep) for i in manifest['inputs']
              for v in manifest['variants'] for rep in range(manifest['repeats'])}
    missing=expected-set(keys); extra=set(keys)-expected
    if extra or (missing and not allow_partial):
        raise ValueError(f'invalid request keys: missing={len(missing)} extra={len(extra)}')
    inputs={(i['case_id'],i['stage']):i for i in manifest['inputs']}
    for r in rows:
        target=inputs[(r['case_id'],r['stage'])]
        if r['payload_sha256']!=target['payload_sha256'] or r['question_sha256']!=target['question_sha256']:
            raise ValueError('changed frozen input')
    blockkeys=[(b['variant'],b['repeat'],b['block']) for b in blocks]
    if len(blockkeys)!=len(set(blockkeys)):
        raise ValueError('duplicate burst observations')
    expected_blocks={(v,rep,b) for v in manifest['variants'] for rep in range(manifest['repeats'])
                     for b in range(math.ceil(len(manifest['inputs'])/manifest['burst']))}
    if set(blockkeys)-expected_blocks or (expected_blocks-set(blockkeys) and not allow_partial):
        raise ValueError('missing or unexpected burst records')
    groups=Counter((r['variant'],r['repeat'],r['block']) for r in rows)
    if any(groups[(b['variant'],b['repeat'],b['block'])]!=b['n'] for b in blocks):
        raise ValueError('burst counts do not match request records')
    byvariant=defaultdict(list)
    for r in rows:byvariant[r['variant']].append(r)
    result={'scope':manifest['scope'],'rows':len(rows),'expected_rows':len(expected),
            'missing':len(missing),'extra':len(extra),'complete':not missing and not extra,
            'unique_questions':len(set(r['case_id'] for r in rows)), 'variants':{},'comparisons':{},
            'warning':'output agreement is not expert accuracy; these are not full RAG requests'}
    def group_stats(values):
        return {'n':len(values),'ok':sum(r['ok'] for r in values),
                'valid_json':sum(r.get('valid_json',False) for r in values),
                'errors':dict(Counter(r.get('error_type') for r in values if not r['ok'])),
                'finish_reasons':dict(Counter(r.get('finish_reason','missing') for r in values)),
                'latency':timing([r['elapsed_s'] for r in values]),
                'mean_prompt_tokens':statistics.fmean(r['usage']['prompt_tokens'] for r in values if r.get('usage')) if any(r.get('usage') for r in values) else None,
                'mean_completion_tokens':statistics.fmean(r['usage']['completion_tokens'] for r in values if r.get('usage')) if any(r.get('usage') for r in values) else None,
                'mean_reasoning_chars':statistics.fmean(r.get('reasoning_chars',0) for r in values)}
    for variant, values in byvariant.items():
        entry=group_stats(values)
        entry['stages']={s:group_stats([r for r in values if r['stage']==s]) for s in sorted(set(r['stage'] for r in values))}
        all_bs=[b for b in blocks if b['variant']==variant]
        bs=[b for b in all_bs if b['wall_s'] is not None]
        entry['bursts_with_missing_wall']=len(all_bs)-len(bs)
        entry['burst_wall']=timing([b['wall_s'] for b in bs])
        entry['burst_completion_rate_per_s']=sum(b['n'] for b in bs)/sum(b['wall_s'] for b in bs) if bs else None
        telemetry=[]
        for b in all_bs:
            before=b.get('counters_before',{});after=b.get('counters_after',{})
            names=['request_success_total','prefix_cache_queries_total','prefix_cache_hits_total','num_preemptions_total']
            if all('vllm:'+name in before and 'vllm:'+name in after for name in names):
                delta={name:after['vllm:'+name]-before['vllm:'+name] for name in names}
                if any(value<0 for value in delta.values()):continue
                telemetry.append((b,delta))
        cache_queries=sum(d['prefix_cache_queries_total'] for b,d in telemetry)
        entry['telemetry']={'observed_bursts':len(telemetry),
            'success_counter_matches_experiment_burst':sum(d['request_success_total']==b['ok'] for b,d in telemetry),
            'preemptions_delta':sum(d['num_preemptions_total'] for b,d in telemetry),
            'prefix_cache_token_hit_fraction':sum(d['prefix_cache_hits_total'] for b,d in telemetry)/cache_queries if cache_queries else None,
            'limitation':'sampled server counters, not proof of exclusive GPU use'}
        if variant!='proxy':
            entry['gateway_wait']=timing([r.get('gateway_wait_s') for r in values])
            entry['upstream']=timing([r.get('upstream_s') for r in values])
            entry['outside_wait_and_upstream']=timing([r['elapsed_s']-r['gateway_wait_s']-r['upstream_s'] for r in values if r.get('ok')])
        byquery=defaultdict(dict)
        for r in values:byquery[(r['case_id'],r['stage'])][r['repeat']]=r
        pairs=[(rs[0],rs[1]) for rs in byquery.values() if all(x in rs and rs[x]['ok'] for x in [0,1])]
        entry['repeat_agreement']={'pairs':len(pairs),'normalized_exact':statistics.fmean(a['normalized_sha256']==b['normalized_sha256'] for a,b in pairs) if pairs else None}
        result['variants'][variant]=entry
    configs=[('proxy','pass'),('pass','fifo8'),('pass','fair8'),('fifo8','fair8')]
    for base,candidate in configs:
        if base not in byvariant or candidate not in byvariant:continue
        left={(r['case_id'],r['stage'],r['repeat']):r for r in byvariant[base]}
        right={(r['case_id'],r['stage'],r['repeat']):r for r in byvariant[candidate]}
        matched=[(left[k],right[k]) for k in sorted(left.keys()&right.keys())]
        comparison={}
        for stage in ['all','classify','prepare']:
            pairs=[(a,b) for a,b in matched if stage=='all' or a['stage']==stage]
            grouped=defaultdict(list)
            for a,b in pairs:
                if (a['repeat'],a['block'])!=(b['repeat'],b['block']):raise ValueError('unpaired burst composition')
                grouped[(a['repeat'],a['block'])].append((a,b))
            clusters=[(sum(a['elapsed_s'] for a,b in ps),sum(b['elapsed_s'] for a,b in ps),len(ps)) for ps in grouped.values()]
            sub=clustered_ratio(clusters)
            byq=defaultdict(list)
            for a,b in pairs:byq[a['case_id']].append((a['elapsed_s'],b['elapsed_s']))
            sub['question_cluster_sensitivity']=paired_cluster_ci([(statistics.fmean(x[0] for x in ps),statistics.fmean(x[1] for x in ps)) for ps in byq.values()])
            valid=[(a,b) for a,b in pairs if a['ok'] and b['ok']]
            sub['output_pairs']=len(valid)
            sub['normalized_output_agreement']=statistics.fmean(a['normalized_sha256']==b['normalized_sha256'] for a,b in valid) if valid else None
            usage_pairs=[(a,b) for a,b in valid if a.get('usage') and b.get('usage')]
            sub['completion_token_count_agreement']=statistics.fmean(a['usage']['completion_tokens']==b['usage']['completion_tokens'] for a,b in usage_pairs) if usage_pairs else None
            comparison[stage]=sub
        lb={(b['repeat'],b['block']):b for b in blocks if b['variant']==base}
        rb={(b['repeat'],b['block']):b for b in blocks if b['variant']==candidate}
        comparison['burst_completion']=clustered_ratio([(lb[k]['wall_s'],rb[k]['wall_s'],1) for k in sorted(lb.keys()&rb.keys()) if lb[k]['wall_s'] is not None and rb[k]['wall_s'] is not None])
        result['comparisons'][base+'_vs_'+candidate]=comparison
    if traces is not None:
        relevant={r['request_id']:r for r in rows if r['variant']!='proxy'}
        byid=defaultdict(list)
        for t in traces:
            if t.get('request_id') in relevant:byid[t['request_id']].append(t)
        for rid,r in relevant.items():
            if len(byid[rid])!=1:raise ValueError('missing or duplicate gateway trace')
            t=byid[rid][0]
            if t['input_sha256']!=r['wire_sha256'] or t['forward_sha256']!=r['wire_sha256']:
                raise ValueError('gateway request body changed')
            if r['ok'] and (t['response_sha256']!=r['response_sha256'] or t['stage']!=r['stage']):
                raise ValueError('response bytes or stage attribution changed')
        result['gateway_audit']={'requests':len(relevant),'exact_input_forwarding':True,'exact_response_forwarding':True}
    return result


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--rows',type=Path,required=True)
    ap.add_argument('--traces',type=Path)
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--allow-partial',action='store_true')
    args=ap.parse_args()
    rows=[json.loads(x) for x in args.rows.read_text().splitlines()]
    blocks=[json.loads(x) for x in args.rows.with_suffix('.blocks.jsonl').read_text().splitlines()]
    manifest=json.loads(args.rows.with_suffix('.manifest.json').read_text())
    traces=[json.loads(x) for x in args.traces.read_text().splitlines()] if args.traces else None
    result=summarize(rows,blocks,manifest,traces,args.allow_partial)
    args.output.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    fields=['case_id','kind','stage','variant','repeat','block','ok','status','elapsed_s','gateway_wait_s','upstream_s',
            'valid_json','finish_reason','normalized_sha256','question_sha256','payload_sha256','wire_sha256','response_sha256','reasoning_chars','usage','error_type']
    with args.output.with_suffix('.metrics.jsonl').open('w') as sink:
        for r in rows:sink.write(json.dumps({k:r.get(k) for k in fields})+'\n')
    print(json.dumps({'rows':result['rows'],'complete':result['complete'],'variants':{
        name:{'mean_s':r['latency']['mean_s'],'ok':r['ok'],'n':r['n']} for name,r in result['variants'].items()}}))


if __name__=='__main__':main()

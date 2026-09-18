"""Audit response completion against recorded stage outcomes, without raw text export.

A successful SSE response is not evidence that dependencies succeeded. This
analysis preserves the original counters and produces additional, stricter ones.
No-error-observed is not an expert correctness label.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

from moleg_scaling_metrics import describe
from moleg_paper_metrics import paired_cluster_ci

EMPTY_PREPARATION = hashlib.sha256(json.dumps(dict.fromkeys([
    'country', 'transformed_query', 'keywords_original', 'keywords_transformed'
]), sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def classify(pipeline_ok, trace, workflow):
    if trace is None or workflow is None:
        return {'trace_complete': False, 'stage_clean_completion': False,
                'dependency_clean_completion': False, 'preparation_failed': None,
                'failed_model_calls': None, 'dependency_statuses': {}}
    failed = [s for s in workflow.get('spans', [])
              if s.get('kind') == 'model_rpc_lifetime' and s.get('success') is False]
    explicit = trace.get('dependency_failure', [])
    # Legacy traces require BOTH a failed preparation RPC and the fingerprinted
    # fallback's empty output; an empty legitimate retrieval alone is not failure.
    prep_failed = (any(e.get('component') == 'preparation' for e in explicit)
                   or (any(s.get('stage') == 'preparation' for s in failed)
                       and any(p.get('output_sha256') == EMPTY_PREPARATION
                               for p in trace.get('preparation', []))))
    statuses = Counter(f'{kind}:{item.get("status", "missing")}'
                       for kind in ('guardrail', 'rerank') for item in trace.get(kind, []))
    dependency_bad = any(item.get('status') != 200
                         for kind in ('guardrail', 'rerank') for item in trace.get(kind, []))
    # Successful recovery remains a completed response but is explicitly degraded.
    stage_clean = bool(pipeline_ok and not prep_failed and not failed
                       and not trace.get('errors') and not trace.get('model_attempt_failure'))
    return {'trace_complete': True, 'stage_clean_completion': stage_clean,
            'dependency_clean_completion': stage_clean and not dependency_bad and not explicit,
            'preparation_failed': bool(prep_failed), 'failed_model_calls': len(failed),
            'failed_model_stages': dict(Counter(s.get('stage', 'unknown') for s in failed)),
            'dependency_statuses': dict(statuses)}


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


def analyze(root):
    summary = json.loads((root/'summary.json').read_text())
    public = read_rows(root/'requests.jsonl')
    private = read_rows(root/'private-responses.jsonl')
    key = lambda r: (r['trial'], r['case_id'], r['index'])
    raw = {key(r): r for r in private}
    if len(raw) != len(private) or {key(r) for r in public} != set(raw):
        raise ValueError('duplicate or missing private/public request join')
    arms = sorted({r['arm'] for r in public})
    traces, workflows = {}, {}
    for arm in arms:
        ts = read_rows(root/'apps'/f'{arm}.trace.jsonl')
        ws = read_rows(root/'apps'/f'{arm}.workflow.jsonl')
        traces[arm] = {t['session_id']: t for t in ts}
        workflows[arm] = {t['trace_id']: t for t in ws}
        if len(traces[arm]) != len(ts) or len(workflows[arm]) != len(ws):
            raise ValueError('duplicate trace identities')
    audited = []
    dispatch = defaultdict(list)
    for row in public:
        arm = row['arm']
        sid = (raw[key(row)].get('result') or {}).get('session_id')
        trace = traces[arm].get(sid)
        workflow = workflows[arm].get(trace.get('workflow_trace_id')) if trace else None
        outcome = classify(row.get('pipeline_ok', False), trace, workflow)
        # Only explicitly allowed numeric/hash identifiers are exported.
        audited.append({k: row.get(k) for k in ('trial','arm','repeat','level','case_id',
                        'index','question_sha256','elapsed_s','ok','pipeline_ok','source_law_hit')}
                       | outcome)
        for event in (trace or {}).get('dispatch', []):
            if event.get('event') == 'release':
                dispatch[(row['level'], arm)].append(event)
    loads = []
    for level in sorted({r['level'] for r in public}):
        entries = {}
        for arm in arms:
            rows = [r for r in audited if r['level']==level and r['arm']==arm]
            trials = [t for t in summary['trials'] if t['level']==level and t['arm']==arm]
            wall = sum(t['wall_s'] for t in trials)
            status = Counter()
            failed_stages = Counter()
            for r in rows:
                status.update(r['dependency_statuses'])
                failed_stages.update(r.get('failed_model_stages', {}))
            events = dispatch[(level,arm)]
            entries[arm] = {'n':len(rows), 'wall_s':wall,
                **{field:sum(bool(r.get(field)) for r in rows) for field in
                   ['ok','pipeline_ok','trace_complete','stage_clean_completion',
                    'dependency_clean_completion','preparation_failed']},
                'completed_with_missing_trace':sum(r['pipeline_ok'] and not r['trace_complete'] for r in rows),
                'latency_including_failures':describe([r['elapsed_s'] for r in rows]),
                'stage_clean_throughput_rps':sum(r['stage_clean_completion'] for r in rows)/wall if wall else None,
                'dependency_clean_throughput_rps':sum(r['dependency_clean_completion'] for r in rows)/wall if wall else None,
                'source_law_hit':describe([float(r['source_law_hit']) if r['source_law_hit'] is not None else None for r in rows]),
                'dependency_statuses':dict(status), 'failed_model_stages':dict(failed_stages),
                'dispatch_wait_s':describe([e.get('wait_s') for e in events]),
                'dispatch_service_s':describe([e.get('service_s') for e in events]),
                'dispatch_calls':len(events)}
        comparisons = {}
        if 'baseline' in arms:
            for arm in arms:
                if arm=='baseline': continue
                bs={(r['case_id'],r['repeat']):r for r in audited if r['level']==level and r['arm']=='baseline'}
                xs={(r['case_id'],r['repeat']):r for r in audited if r['level']==level and r['arm']==arm}
                if set(bs)!=set(xs):
                    comparisons[arm]={'error':'incomplete paired question set'};continue
                by_case=defaultdict(list)
                for k,b in bs.items():
                    a=xs[k]
                    if a['question_sha256']!=b['question_sha256']:raise ValueError('question mismatch')
                    by_case[k[0]].append((b['elapsed_s'],a['elapsed_s']))
                comparisons[arm]=paired_cluster_ci([(statistics.fmean(b for b,a in v),statistics.fmean(a for b,a in v)) for v in by_case.values()])
        loads.append({'level':level,'arms':entries,'paired_latency_vs_baseline':comparisons})
    errors=[]
    if any(r['pipeline_ok'] and not r['trace_complete'] for r in audited):errors.append('completed response missing trace')
    return {'complete':summary['complete'],'n':len(audited),'audit_errors':errors,
            'scope':'trace-based dependency health, not expert answer correctness; timeouts retained',
            'loads':loads, 'input_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
             for p in [root/'requests.jsonl',root/'private-responses.jsonl',root/'summary.json']
             +sorted((root/'apps').glob('*.trace.jsonl'))+sorted((root/'apps').glob('*.workflow.jsonl'))}}, audited


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result,rows=analyze(args.directory)
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/'stage-outcomes.json').write_text(json.dumps(result,indent=2)+'\n')
    (args.output/'audited-requests.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    print(json.dumps({'n':result['n'],'complete':result['complete'],'audit_errors':result['audit_errors']}))

if __name__=='__main__':main()

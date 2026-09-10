"""Pair the actual adapter trials, including the incremental fixed16 comparison.

Optional private traces are joined via server-generated correlation IDs, then
exported only as stage/model aggregate numbers. No raw texts/identifiers leak.
"""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from moleg_paper_metrics import paired_cluster_ci, paired_difference_ci
from moleg_scaling_metrics import describe
from summarize_moleg_scaling import aggregate


def paired_comparisons(rows):
    grouped = defaultdict(dict)
    for row in rows:
        key = (row['case_id'], row['repeat'])
        group = grouped[(row['users'], row['policy'])]
        if key in group:
            raise ValueError('duplicate question/repeat observation')
        group[key] = row
    comparisons = []
    for users in sorted({r['users'] for r in rows}):
        for control, candidate in [('baseline','fixed16'), ('baseline','budget'), ('fixed16','budget')]:
            before, after = grouped[(users, control)], grouped[(users, candidate)]
            if not before or set(before) != set(after):
                raise ValueError('incomplete paired comparison')
            by_question = defaultdict(list)
            exact = []
            for key in sorted(before):
                b, a = before[key], after[key]
                if b['question_sha256'] != a['question_sha256']:
                    raise ValueError('paired questions differ')
                by_question[key[0]].append((b, a))
                exact.append(b['ok'] and a['ok'] and b['evidence_ids_sha256']==a['evidence_ids_sha256'])
            latency = [(statistics.fmean(b['elapsed_s'] for b,a in pairs),
                        statistics.fmean(a['elapsed_s'] for b,a in pairs)) for pairs in by_question.values()]
            entry = {'users':users, 'control':control, 'candidate':candidate,
                     'latency':paired_cluster_ci(latency),
                     'evidence_list_exact_fraction':statistics.fmean(exact),
                     'paired_requests':len(before)}
            for field in ['source_law_hit','source_chunk_hit']:
                pairs = []
                for observations in by_question.values():
                    bs = [float(b[field]) for b,a in observations if b[field] is not None]
                    cs = [float(a[field]) for b,a in observations if a[field] is not None]
                    if bs and cs:
                        pairs.append((statistics.fmean(bs), statistics.fmean(cs)))
                entry[field + '_change'] = paired_difference_ci(pairs)
            comparisons.append(entry)
    return comparisons


def workflow_summary(directory, workflow_directory):
    state = json.loads((directory/'summary.json').read_text())
    app_states = [json.loads(line) for line in (directory/'app-state.jsonl').open()]
    output = []
    for policy in ['baseline','fixed16','budget']:
        trials = [t for t in state['trials'] if t['policy']==policy]
        groups = {}
        for line in (workflow_directory/(policy+'.trace.jsonl')).open():
            row = json.loads(line)
            prefix, index = row['session_id'].rsplit('-',1)
            groups.setdefault(prefix, []).append(row)
        workflow = {}
        for line in (workflow_directory/(policy+'.workflow.jsonl')).open():
            record = json.loads(line)
            if record['trace_id'] in workflow:
                raise ValueError('duplicate workflow trace identity')
            workflow[record['trace_id']] = record
        if len(groups) != len(trials):
            raise ValueError('private trace groups do not match trial order')
        joined_ids = set()
        for trial, records in zip(trials, groups.values(), strict=True):
            if len(records) != trial['n']:
                raise ValueError('incomplete private execute trace')
            traces = [workflow[r['workflow_trace_id']] for r in records]
            if len({r['workflow_trace_id'] for r in records}) != len(records):
                raise ValueError('duplicate private-to-workflow join')
            joined_ids.update(r['workflow_trace_id'] for r in records)
            spans = [s for r in traces for s in r['spans']]
            stages = sorted({(s['stage'], s['model']) for s in spans if s['kind']=='model_rpc_lifetime'})
            stage_rows = []
            for stage, model in stages:
                per_request = []
                waits = []
                for trace in traces:
                    per_request.append(sum(s['end_s']-s['start_s'] for s in trace['spans']
                                           if s['kind']=='model_rpc_lifetime' and s['stage']==stage and s['model']==model))
                    waits.append(sum(s['end_s']-s['start_s'] for s in trace['spans']
                                     if s['kind']=='adapter_wait' and s['stage']==stage and s['model']==model))
                stage_rows.append({'stage':stage, 'model':model, 'model_work_per_api_s':describe(per_request),
                                   'adapter_wait_work_per_api_s':describe(waits)})
            usage = [s for s in spans if s['kind']=='usage' and s['model']=='orchestrator']
            model_rows = []
            for model in sorted({s['model'] for s in spans if s['kind']=='model_rpc_lifetime'}):
                calls = [s for s in spans if s['kind']=='model_rpc_lifetime' and s['model']==model]
                tokens = [s for s in spans if s['kind']=='usage' and s['model']==model]
                reasoning = [s for s in spans if s['kind']=='reasoning_size' and s['model']==model]
                model_rows.append({'model':model, 'calls':len(calls),
                    'unclassified_calls':sum(s['stage']=='unclassified' for s in calls),
                    'input_tokens':describe([s.get('input_tokens') for s in tokens]),
                    'output_tokens':describe([s.get('output_tokens') for s in tokens]),
                    'reasoning_tokens':describe([s.get('reasoning_tokens') for s in tokens]),
                    'reasoning_characters':describe([s.get('reasoning_characters') for s in reasoning])})
            entry = {k:trial[k] for k in ['trial','policy','users','repeat']}
            samples = [r['state'] for r in app_states if r['trial']==trial['trial'] and 'state' in r]
            gates = [s.get('adapter',{}).get('models',{}).get('orchestrator') for s in samples]
            gates = [s for s in gates if s is not None]
            entry.update(trace_count=len(traces), dropped_spans=sum(r['diagnosis']['dropped_spans'] for r in traces),
                before_first_stage_s=describe([r['diagnosis']['before_first_instrumented_stage_s'] for r in traces]),
                stage_model_work=stage_rows,
                model_usage=model_rows,
                sampled_orchestrator_gate={k:describe([s.get(k) for s in gates]) for k in
                    ['active','queued','reserved_kv_tokens','reserved_prefill_tokens']},
                tokenize_failures=sum(s['kind']=='tokenization' and not s['success'] for s in spans),
                input_estimate_errors=describe([s.get('input_estimate_error') for s in usage]),
                output_reserve_exceeded_calls=sum((s.get('output_tokens') or 0)>512 for s in usage),
                reasoning_tokens_reported_calls=sum(s.get('reasoning_tokens') is not None for s in spans),
                internal_model_errors=sum(len(r.get('errors',[])) for r in records))
            output.append(entry)
        if joined_ids != set(workflow):
            raise ValueError('unjoined or duplicate workflow observations')
    return {'scope':'stage work sums are not a critical path; RPC durations are not exclusive GPU service',
            'trials':output}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--workflow-directory', type=Path)
    p.add_argument('--allow-stopped', action='store_true',
                   help='explicit descriptive failure audit, not a completed primary analysis')
    args = p.parse_args()
    result = aggregate(args.directory)
    if not result['complete'] and not args.allow_stopped:
        raise ValueError('main analysis requires a complete campaign')
    rows = [json.loads(line) for line in (args.directory/'requests.jsonl').open()]
    result['adapter_comparisons'] = paired_comparisons(rows)
    result['pipeline_success'] = sum(r.get('pipeline_ok',False) for r in rows)
    result['analysis_status'] = 'completed_primary' if result['complete'] else 'stopped_descriptive_only'
    result['latency_endpoint'] = 'observed time to completion or failure/deadline; failures retained, not completed-response mean'
    result['failure_rows'] = [{k:r.get(k) for k in ['trial','policy','users','repeat','case_id','kind',
                              'elapsed_s','ttft_s','error_type','status','pipeline_ok']} for r in rows if not r['ok']]
    result['method_scope'] = 'combined model call cap=8 and token estimates; not their separate causal effects'
    if args.workflow_directory:
        result['workflow'] = workflow_summary(args.directory, args.workflow_directory)
    with (args.directory/'workflow-analysis.json').open('x') as sink:
        json.dump(result, sink, indent=2)
        sink.write('\n')
    for row in result['adapter_comparisons']:
        print(json.dumps({k:row[k] for k in ['users','control','candidate','latency']}))


if __name__ == '__main__':
    main()

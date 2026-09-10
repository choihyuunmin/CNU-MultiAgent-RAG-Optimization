"""Allowlisted audit of a stopped campaign, including fallback and no-answer streams."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read_rows(path):
    return [json.loads(line) for line in path.open()]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--output', type=Path)
    args = p.parse_args()
    root = args.directory
    state = json.loads((root/'validation/summary.json').read_text())
    manifest = json.loads((root/'validation/manifest.json').read_text())
    rows = read_rows(root/'validation/requests.jsonl')
    indexed = {}
    private_by_trial = {}
    for policy in manifest['policies']:
        groups = {}
        for record in read_rows(root/f'main-apps/{policy}.trace.jsonl'):
            prefix, index = record['session_id'].rsplit('-',1)
            groups.setdefault(prefix,{})[int(index)] = record
        trials = [t for t in state['trials'] if t['policy']==policy]
        if len(groups)!=len(trials):
            raise ValueError('trace group mismatch')
        for trial,group in zip(trials,groups.values(),strict=True):
            private_by_trial[trial['trial']] = group
        indexed[policy] = {t['trace_id']:t for t in read_rows(root/f'main-apps/{policy}.workflow.jsonl')}
    output = {'complete':state['complete'], 'planned_trials':len(manifest['planned_order']),
              'executed_trials':len(state['trials']),
              'planned_requests':len(manifest['planned_order'])*len(manifest['selected_cases']),
              'attempted_requests':len(rows),'sse_success':sum(r['ok'] for r in rows),
              'unique_question_count':len({r['case_id'] for r in rows}),
              'duplicate_question_condition_repeat_rows':len(rows)-len({(r['case_id'],r['policy'],r['users'],r['repeat']) for r in rows}),
              'scope':'stopped experiment; no imputation of unexecuted arms/loads/repeats',
              'failed_span_scope':'unsuccessful scope exit includes cancellation/consumer close; not always a serving error',
              'model_failed_spans':[], 'failed_question_comparisons':[]}
    for row in rows:
        private = private_by_trial[row['trial']][row['index']]
        trace = indexed[row['policy']][private['workflow_trace_id']]
        failed = [s for s in trace['spans'] if s['kind']=='model_rpc_lifetime' and not s['success']]
        if failed:
            output['model_failed_spans'].append({k:row[k] for k in ['case_id','policy','users','repeat','ok']} | {
                'calls':[{'stage':s['stage'],'model':s['model'],'elapsed_s':s['end_s']-s['start_s']} for s in failed],
                'legacy_error_types':dict(Counter(str(e.get('error',e.get('error_type','unclassified'))) for e in private.get('errors',[])))})
    for failed in [r for r in rows if not r['ok']]:
        matches = [r for r in rows if r['case_id']==failed['case_id'] and r['users']==failed['users'] and r['repeat']==failed['repeat']]
        reference = next(r for r in matches if r['policy']=='fixed16')
        ref_private = private_by_trial[reference['trial']][reference['index']]
        ref_calls = [s for s in ref_private['llm'] if not s['stream']]
        comparison = {'case_id':failed['case_id'],'kind':failed['kind'],'arms':[]}
        for row in matches:
            private = private_by_trial[row['trial']][row['index']]
            trace = indexed[row['policy']][private['workflow_trace_id']]
            calls = [s for s in private['llm'] if not s['stream']]
            comparison['arms'].append({k:row.get(k) for k in ['policy','ok','elapsed_s','ttft_s','answer_chars']} | {
                'adapter_wait_union_s':trace['diagnosis']['interval_union_s'].get('adapter_wait'),
                'stream_sizes':[{k:s.get(k) for k in ['model','elapsed_s','chunks','characters','reasoning_characters']} for s in private['llm'] if s['stream']],
                'nonstream_vs_fixed16':[{'model':a['model'],
                    'input_hash_equal':a.get('input_sha256')==b.get('input_sha256'),
                    'output_hash_equal':a.get('output_sha256')==b.get('output_sha256')}
                    for a,b in zip(calls,ref_calls)] if len(calls)==len(ref_calls) else None,
                'scope':'nonstream hashes include request_id metadata, so unequal hashes do not prove semantic input differences; stream input hashes unavailable'})
        output['failed_question_comparisons'].append(comparison)
    output['raw_sha256'] = {str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [root/'validation/requests.jsonl', root/'validation.responses.jsonl']}
    with (args.output or root/'validation/failure-audit.json').open('x') as sink:
        json.dump(output,sink,indent=2)
        sink.write('\n')
    print(json.dumps(output))


if __name__ == '__main__':
    main()

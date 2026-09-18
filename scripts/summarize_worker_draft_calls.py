"""Export prompt-free call parity and stage timing from private E2E traces."""
from collections import Counter, defaultdict
import argparse
import json
from pathlib import Path

from moleg_scaling_metrics import describe


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def analyze(root):
    rows = read(root / 'requests.jsonl')
    private = {(r['trial'], r['case_id'], r['index']): r
               for r in read(root / 'private-responses.jsonl')}
    traces, workflows = {}, {}
    for arm in {r['arm'] for r in rows}:
        traces[arm] = {r['session_id']: r for r in read(root / 'apps' / f'{arm}.trace.jsonl')}
        workflows[arm] = {r['trace_id']: r for r in read(root / 'apps' / f'{arm}.workflow.jsonl')}
    records, missing = [], 0
    for row in rows:
        raw = private[(row['trial'], row['case_id'], row['index'])]
        sid = (raw.get('result') or {}).get('session_id')
        trace = traces[row['arm']].get(sid)
        flow = workflows[row['arm']].get(trace.get('workflow_trace_id')) if trace else None
        if not trace or not flow:
            missing += 1
            continue
        calls = [s for s in flow['spans'] if s['kind'] == 'model_rpc_lifetime']
        stage_seconds = defaultdict(float)
        for s in calls:
            stage_seconds[s['stage']] += s['end_s'] - s['start_s']
        llm = defaultdict(list)
        for call in trace.get('llm', []):
            if not call.get('stream'):
                llm[call['model']].append({k: call.get(k) for k in ['input_sha256', 'output_sha256']})
        records.append({k: row[k] for k in ['case_id', 'arm', 'level', 'repeat']}
                       | {'counts': dict(Counter(s['stage'] for s in calls)),
                          'model_rpc_seconds': dict(stage_seconds), 'llm_hashes': dict(llm),
                          'retrieval_executions': len(trace.get('retrieval', [])),
                          'preparation_hashes': [p['output_sha256'] for p in trace.get('preparation', [])]})
    loads = []
    for level in sorted({r['level'] for r in rows}):
        selected = [r for r in records if r['level'] == level]
        groups = {arm: {(r['case_id'], r['repeat']): r for r in selected if r['arm'] == arm}
                  for arm in ['baseline', 'improved']}
        b, c = groups['baseline'], groups['improved']
        pairs = sorted(set(b) & set(c))
        stats = {}
        for arm, group in groups.items():
            stage_names = sorted({s for r in group.values() for s in r['counts']})
            stats[arm] = {'requests': len(group), 'stages': {
                s: {'calls': sum(r['counts'].get(s, 0) for r in group.values()),
                    'cumulative_rpc_seconds_per_request': describe([
                        r['model_rpc_seconds'].get(s, 0) for r in group.values()])}
                for s in stage_names}}
        loads.append({'level': level, 'arms': stats, 'paired_requests': len(pairs),
                      'same_stage_call_counts': sum(b[k]['counts'] == c[k]['counts'] for k in pairs),
                      'same_retrieval_execution_counts': sum(b[k]['retrieval_executions'] == c[k]['retrieval_executions'] for k in pairs),
                      'same_preparation_outputs': sum(b[k]['preparation_hashes'] == c[k]['preparation_hashes'] for k in pairs),
                      'same_nonstream_call_envelope_hashes': sum(
                          bool(b[k]['llm_hashes']) and bool(c[k]['llm_hashes']) and
                          all(v['input_sha256'] for r in (b[k], c[k]) for seq in r['llm_hashes'].values() for v in seq) and
                          {m: [v['input_sha256'] for v in seq] for m, seq in b[k]['llm_hashes'].items()} ==
                          {m: [v['input_sha256'] for v in seq] for m, seq in c[k]['llm_hashes'].items()} for k in pairs)})
    return {'scope': 'Observed calls only. Legacy nonstream hashes include _log_request_id, so differing hashes do not establish changed model inputs; streaming inputs are not hashed. Cumulative overlapping RPC durations are not additive end-to-end time.',
            'requests': len(rows), 'missing_trace': missing, 'loads': loads}, records


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    summary, records = analyze(args.directory)
    (args.directory / 'call-comparison.json').write_text(json.dumps(summary, indent=2) + '\n')
    (args.directory / 'call-records.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in records))
    print(json.dumps(summary))

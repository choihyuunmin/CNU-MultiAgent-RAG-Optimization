"""Export allowlisted pipeline-health counts from private, ordered app traces.

Groups are consecutive trial-session prefixes, in first-observed order. Mapping
group ordinals to campaigns must be documented separately. No prompt, output,
model endpoint, request ID or session ID is exported.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def summarize(path):
    groups = {}
    with path.open() as source:
        for line in source:
            row = json.loads(line)
            session = row['session_id']
            prefix, index = session.rsplit('-', 1)
            if not session.startswith('scale-') or not index.isdecimal():
                raise ValueError('unexpected non-scaling session in private trace')
            groups.setdefault(prefix, []).append((int(index), row))
    output = []
    for ordinal, records in enumerate(groups.values()):
        indices = [index for index,row in records]
        rows = [row for index,row in records]
        counts = {}
        for kind in ['rerank', 'guardrail']:
            statuses = Counter(str(int(event['status'])) for row in rows for event in row.get(kind, []))
            counts[kind+'_http_status_counts'] = dict(statuses)
        reasons = Counter()
        for row in rows:
            for event in row.get('llm', []):
                if not event.get('stream'):
                    reason = event.get('finish_reason')
                    reasons[reason if reason in ['stop', 'length', 'tool_calls', 'content_filter', 'function_call'] else 'other'] += 1
        output.append({'group_ordinal': ordinal, 'executions_traced': len(rows),
                       'execute_returned_count': sum('response_chars' in row for row in rows),
                       'unique_indices': len(set(indices)),
                       'complete_64_index_set': sorted(indices) == list(range(64)),
                       'errors_recorded': sum(len(row.get('errors', [])) for row in rows),
                       'nonstream_finish_reasons': dict(reasons), **counts})
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trace', action='append', required=True, help='arm=private-path')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    traces = dict(item.split('=', 1) for item in args.trace)
    result = {'scope': 'instrumented execute_generate traces; not SSE completion or expert answer correctness',
              'groups': {arm: summarize(Path(path)) for arm,path in traces.items()},
              'private_trace_sha256': {arm: hashlib.sha256(Path(path).read_bytes()).hexdigest() for arm,path in traces.items()}}
    with args.output.open('x') as sink:
        json.dump(result, sink, indent=2)
        sink.write('\n')
    print(json.dumps({'arms': list(traces), 'output_written': True}))


if __name__ == '__main__':
    main()

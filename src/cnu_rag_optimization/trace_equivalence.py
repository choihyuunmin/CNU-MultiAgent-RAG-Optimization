from collections import defaultdict


def join_branch_events(events):
    groups = defaultdict(lambda: defaultdict(list))
    for event in events:
        kind = event.get('event')
        if kind not in {'review_prepared', 'review_handler', 'review_boundary'}:
            continue
        request, branch = event.get('request_id'), event.get('branch_id')
        if not request or not branch:
            raise ValueError('trace event missing request_id or branch_id')
        groups[request, branch][kind].append(event)
    records = []
    for (request, branch), kinds in groups.items():
        p, h, b = (kinds[k] for k in ('review_prepared', 'review_handler', 'review_boundary'))
        issues = []
        if len(p) != 1 or len(h) != 1 or len(b) != 1:
            issues.append('missing_or_duplicate_events')
        prepared = p[0].get('prepared_hash') if len(p) == 1 else None
        handler = h[0] if len(h) == 1 else {}
        boundary = b[0] if len(b) == 1 else {}
        if not prepared or handler.get('prepared_hash') != prepared:
            issues.append('prepared_hash_missing_or_conflicting')
        if handler.get('status') != 'ok' or boundary.get('status') != 'ok':
            issues.append('execution_failed_or_status_missing')
        if boundary.get('handler_calls') != 1:
            issues.append('handler_count_not_one')
        if handler.get('executed_hash') != prepared or not prepared:
            issues.append('executed_arguments_differ_or_missing')
        record = {'request_id': request, 'branch_id': branch, 'prepared': prepared,
                  'output': handler.get('output_hash'), 'result': boundary.get('result_hash'),
                  'law_ids': boundary.get('law_ids_hash'), 'issues': issues}
        if not all(record[k] for k in ('output', 'result', 'law_ids')):
            issues.append('output_hash_missing')
        record['valid'] = not issues
        records.append(record)
    return records


def compare_branch_sets(left, right):
    """Compare one question across runs; fail closed on ambiguous fan-out."""
    if not left or not right or any(not r['valid'] for r in left + right):
        return {'status': 'unverifiable', 'first_difference': None}
    a = {r['prepared']: r for r in left}
    b = {r['prepared']: r for r in right}
    if len(a) != len(left) or len(b) != len(right):
        return {'status': 'ambiguous_repeated_inputs', 'first_difference': None}
    if a.keys() != b.keys():
        return {'status': 'different', 'first_difference': 'prepared'}
    equal = {key: all(a[p][key] == b[p][key] for p in a)
             for key in ('output', 'result', 'law_ids')}
    first = next((key for key, matches in equal.items() if not matches), None)
    return {'status': 'equal' if first is None else 'different',
            'first_difference': first, 'prepared_equal': True, **equal}

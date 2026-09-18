"""Regression: fan-out completion order must not pair unrelated branch hashes."""
import importlib.util
import json
import copy
from pathlib import Path


FOLLOWUP = Path(__file__).parents[1] / 'docs/paper-draft-20260917/followup'


def load_analyzer():
    spec = importlib.util.spec_from_file_location('dispatch_followup', FOLLOWUP / 'analyze_followup.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_reversed_completion_order_preserves_branch_lineage(tmp_path):
    root, logs = tmp_path / 'trial', tmp_path / 'logs'
    root.mkdir()
    logs.mkdir()
    methods = ['original-a', 'capability-a']
    (root / 'status.json').write_text(json.dumps({'completed': [
        {'round': 1, 'method': arm} for arm in methods]}))
    for arm in methods:
        folder = root / 'round-1/responses/review-trial' / arm
        folder.mkdir(parents=True)
        row = {'question_id': 'q1', 'status': 'ok', 'response': {'laws': [], 'comment': 'ok'},
               'law_ids': ['law1'], 'duration_ms': 1000,
               'started_at': '2026-09-18T00:00:00+00:00',
               'completed_at': '2026-09-18T00:00:02+00:00'}
        (folder / 'client_requests.jsonl').write_text(json.dumps(row) + '\n')
        events = [{'event': 'llm_finished', 'application_request_id': 'req', 'case_id': 'q1-r1-c4'}]
        for branch in ['a', 'b']:
            events.append({'event': 'review_prepared', 'request_id': 'req', 'branch_id': branch,
                           'prepared_hash': 'input-' + branch})
        order = ['a', 'b'] if arm == 'original-a' else ['b', 'a']
        for branch in order:
            events += [
                {'event': 'review_handler', 'request_id': 'req', 'branch_id': branch,
                 'prepared_hash': 'input-' + branch, 'executed_hash': 'input-' + branch,
                 'status': 'ok', 'output_hash': 'out-' + branch},
                {'event': 'review_boundary', 'request_id': 'req', 'branch_id': branch,
                 'status': 'ok', 'handler_calls': 1, 'result_hash': 'next-' + branch,
                 'law_ids_hash': 'ids-' + branch}]
        (logs / f'{arm}-application.log').write_text(''.join(
            '2026-09-18T00:00:01.000000000+00:00 ' + json.dumps(e) + '\n' for e in events))
    output = tmp_path / 'summary.json'
    load_analyzer().main(root, logs, output)
    result = json.loads(output.read_text())['hashes'][0]
    assert result['equal_prepared'] == 1
    assert result['equal_handler_output_given_equal_prepared'] == 1
    assert result['equal_next_stage_object'] == 1


def branch_events(branch='a', prepared='p'):
    return [
        {'event': 'review_prepared', 'request_id': 'req', 'branch_id': branch, 'prepared_hash': prepared},
        {'event': 'review_handler', 'request_id': 'req', 'branch_id': branch, 'status': 'ok',
         'prepared_hash': prepared, 'executed_hash': prepared, 'output_hash': 'output'},
        {'event': 'review_boundary', 'request_id': 'req', 'branch_id': branch, 'status': 'ok',
         'handler_calls': 1, 'result_hash': 'result', 'law_ids_hash': 'ids'}]


def test_incomplete_duplicate_and_failed_trace_never_matches():
    from cnu_rag_optimization.trace_equivalence import join_branch_events, compare_branch_sets
    events = branch_events()
    good = join_branch_events(events)
    bad = copy.deepcopy(events)
    bad[1]['status'] = 'error'
    for incomplete in [events[:-1], events + [events[0]], bad]:
        assert compare_branch_sets(good, join_branch_events(incomplete))['status'] == 'unverifiable'
    assert compare_branch_sets([], [])['status'] == 'unverifiable'


def test_duplicate_input_branches_not_aligned_using_output_order():
    from cnu_rag_optimization.trace_equivalence import join_branch_events, compare_branch_sets
    records = join_branch_events(branch_events('a') + branch_events('b'))
    assert compare_branch_sets(records, records)['status'] == 'ambiguous_repeated_inputs'


def test_first_changed_stage_reported_without_equating_inputs_to_outputs():
    from cnu_rag_optimization.trace_equivalence import join_branch_events, compare_branch_sets
    left = join_branch_events(branch_events())
    for key in ['output', 'result', 'law_ids']:
        right = copy.deepcopy(left)
        right[0][key] = 'changed'
        report = compare_branch_sets(left, right)
        assert report['first_difference'] == key
        assert report['prepared_equal'] is True

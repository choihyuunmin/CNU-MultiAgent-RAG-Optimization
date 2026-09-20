import copy


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

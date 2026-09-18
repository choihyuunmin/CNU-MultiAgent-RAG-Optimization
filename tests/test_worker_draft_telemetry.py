import copy
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_worker_draft_telemetry import analyze


def fixture():
    rows = []
    for arm in ['baseline', 'improved']:
        for count, total in [(1, 10), (9, 18)]:
            rows.append({'level': 4, 'arm': arm, 'telemetry': {'worker': {
                'available': True, 'counter_reset_detected': False,
                'histograms': {name: {'count': count, 'sum': total} for name in [
                    'request_queue_time_seconds', 'request_prefill_time_seconds',
                    'request_decode_time_seconds', 'request_generation_tokens', 'request_prompt_tokens']},
                'gauges': {'num_requests_waiting': {'max': 3}, 'kv_cache_usage_perc': {'max': .4}},
                'counters': {'num_preemptions_total': 0}}}})
    return {'complete': True, 'trials': rows}


def test_pools_counts_instead_of_averaging_histogram_means():
    result = analyze(fixture())['loads'][0]['arms']['baseline']
    assert result['histograms']['request_decode_time_seconds']['mean_per_model_call'] == 2.8
    assert result['peak_observed_waiting_calls'] == 3


@pytest.mark.parametrize('corruption', ['unfinished', 'reset', 'missing'])
def test_incomplete_or_reset_measurements_cannot_be_pooled(corruption):
    data = copy.deepcopy(fixture())
    if corruption == 'unfinished':
        data['complete'] = False
    elif corruption == 'reset':
        data['trials'][0]['telemetry']['worker']['counter_reset_detected'] = True
    else:
        data['trials'][0]['telemetry']['worker']['histograms'].pop('request_decode_time_seconds')
    with pytest.raises(ValueError):
        analyze(data)

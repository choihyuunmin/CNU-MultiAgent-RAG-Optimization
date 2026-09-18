"""Pool server histogram deltas without treating RPC timing as engine calibration."""
import argparse
import json
from pathlib import Path


def analyze(summary):
    if not summary['complete']:
        raise ValueError('study is not complete')
    result = []
    for level in sorted({t['level'] for t in summary['trials']}):
        arms = {}
        for arm in ['baseline', 'improved']:
            trials = [t for t in summary['trials'] if t['level'] == level and t['arm'] == arm]
            samples = [t['telemetry']['worker'] for t in trials]
            if not samples or any(not s.get('available') or s.get('counter_reset_detected') for s in samples):
                raise ValueError('worker metrics missing or reset')
            histograms = {}
            for name in ['request_queue_time_seconds', 'request_prefill_time_seconds',
                         'request_decode_time_seconds', 'request_generation_tokens', 'request_prompt_tokens']:
                values = [s['histograms'].get(name) for s in samples]
                if any(v is None or v.get('sum') is None or v.get('count') is None for v in values):
                    raise ValueError('required worker histogram unavailable')
                count = sum(v['count'] for v in values)
                total = sum(v['sum'] for v in values)
                histograms[name] = {'count': count, 'sum': total, 'mean_per_model_call': total / count if count else None}
            arms[arm] = {'histograms': histograms,
                         'peak_observed_waiting_calls': max(s['gauges']['num_requests_waiting']['max'] for s in samples),
                         'peak_observed_kv_fraction': max(s['gauges']['kv_cache_usage_perc']['max'] for s in samples),
                         'preemptions': sum(s['counters']['num_preemptions_total'] for s in samples)}
        result.append({'level': level, 'arms': arms})
    return {'scope': 'Deltas for the owned loopback worker only, outside warmup. Means are per MODEL CALL, not per search request. Server request decode intervals are not synchronized engine windows or pure GPU kernel costs and cannot calibrate the draft-budget policy.',
            'loads': result}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(json.loads((args.directory / 'summary.json').read_text()))
    (args.directory / 'worker-telemetry.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

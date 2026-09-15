"""Freeze a selection-only winner before a separate validation campaign."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


def select(data):
    if not data['complete'] or len(data['groups']) != 5:
        raise ValueError('need five completed factorial groups')
    baseline = next(g for g in data['groups'] if g['policy'] == 'baseline')
    eligible = [g for g in data['groups'] if g['policy'] != 'baseline'
                and g['n'] == g['success'] == 64
                and g['source_law_hit'] >= baseline['source_law_hit']
                and g['source_chunk_hit'] >= baseline['source_chunk_hit']]
    if not eligible:
        raise ValueError('no eligible candidate; do not run a speed-only validation')
    return min(eligible, key=lambda g: g['metrics']['elapsed_s']['mean'])['policy']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    path = args.directory / 'aggregate.json'
    data = json.loads(path.read_text())
    winner = select(data)
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'selected': winner,
              'rule': 'minimum mean latency among non-baseline 64/64 candidates with both source inclusion rates at least baseline',
              'source_aggregate_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
              'validation': {'users': [16, 32], 'repeats': 2, 'cases_per_trial': 64,
                             'timeout_s': 240, 'slo_s': 30, 'planned_requests': 512,
                             'variants': ['baseline', winner]},
              'limitations': ['selection and validation use the same question set but new executions',
                              'source labels do not establish expert answer quality',
                              'selection trials are excluded from validation estimates']}
    with (args.directory / 'selection.json').open('x') as sink:
        json.dump(result, sink, indent=2)
        sink.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

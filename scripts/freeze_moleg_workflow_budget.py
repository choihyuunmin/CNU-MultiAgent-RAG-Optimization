"""Apply the predeclared development-only rule, then freeze main-test settings."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path

from moleg_scaling_metrics import describe


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory', type=Path, required=True)
    p.add_argument('--verification-trace', type=Path)
    args = p.parse_args()
    root = args.directory
    if (root / 'calibration.json').exists():
        p.error('calibration is already frozen')
    state = json.loads((root / 'development/summary.json').read_text())
    if not state['complete'] or any(t['n'] != 32 or t['pipeline_success'] != 32 for t in state['trials']):
        raise ValueError('development is not complete and healthy')
    paths = [root / f'development-apps/{name}.workflow.jsonl' for name in ['baseline','fixed16']]
    traces = [json.loads(line) for path in paths for line in path.open()]
    if len(traces) != 64 or any(t['diagnosis']['dropped_spans'] for t in traces):
        raise ValueError('incomplete development traces')
    usage = [s for t in traces for s in t['spans'] if s['kind'] == 'usage' and s['model'] == 'orchestrator']
    tokenization = [s for t in traces for s in t['spans'] if s['kind'] == 'tokenization']
    errors = [s for s in usage if s.get('input_estimate_error') not in [0, None]]
    development_meter_failures = sum(not s['success'] for s in tokenization)
    verification = None
    if development_meter_failures and args.verification_trace:
        checked = [json.loads(line) for line in args.verification_trace.open()]
        checked_tokens = [s for t in checked for s in t['spans'] if s['kind'] == 'tokenization']
        checked_usage = [s for t in checked for s in t['spans'] if s['kind'] == 'usage' and s['model']=='orchestrator']
        verification = {'traces':len(checked), 'tokenizations':len(checked_tokens),
            'failures':sum(not s['success'] for s in checked_tokens),
            'count_mismatches':sum(s.get('input_estimate_error') not in [0,None] for s in checked_usage),
            'sha256':hashlib.sha256(args.verification_trace.read_bytes()).hexdigest()}
    if (not usage or errors or (development_meter_failures and not (verification
            and verification['traces']==32 and verification['tokenizations']>0
            and verification['failures']==0 and verification['count_mismatches']==0))):
        raise ValueError('tokenization must be verified before fixed calibration')
    inputs = describe([s.get('input_tokens') for s in usage])
    outputs = describe([s.get('output_tokens') for s in usage])
    output_reserve = max(512, math.ceil(outputs['max'] * 1.5))
    budget = {'max_calls': 8, 'max_prefill_tokens': math.ceil(8 * inputs['p95']),
              'max_kv_tokens': math.ceil(8 * inputs['p95'] + 8 * output_reserve)}
    base = json.loads((root / 'observe.json').read_text())
    base['serving_meter']['output_reserve_tokens'] = output_reserve
    (root / 'observe-final.json').write_text(json.dumps(base, indent=2) + '\n')
    candidate = {**base, 'mode': 'budget', 'budgets': {'orchestrator': budget}}
    (root / 'budget-final.json').write_text(json.dumps(candidate, indent=2) + '\n')
    arms = []
    for name, slots, emission, port, config in [
        ('baseline',4,'original',28330,'observe-final.json'),
        ('fixed16',16,'immediate',28331,'observe-final.json'),
        ('budget',16,'immediate',28332,'budget-final.json')]:
        arms.append(dict(name=name, slots=slots, emission=emission, port=port,
                         adapter_config=str(root / config)))
    (root / 'main-arms.json').write_text(json.dumps(arms, indent=2) + '\n')
    result = {'utc': datetime.now(timezone.utc).isoformat(), 'development_traces': len(traces),
              'orchestrator_calls': len(usage), 'input_tokens': inputs, 'output_tokens': outputs,
              'tokenizer_verified_pairs': sum(s.get('input_estimate_error') == 0 for s in usage),
              'output_reserve_tokens': output_reserve, 'budget': budget,
              'development_meter_failures_retained': development_meter_failures,
              'separate_meter_recheck': verification,
              'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
              'heldout_results_accessed_for_calibration': False}
    with (root / 'calibration.json').open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps(result))


if __name__ == '__main__':
    main()

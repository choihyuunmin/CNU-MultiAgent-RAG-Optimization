"""Summarize paired fixed-K HTTP timings; never treat these as decode costs."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from moleg_paper_metrics import paired_cluster_ci
from moleg_scaling_metrics import describe


def analyze(root):
    plan = json.loads((root / 'plan.json').read_text())
    rows = [json.loads(x) for x in (root / 'rows.jsonl').read_text().splitlines()]
    blocks = [json.loads(x) for x in (root / 'blocks.jsonl').read_text().splitlines()]
    keys = [(r['case_id'], r['k'], r['repeat'], r['concurrency']) for r in rows]
    cases = sorted({r['case_id'] for r in rows})
    expected = {(c, k, r, level) for c in cases for k in plan['budgets']
                for r in range(plan['repeats']) for level in plan['levels']}
    if (len(cases) != plan['cases'] or len(keys) != len(set(keys))
            or set(keys) != expected or len(rows) != plan['expected_requests']):
        raise ValueError('incomplete or duplicate request coverage')
    block_keys = [(b['k'], b['repeat'], b['concurrency']) for b in blocks]
    expected_blocks = {(k, r, level) for k in plan['budgets']
                       for r in range(plan['repeats']) for level in plan['levels']}
    if len(block_keys) != len(set(block_keys)) or set(block_keys) != expected_blocks:
        raise ValueError('incomplete or duplicate timing blocks')
    inputs = defaultdict(set)
    for row in rows:
        inputs[row['case_id']].add(row['input_sha256'])
    if any(len(values) != 1 for values in inputs.values()):
        raise ValueError('input changed within paired comparison')
    levels = []
    lookup = dict(zip(keys, rows))
    for level in plan['levels']:
        arms = {}
        for k in plan['budgets']:
            subset = [r for r in rows if r['k'] == k and r['concurrency'] == level]
            selected = [b for b in blocks if b['k'] == k and b['concurrency'] == level]
            wall = sum(b['wall_s'] for b in selected)
            counts = {}
            for name in ['spec_decode_num_drafts_total', 'spec_decode_num_draft_tokens_total',
                         'spec_decode_num_accepted_tokens_total', 'num_preemptions_total']:
                key = 'vllm:' + name
                deltas = [b['counters_after'][key] - b['counters_before'][key]
                          for b in selected if key in b['counters_before'] and key in b['counters_after']]
                counts[name] = sum(deltas) if len(deltas) == len(selected) else None
            arms[str(k)] = {'n': len(subset), 'ok': sum(r['ok'] for r in subset),
                           'arguments_exact': sum(r['arguments_exact'] for r in subset),
                           'latency_s': describe([r['elapsed_s'] for r in subset]),
                           'wall_s': wall, 'successful_rps': sum(r['ok'] for r in subset) / wall,
                           'counters': counts}
        comparisons = {}
        for k in plan['budgets']:
            if k == 0:
                continue
            pairs, identical = [], 0
            for case in cases:
                baseline = [lookup[case, 0, rep, level] for rep in range(plan['repeats'])]
                candidate = [lookup[case, k, rep, level] for rep in range(plan['repeats'])]
                pairs.append((statistics.fmean(r['elapsed_s'] for r in baseline),
                              statistics.fmean(r['elapsed_s'] for r in candidate)))
                identical += sum(b.get('output_sha256') is not None and
                                 b['output_sha256'] == c.get('output_sha256')
                                 for b, c in zip(baseline, candidate))
            comparisons[str(k)] = {**paired_cluster_ci(pairs), 'identical_tool_output_pairs': identical,
                                   'paired_requests': len(cases) * plan['repeats']}
        levels.append({'concurrency': level, 'arms': arms, 'comparisons': comparisons})
    return {'complete': True, 'n': len(rows), 'unique_questions': len(cases),
            'scope': 'Matched isolated worker HTTP stage; not end-to-end or engine decode cost.',
            'dynamic_k': False, 'levels': levels}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.directory)
    (args.directory / 'summary.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result))

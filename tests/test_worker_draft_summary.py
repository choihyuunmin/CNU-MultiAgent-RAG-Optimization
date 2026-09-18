import json
from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from summarize_worker_draft_pilot import analyze


def fixture(root):
    plan = {'cases': 2, 'repeats': 2, 'budgets': [0, 4], 'levels': [1], 'expected_requests': 8}
    rows = [dict(case_id=c, k=k, repeat=r, concurrency=1, input_sha256=c,
                 output_sha256='same', ok=True, arguments_exact=True,
                 elapsed_s=2 if k == 0 else 1)
            for c in ['a', 'b'] for k in [0, 4] for r in range(2)]
    blocks = [dict(k=k, repeat=r, concurrency=1, wall_s=4 if k == 0 else 2,
                   counters_before={}, counters_after={}) for k in [0, 4] for r in range(2)]
    (root / 'plan.json').write_text(json.dumps(plan))
    (root / 'blocks.jsonl').write_text(''.join(json.dumps(b) + '\n' for b in blocks))
    return rows


def write_rows(root, rows):
    (root / 'rows.jsonl').write_text(''.join(json.dumps(r) + '\n' for r in rows))


@pytest.mark.parametrize('corruption', ['missing', 'duplicate', 'input_changed'])
def test_rejects_unpaired_or_changed_inputs(tmp_path, corruption):
    rows = fixture(tmp_path)
    if corruption == 'missing':
        rows.pop()
    elif corruption == 'duplicate':
        rows[-1] = rows[0]
    else:
        rows[-1]['input_sha256'] = 'different'
    write_rows(tmp_path, rows)
    with pytest.raises(ValueError):
        analyze(tmp_path)


def test_failures_remain_in_latency_and_never_equal_missing_output(tmp_path):
    rows = fixture(tmp_path)
    for row in rows:
        if row['case_id'] == 'b':
            row.update(ok=False, arguments_exact=False, elapsed_s=120)
            row.pop('output_sha256')
    write_rows(tmp_path, rows)
    result = analyze(tmp_path)['levels'][0]
    assert result['arms']['4']['latency_s']['mean'] == 60.5
    assert result['arms']['4']['ok'] == 2
    assert result['arms']['4']['successful_rps'] == .5
    assert result['comparisons']['4']['identical_tool_output_pairs'] == 2
    assert result['comparisons']['4']['n'] == 2

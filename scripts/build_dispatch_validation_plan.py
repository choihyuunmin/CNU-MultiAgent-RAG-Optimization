"""Create an offline draft plan. Never contact an endpoint or start a GPU job."""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from cnu_rag_optimization.performance_gate import PerformanceThresholds


def build_plan(questions, *, seed=20260918, question_sha256=None):
    if len(questions) != 200 or len({q['id'] for q in questions}) != 200:
        raise ValueError('exactly 200 unique question IDs required')
    if any(not isinstance(q['id'], str) or not q['id'].strip() for q in questions):
        raise ValueError('question IDs must be nonempty strings')
    thresholds = PerformanceThresholds()
    # Four-position Williams design: every arm occupies each position once;
    # directed adjacent arm pairs are balanced. Counterbalance physical slots
    # on a fresh deployment in each block; never switch a live production pod.
    arms = ['original-a', 'direct-a', 'original-b', 'direct-b']
    order = [[0, 1, 3, 2], [1, 2, 0, 3], [2, 3, 1, 0], [3, 0, 2, 1]]
    cells = []
    for block in range(4):
        ids = [q['id'] for q in questions]
        random.Random(seed + block).shuffle(ids)
        loads = list(thresholds.loads)
        random.Random(seed + 100 + block).shuffle(loads)
        for load in loads:
            for position, index in enumerate(order[block], 1):
                arm = arms[index]
                cells.append({'block': block + 1, 'concurrency': load, 'position': position,
                              'arm': arm, 'mode': arm.split('-')[0],
                              'request_order_seed': seed + block,
                              'physical_slot': (index + block) % 4,
                              'question_ids': ids, 'timeout_seconds': 600})
    return {'status': 'draft_requires_approval_and_expert_labels', 'seed': seed,
            'question_sha256': question_sha256, 'unique_questions': 200,
            'planned_requests': len(cells) * 200, 'cells': cells,
            'thresholds': asdict(thresholds),
            'quality_margins': {'source': 0.0, 'correctness': 0.0, 'groundedness': 0.0},
            'execution_equivalence_required': True,
            'notes': ['No inference starts from this plan.',
                      'Freeze dataset, code, image and configuration hashes before execution.',
                      'Primary load is 100; other loads are regression checks.',
                      'Repeated questions are not new independent questions.',
                      'All quality margins and 5% latency target require approval before runs.']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--questions', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    raw = args.questions.read_bytes()
    questions = [json.loads(line) for line in raw.splitlines() if line.strip()]
    plan = build_plan(questions, question_sha256=hashlib.sha256(raw).hexdigest())
    with args.output.open('x') as output:
        json.dump(plan, output, indent=2, ensure_ascii=False)
        output.write('\n')
    print(f"Draft only: {plan['planned_requests']} requests; no GPU work started")


if __name__ == '__main__':
    main()

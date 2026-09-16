# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.1"]
# ///
"""Separate actual worker-stage and complete-retriever measurements."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, required=True)
    args = parser.parse_args()
    root = args.directory
    stage = json.loads((root / 'pilot-k4/summary.json').read_text())
    e2e = json.loads((root / 'e2e-k4/stage-outcomes.json').read_text())
    if not stage['complete'] or not e2e['complete'] or e2e['audit_errors']:
        raise ValueError('only completed audited measurements')
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), layout='constrained')
    for ax, loads, title, kind in [
        (axes[0], stage['levels'], 'Search-tool model call: 12 questions', 'stage'),
        (axes[1], e2e['loads'], 'Complete retriever: 8 held-out questions', 'e2e')]:
        levels = []
        for i, load in enumerate(loads):
            levels.append(load.get('concurrency', load.get('level')))
            for offset, arm, label, color in [(-.19, '0' if kind == 'stage' else 'baseline', 'Plain', '#64748b'),
                                               (.19, '4' if kind == 'stage' else 'improved', '4-token draft', '#087f8c')]:
                value = load['arms'][arm]
                timing = value['latency_s' if kind == 'stage' else 'latency_including_failures']
                x = i + offset
                ax.bar(x, timing['mean'], width=.34, color=color, label=label if i == 0 else None)
                ax.plot(x, timing['p95'], 'D', color='#222222', markersize=5)
                ax.text(x, timing['mean'] / 2, f"{timing['mean']:.2f}", ha='center', va='center',
                        color='white', fontsize=9)
                ax.text(x, timing['p95'] + .025 * ax.get_ylim()[1], f"{timing['p95']:.2f}", ha='center', fontsize=8)
        ax.set_xticks(range(len(levels)), [f'C={level}' for level in levels])
        ax.set_title(title)
        ax.set_ylabel('Observed time (seconds)')
        ax.set_ylim(0, ax.get_ylim()[1] * 1.25)
        ax.grid(axis='y', alpha=.2)
        ax.set_axisbelow(True)
        ax.legend(loc='upper left')
    fig.suptitle('Matched isolated-worker comparisons: mean (bars), p95 (diamonds)\n'
                 'Two counterbalanced repetitions. Separate question sets; no production-performance claim.', fontsize=11)
    fig.savefig(root / 'worker-draft.png', dpi=180)
    fig.savefig(root / 'worker-draft.pdf')


if __name__ == '__main__':
    main()

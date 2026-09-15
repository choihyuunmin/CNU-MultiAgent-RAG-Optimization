# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Selection-only ablation figure; do not pool with new validation executions."""
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
    data = json.loads((args.directory/'aggregate.json').read_text())
    resources = json.loads((args.directory/'resources-audited.json').read_text())
    if not data['complete']:
        raise ValueError('selection campaign is incomplete')
    names = ['baseline', 'emit', 'slots8', 'emit8', 'emit16']
    groups = {g['policy']: g for g in data['groups']}
    states = {t['policy']: t for t in resources['trials']}
    rows = [groups[name] for name in names]
    labels = ['A: 4/orig', 'B: 4/fast', 'C: 8/orig', 'D: 8/fast', 'E: 16/fast']
    x = list(range(len(names)))
    plt.rcParams.update({'font.size': 9, 'pdf.fonttype': 42,
                         'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), layout='constrained')
    for metric, marker, label in [('mean', 'o', 'mean'), ('p95', 's', 'p95')]:
        axes[0,0].plot(x, [r['metrics']['elapsed_s'][metric] for r in rows],
                       marker+'-', label=label)
    axes[0,0].set(title='Full response latency', ylabel='seconds')
    before = [r['metrics']['first_event_s']['mean'] for r in rows]
    axes[0,1].bar(x, before, label='before first progress', color='#B75555')
    axes[0,1].bar(x, [r['metrics']['after_first_event_s']['mean'] for r in rows],
                  bottom=before, label='after first progress', color='#2687A8')
    axes[0,1].set(title='Observed milestones (not causal stages)', ylabel='mean seconds / API')
    for key, label in [('throughput_rps', 'all completed'), ('slo_goodput_rps', 'completed within 30 s')]:
        axes[0,2].plot(x, [r[key] for r in rows], 'o-', label=label)
    axes[0,2].set(title='Throughput and goodput', ylabel='requests / second')
    for phase in ['prefill', 'decode']:
        axes[1,0].plot(x, [r['telemetry']['orchestrator']['histograms'][
            'request_'+phase+'_time_seconds']['mean'] for r in rows], 'o-', label=phase)
    axes[1,0].set(title='Shared orchestrator timing counters', ylabel='mean seconds / model call')
    axes[1,1].plot(x, [100*r['telemetry']['orchestrator']['peak_kv_usage'] for r in rows],
                   'o-', color='#B75555', label='peak KV allocation')
    axes[1,1].set(title='KV capacity pressure (2-second samples)', ylabel='percent of logical KV capacity')
    for gpu, color in [('0', '#2687A8'), ('1', '#E28E30')]:
        metric = 'DRAM Read Bandwidth [Throughput %]'
        for stat, style in [('mean', 'o-'), ('p95', 's--')]:
            axes[1,2].plot(x, [states[n]['nsys_gpus'][gpu][metric][stat] for n in names],
                           style, color=color, label='GPU '+gpu+' '+stat)
    axes[1,2].set(title='Nsight whole-device DRAM read activity', ylabel='reported percent (not GB/s)')
    for ax in axes.flat:
        ax.set_xticks(x, labels, rotation=15)
        ax.set_ylim(bottom=0)
        ax.grid(axis='y', alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle('Selection only: 16 users, 64 fixed questions per arm, one pass; both app limits changed together\n'
                 'Same serving processes; GPU metrics at 100 Hz; baseline first ~1.6 s outside Nsight coverage', fontsize=11)
    for suffix in ['pdf', 'png']:
        fig.savefig(args.directory/('ablation.'+suffix), dpi=180, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()

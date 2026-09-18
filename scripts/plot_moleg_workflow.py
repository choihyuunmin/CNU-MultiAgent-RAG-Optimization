"""Plot the frozen workflow holdout only; do not pool with earlier experiments."""
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
    data = json.loads((args.directory/'workflow-analysis.json').read_text())
    if not data['complete']:
        raise ValueError('do not present stopped descriptive trials as completed primary curves')
    groups = {(r['users'],r['policy']):r for r in data['groups']}
    loads = [16,32]
    arms = [('baseline','A: original 4/4','#78828A'),
            ('fixed16','B: fixed 16/16','#2687A8'),
            ('budget','C: B + call/token budget','#B75555')]
    plt.rcParams.update({'font.size':10, 'pdf.fonttype':42,
                         'axes.spines.top':False, 'axes.spines.right':False})
    fig, axes = plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for name,label,color in arms:
        rows = [groups[u,name] for u in loads]
        for ax,metric,title in [(axes[0,0],'mean','Mean full-response latency'),
                                (axes[0,1],'p95','P95 full-response latency')]:
            ax.plot(loads,[r['metrics']['elapsed_s'][metric] for r in rows],
                    'o-',color=color,label=label)
            ax.set(title=title,ylabel='seconds')
        axes[1,0].plot(loads,[r['throughput_rps'] for r in rows],
                       'o-',color=color,label=label)
        axes[1,1].plot(loads,[100*r['slo_attainment'] for r in rows],
                       'o-',color=color,label=label)
    axes[1,0].set(title='Finite-workload throughput',ylabel='completed requests / second')
    axes[1,1].set(title='Completed within 30 seconds',ylabel='percent of all requests')
    for ax in axes.flat:
        ax.set_xticks(loads)
        ax.set(xlabel='concurrent users',ylim=(0,None))
        ax.grid(axis='y',alpha=.2)
        ax.legend(frameon=False,fontsize=8)
    fig.suptitle('Workflow adapter holdout: 64 questions × 2 repeats per arm/load\n'
                 'Same model requests; combined call/token cap; shared warm serving caches',fontsize=11)
    for suffix in ['png','pdf']:
        fig.savefig(args.directory/('workflow-holdout.'+suffix),dpi=180,bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    main()

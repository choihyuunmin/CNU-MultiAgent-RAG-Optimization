# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.1"]
# ///
"""Publication figures from validated public study metrics."""
import argparse
import json

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',required=True)
    args=p.parse_args()
    from pathlib import Path
    root=Path(args.directory)
    a=json.loads((root/'analysis-vs-original.json').read_text())
    b=json.loads((root/'analysis-vs-prior.json').read_text())
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    names={'baseline':'Original','prior':'Prior: low reasoning','improved':'Adaptive ID harness + low reasoning'}
    colors={'baseline':'#64748b','prior':'#d97706','improved':'#087f8c'}
    for arm in names:
        data=b if arm=='prior' else a
        points=[r for r in data['loads']]
        xs=[r['concurrency'] for r in points]
        stats=[r['arms'][arm] for r in points]
        values=[([s['latency_including_failures']['mean'] for s in stats],'Mean response time (s)'),
                ([s['latency_including_failures']['p95'] for s in stats],'P95 response time (s)'),
                ([s['throughput_rps'] for s in stats],'Successful pipeline requests / s'),
                ([s['goodput_30s_rps'] for s in stats],'Successful within 30 s / s')]
        for ax,(ys,title) in zip(axes.flat,values):
            ax.plot(xs,ys,'o-',label=names[arm],color=colors[arm])
            ax.set_title(title)
    for ax in axes.flat:
        ax.set_xscale('log');ax.set_xticks(xs,[str(x) for x in xs]);ax.set_ylim(bottom=0)
        ax.set_xlabel('Concurrent requests');ax.grid(alpha=.2)
    axes[0,0].legend(fontsize=8,frameon=False)
    fig.suptitle('100 held-out regression questions; two repeats; shared model servers\nFailures retained; pipeline success is not expert correctness',fontsize=11)
    fig.savefig(root/'performance.png',dpi=180)
    fig.savefig(root/'performance.pdf')
    plt.close(fig)


if __name__=='__main__':main()

# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.1"]
# ///
"""Publish the audited 1-to-200 request concurrency measurements."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    root=parser.parse_args().directory
    data=json.loads((root/'analysis.json').read_text())
    loads=data['loads'];xs=[r['concurrency'] for r in loads]
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    names={'baseline':'Original','improved':'Short IDs + restoration + low reasoning'}
    colors={'baseline':'#64748b','improved':'#087f8c'}
    definitions=[('Mean observed time, failures included (s)',lambda s:s['latency_including_failures']['mean']),
                 ('P95 observed time, 240 s deadline (s)',lambda s:s['latency_including_failures']['p95']),
                 ('Pipeline completion (%)',lambda s:100*s['pipeline_success']/s['n']),
                 ('Successful pipeline requests / s',lambda s:s['throughput_rps']),
                 ('Successful within 30 s / s',lambda s:s['goodput_30s_rps'])]
    for arm in names:
        points=[r['arms'][arm] for r in loads]
        for ax,(title,value) in zip(axes.flat,definitions):
            ax.plot(xs,[value(s) for s in points],'o-',label=names[arm],color=colors[arm],markersize=4)
            ax.set_title(title)
    ax=axes[1,2]
    for key,label,color in [('candidate_recall_vs_baseline','Improved vs original','#087f8c'),
                            ('baseline_repeat_recall','Original vs its repeat','#64748b')]:
        ax.plot(xs,[100*r[key] if r[key] is not None else float('nan') for r in loads],
                'o-',label=label,color=color,markersize=4)
    ax.set_title('Evidence ID recall (%) — not accuracy')
    ax.legend(frameon=False,fontsize=8)
    axes[0,0].legend(frameon=False,fontsize=8)
    for ax in axes.flat:
        ax.set_xscale('log');ax.set_xticks(xs,[str(x) for x in xs],rotation=45)
        ax.set_xlabel('Concurrent requests');ax.set_ylim(bottom=0);ax.grid(alpha=.2)
    axes[0,2].set_ylim(0,105);axes[1,2].set_ylim(0,105)
    fig.suptitle('Reference ID shortening and restoration: actual RAG, 2 repeats per condition\n'
                 '32–200 questions per trial; at C ≥ 32 a single burst; failures retained; shared model servers',fontsize=11)
    fig.savefig(root/'scaling.png',dpi=180)
    fig.savefig(root/'scaling.pdf')


if __name__=='__main__':main()

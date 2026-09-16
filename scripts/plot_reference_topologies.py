# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.1"]
# ///
"""Standalone figure for the synthetic portability study."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--directory',type=Path,required=True)
    root=parser.parse_args().directory
    data=json.loads((root/'analysis.json').read_text())
    plt.rcParams.update({'font.size':10,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(1,2,figsize=(10,4.4),layout='constrained')
    names={'original':'Original references','calibrated':'Short IDs + original ID restoration'}
    colors={'original':'#64748b','calibrated':'#087f8c'}
    labels=['Sequential','Fork / join','Dynamic loop']
    for i,mode in enumerate(names):
        points=[r['arms'][mode] for r in data['topologies']]
        x=[j+(i-.5)*.36 for j in range(3)]
        for ax,metric,title in zip(axes,['mean_s','output_tokens'],['Mean workflow latency (s)','Output tokens per workflow']):
            values=[r[metric] if metric=='mean_s' else r[metric]/r['n'] for r in points]
            ax.bar(x,values,width=.34,label=names[mode],color=colors[mode])
            ax.set_title(title)
            ax.set_xticks(range(3),labels)
            ax.grid(axis='y',alpha=.2)
    axes[0].set_ylim(0, max(r['arms']['original']['mean_s'] for r in data['topologies'])*1.28)
    axes[0].legend(frameon=False,fontsize=8)
    fig.suptitle('Synthetic UUID record filtering: 3 schemas, 18 held-out cases, 2 repeats\nTopology portability evidence; not real RAG speedup or universal task accuracy',fontsize=11)
    fig.savefig(root/'topologies.png',dpi=180)
    fig.savefig(root/'topologies.pdf')


if __name__=='__main__':main()

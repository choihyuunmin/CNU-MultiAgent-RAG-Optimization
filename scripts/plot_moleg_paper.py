# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib>=3.9"]
# ///
"""Standalone publication figures from measured, sanitized results only."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import ScalarFormatter


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--directory',type=Path,required=True)
    args=ap.parse_args();root=args.directory
    summary=json.loads((root/'summary.json').read_text())
    rows=[json.loads(x) for x in (root/'e2e_stream.metrics.jsonl').read_text().splitlines()]
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
        'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(1,3,figsize=(12,3.5),layout='constrained')
    colors={'baseline':'#536274','speed':'#2687A8','balanced':'#E28E30'}
    for name in ['baseline','speed','balanced']:
        values=sorted(r['elapsed_s'] for r in rows if r['variant']==name)
        if values:axes[0].plot(values,[(i+1)/len(values) for i in range(len(values))],label=name,color=colors[name])
    axes[0].set(xlabel='Full SSE completion (seconds, log scale)',ylabel='Cumulative request fraction',title='A  End-to-end latency',xscale='log')
    axes[0].xaxis.set_major_formatter(ScalarFormatter())
    axes[0].legend(frameon=False,loc='lower right');axes[0].grid(alpha=.2)
    retrieval=summary['retrieval']['variants'];names=['raw','auth_content','metadata','window']
    axes[1].bar(range(4),[retrieval[n]['metrics']['chunk_hit1'] for n in names],color=['#536274','#E28E30','#2687A8','#77A892'])
    axes[1].set_xticks(range(4),['Raw','Auth','Metadata','Window'],rotation=20)
    axes[1].set(ylabel='Source-chunk Hit@1',ylim=(0,1),title='B  Silver retrieval labels')
    for i,n in enumerate(names):
        value=retrieval[n]['metrics']['chunk_hit1'];axes[1].text(i,value+.025,f'{value:.3f}',ha='center')
    variants=summary['e2e_stream']['variants']
    names=['baseline','speed','balanced']
    axes[2].bar(range(3),[variants[n]['known_item_in_returned_evidence']['source_law_in_returned_evidence'] for n in names],
                color=[colors[n] for n in names])
    axes[2].set_xticks(range(3),names,rotation=20)
    axes[2].set(ylabel='Source law in returned evidence',ylim=(0,1),title='C  Evidence preservation')
    fig.savefig(root/'paper_figure.pdf',bbox_inches='tight')
    fig.savefig(root/'paper_figure.png',dpi=220,bbox_inches='tight')
    plt.close(fig)


if __name__=='__main__':main()

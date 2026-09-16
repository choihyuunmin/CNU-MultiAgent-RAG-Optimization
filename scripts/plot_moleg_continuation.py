# /// script
# requires-python = ">=3.11"
# dependencies = ["matplotlib==3.10.1"]
# ///
"""Plot separate paired studies; do not splice different controls into one curve."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--allow-unretrieved',action='store_true',help='label an unavailable C200 panel; do not fabricate it')
    args=p.parse_args();root=args.directory
    studies=[('pilot-auth',32,'Restore reranker authentication, C=32',
              {'baseline':'Auth error','improved':'Auth repaired'}),
             ('pilot-dispatch',64,'16 model slots, C=64',
              {'baseline':'No limit','fifo':'FIFO 16','improved':'Continuation 16'}),
             ('pilot-window32',100,'32 model slots, C=100',
              {'baseline':'No limit','improved':'Continuation 32'}),
             ('validation200',200,'32 model slots, C=200',
              {'baseline':'No limit','fifo':'FIFO 32','improved':'Continuation 32'})]
    plt.rcParams.update({'font.size':9,'pdf.fonttype':42,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
    colors={'baseline':'#64748b','fifo':'#ba8530','improved':'#087f8c'}
    for ax,(name,level,title,names) in zip(axes.flat,studies):
        path=root/name/'stage-outcomes.json'
        if not path.exists() and name=='validation200' and args.allow_unretrieved:
            ax.set_axis_off()
            ax.text(.5,.55,'C=200: results not retrieved',ha='center',transform=ax.transAxes,fontweight='bold')
            ax.text(.5,.42,'Internal-network connection unavailable.\nNo final outcome is inferred.',ha='center',transform=ax.transAxes,color='#64748b')
            continue
        data=json.loads(path.read_text())
        if not data['complete'] or data['audit_errors']:raise ValueError('only completed audited studies')
        load=next(x for x in data['loads'] if x['level']==level)
        arms=[a for a in names if a in load['arms']]
        for i,arm in enumerate(arms):
            row=load['arms'][arm];metric=row['latency_including_failures']
            ax.bar(i,metric['mean'],color=colors[arm],width=.55)
            ax.plot(i,metric['p95'],'D',color='#222222',markersize=5)
            ax.text(i,metric['mean']/2,f"{metric['mean']:.1f}s",ha='center',color='white',fontweight='bold')
            ax.text(i,metric['p95']+3,f"p95 {metric['p95']:.1f}s",ha='center',fontsize=8)
        ax.set_xticks(range(len(arms)),[names[a]+'\n'+str(load['arms'][a]['dependency_clean_completion'])+'/'+str(load['arms'][a]['n'])+' clean' for a in arms])
        ax.set_title(title);ax.set_ylabel('Observed end-to-end time (s)')
        ax.set_ylim(0,max(load['arms'][a]['latency_including_failures']['p95'] for a in arms)*1.2+5)
        ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    fig.suptitle('Incremental actual-RAG experiments: means (bars), p95 (diamonds), two repetitions\n'
                 'Failures retained, 240 s deadline. Clean = no observed stage/dependency failure; not answer accuracy.',fontsize=11)
    fig.savefig(root/'comparisons.png',dpi=180);fig.savefig(root/'comparisons.pdf')

if __name__=='__main__':main()

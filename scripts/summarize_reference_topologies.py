"""Audit and summarize synthetic portability measurements without GPU calls."""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import statistics
import argparse


def analyze(root):
    rows=[json.loads(line) for line in (root/'requests.jsonl').read_text().splitlines()]
    summary=json.loads((root/'summary.json').read_text())
    expected=set()
    domains=['catalog','incidents','bibliography']
    for domain in domains:
        for i in range(12):
            for mode in ['original','candidate']:
                for repeat in range(2):
                    expected.add(('calibration',domain,i,'single',mode,repeat))
        for i in range(6):
            for topology in ['chain','fork_join','loop']:
                for mode in ['original','calibrated']:
                    for repeat in range(2):
                        expected.add(('heldout',domain,i,topology,mode,repeat))
    def key(r):
        return tuple(r[k] for k in ['split','domain','index','topology','mode','repeat'])
    observed=Counter(key(r) for r in rows)
    if set(observed)!=expected or any(n!=1 for n in observed.values()):
        raise ValueError('missing, extra or duplicated workflow measurement')
    paired=[]
    for topology in ['chain','fork_join','loop']:
        pairs=[]
        for domain in domains:
            for i in range(6):
                arms=[]
                for mode in ['original','calibrated']:
                    arm=[r for r in rows if (r['split'],r['domain'],r['index'],r['topology'],r['mode'])
                         ==('heldout',domain,i,topology,mode)]
                    arms.append(statistics.fmean(r['elapsed_s'] for r in arm))
                pairs.append(arms)
        rng=random.Random(20260916)
        samples=[]
        for _ in range(4000):
            picked=rng.choices(pairs,k=len(pairs))
            samples.append(100*(1-sum(x[1] for x in picked)/sum(x[0] for x in picked)))
        samples.sort()
        paired.append({'topology':topology,'unit':'domain and case; repetitions averaged',
                       'n':len(pairs),'reduction_pct':100*(1-sum(x[1] for x in pairs)/sum(x[0] for x in pairs)),
                       'reduction_95ci_pct':[samples[100],samples[3900]],'resamples':4000})
    return {'synthetic':True,'workflow_rows':len(rows),'all_pairs_valid':True,
            'ok':sum(r['ok'] for r in rows),'exact':sum(r['exact'] for r in rows),
            'observed_model_calls':sum(r.get('calls',0) for r in rows),
            'fallbacks':sum(r.get('fallbacks',0) for r in rows),
            'paired':paired,'topologies':summary['topologies'],
            'limitations':['Generated filtering tasks with UUIDs, not real-world task diversity.',
                           'Three async topologies, not interoperability tests of three frameworks.',
                           'Zero observed errors does not prove zero future error probability.']}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    args=parser.parse_args()
    result=analyze(args.directory)
    (args.directory/'analysis.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k not in ['paired','topologies','limitations']}))

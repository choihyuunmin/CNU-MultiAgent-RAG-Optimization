"""Question-clustered uncertainty for trace-validated completion differences."""
import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics
from moleg_paper_metrics import paired_difference_ci


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args();root=a.directory
    outcomes=json.loads((root/'stage-outcomes.json').read_text())
    if not outcomes['complete'] or outcomes['audit_errors']:raise ValueError('incomplete stage audit')
    rows=[json.loads(line) for line in (root/'audited-requests.jsonl').read_text().splitlines()]
    results=[]
    for level in sorted({r['level'] for r in rows}):
        base=[r for r in rows if r['level']==level and r['arm']=='baseline']
        b={(r['case_id'],r['repeat']):r for r in base}
        if len(base)!=len(b):raise ValueError('duplicate baseline observation')
        for arm in sorted({r['arm'] for r in rows}-{'baseline'}):
            selected=[r for r in rows if r['level']==level and r['arm']==arm]
            c={(r['case_id'],r['repeat']):r for r in selected}
            if len(selected)!=len(c) or set(c)!=set(b):raise ValueError('invalid paired completion set')
            comparisons={}
            for field in ['stage_clean_completion','dependency_clean_completion']:
                grouped=defaultdict(list)
                for key,x in b.items():
                    y=c[key]
                    if x['question_sha256']!=y['question_sha256']:raise ValueError('question mismatch')
                    grouped[key[0]].append((float(x[field]),float(y[field])))
                comparisons[field]=paired_difference_ci([(statistics.fmean(x for x,y in vs),statistics.fmean(y for x,y in vs)) for vs in grouped.values()])
            results.append({'level':level,'candidate':arm,'metrics':comparisons})
    output={'scope':'Trace-based completion differences, not expert correctness. Questions cluster repetitions; two runs do not estimate all shared-server variation.', 'comparisons':results}
    (root/'completion-comparison.json').write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output))

if __name__=='__main__':main()

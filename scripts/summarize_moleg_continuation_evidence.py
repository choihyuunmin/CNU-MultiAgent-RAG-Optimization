"""Evidence-ID fidelity and baseline variability from public hashed request rows."""
import argparse
import json
from pathlib import Path
from analyze_moleg_structured import analyze


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    a=p.parse_args()
    rows=[json.loads(line) for line in (a.directory/'requests.jsonl').read_text().splitlines()]
    result={'scope':'Returned evidence item IDs, not parent-law IDs or expert accuracy. Empty reference is unscorable; empty candidate against nonempty reference scores zero.', 'comparisons':{}}
    fields=['concurrency','unique_questions','candidate_recall_vs_baseline','scorable_recall_pairs',
            'candidate_precision_vs_baseline','exact_evidence_fraction','baseline_repeat_recall',
            'baseline_repeat_exact','source_hit_change']
    for candidate in sorted({r['arm'] for r in rows}-{'baseline'}):
        data=analyze(a.directory,candidate=candidate)
        if not data['complete']:raise ValueError('unfinished study')
        result['comparisons'][candidate]=[{k:load[k] for k in fields} for load in data['loads']]
    (a.directory/'evidence-comparison.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result))

if __name__=='__main__':main()

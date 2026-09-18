"""Paired end-to-end analysis with repeated-baseline variability and explicit limits."""
from __future__ import annotations

import argparse
from collections import defaultdict
import json
from pathlib import Path
import statistics

from moleg_paper_metrics import id_recall, paired_cluster_ci, paired_difference_ci
from moleg_scaling_metrics import describe


def mean(values):
    values = [v for v in values if v is not None]
    return statistics.fmean(values) if values else None


def analyze(root, baseline='baseline', candidate='improved'):
    rows = [json.loads(x) for x in (root/'requests.jsonl').read_text().splitlines()]
    summary = json.loads((root/'summary.json').read_text())
    result = {'complete': summary['complete'], 'n': len(rows), 'loads': [],
              'quality_scope': 'source hit is a silver known-item label; evidence overlap is baseline fidelity, not accuracy',
              'latency_scope': 'client-observed completion/failure time including all waiting; repeated queries are clustered',
              'failures': [{k:r.get(k) for k in ('case_id','arm','level','repeat','ok','pipeline_ok','error_type')}
                           for r in rows if not r.get('pipeline_ok')],
              'trials': summary['trials']}
    for load in sorted({r['level'] for r in rows}):
        groups = {}
        for arm in (baseline, candidate):
            selected = [r for r in rows if r['level'] == load and r['arm'] == arm]
            grouped = {(r['case_id'],r['repeat']):r for r in selected}
            if len(grouped) != len(selected):
                raise ValueError('duplicate paired observation')
            groups[arm] = grouped
        b, c = groups[baseline], groups[candidate]
        if not b or b.keys() != c.keys():
            raise ValueError('incomplete paired set')
        by_query, hits = defaultdict(list), defaultdict(list)
        recalls, precision, exact = [], [], []
        for key in sorted(b):
            x, y = b[key], c[key]
            if x['question_sha256'] != y['question_sha256']:
                raise ValueError('paired question differs')
            by_query[key[0]].append((x['elapsed_s'],y['elapsed_s']))
            if x.get('source_law_hit') is not None and y.get('source_law_hit') is not None:
                hits[key[0]].append((float(x['source_law_hit']),float(y['source_law_hit'])))
            recalls.append(id_recall(x['evidence_ids_sha256'],y['evidence_ids_sha256']))
            precision.append(id_recall(y['evidence_ids_sha256'],x['evidence_ids_sha256']))
            exact.append(x['evidence_ids_sha256'] == y['evidence_ids_sha256'])
        pairs = [(mean(x for x,y in v),mean(y for x,y in v)) for v in by_query.values()]
        repeat_pairs = []
        for query in by_query:
            values = sorted((rep,r) for (q,rep),r in b.items() if q == query)
            for (_,x),(_,y) in zip(values,values[1:]):
                repeat_pairs.append((x,y))
        per_arm = {}
        for arm,g in groups.items():
            z = list(g.values())
            trials = [t for t in summary['trials'] if t['level']==load and t['arm']==arm]
            wall = sum(t['wall_s'] for t in trials)
            success = sum(r.get('pipeline_ok',False) for r in z)
            per_arm[arm] = {'n':len(z),'sse_success':sum(r['ok'] for r in z),
                'pipeline_success':success, 'latency_including_failures':describe([r['elapsed_s'] for r in z]),
                'successful_latency':describe([r['elapsed_s'] for r in z if r.get('pipeline_ok')]),
                'throughput_rps':success/wall, 'goodput_30s_rps':sum(r.get('pipeline_ok') and r['elapsed_s']<=30 for r in z)/wall,
                'source_law_hit':mean(float(r['source_law_hit']) for r in z if r.get('source_law_hit') is not None)}
        result['loads'].append({'concurrency':load, 'unique_questions':len(by_query),
            'latency':paired_cluster_ci(pairs),'arms':per_arm,
            'candidate_recall_vs_baseline':mean(recalls), 'scorable_recall_pairs':sum(v is not None for v in recalls),
            'candidate_precision_vs_baseline':mean(precision),'exact_evidence_fraction':mean(exact),
            'baseline_repeat_recall':mean(id_recall(x['evidence_ids_sha256'],y['evidence_ids_sha256']) for x,y in repeat_pairs),
            'baseline_repeat_exact':mean(x['evidence_ids_sha256']==y['evidence_ids_sha256'] for x,y in repeat_pairs),
            'source_hit_change':paired_difference_ci([(mean(x for x,y in v),mean(y for x,y in v)) for v in hits.values()])})
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--directory',type=Path,required=True)
    p.add_argument('--candidate',default='improved')
    args=p.parse_args()
    result=analyze(args.directory,candidate=args.candidate)
    (args.directory/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    for row in result['loads']:
        print(json.dumps({'concurrency':row['concurrency'],'latency':row['latency'],
                          'recall':row['candidate_recall_vs_baseline'],
                          'baseline_repeat_recall':row['baseline_repeat_recall']}))


if __name__=='__main__':main()

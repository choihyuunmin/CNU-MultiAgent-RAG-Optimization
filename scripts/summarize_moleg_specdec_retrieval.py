"""Join retrieval evaluations (per unique preparation output) back to arms and
compute source-label metrics per arm, paired against a reference arm."""
from __future__ import annotations
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path
from moleg_paper_metrics import paired_difference_ci, mcnemar_exact, means


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--evaluated', type=Path, required=True)
    ap.add_argument('--origins', type=Path, required=True)
    ap.add_argument('--reference', required=True, help='origin key file:variant:repeat used as reference')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    ev = {}
    for line in args.evaluated.read_text().splitlines():
        r = json.loads(line); ev[(r['case_id'], r['normalized_sha256'])] = r
    origins = json.loads(args.origins.read_text())
    arms = defaultdict(dict)
    for o in origins:
        key = f"{o['file']}:{o['variant']}:{o['repeat']}"
        arms[key][o['case_id']] = ev.get((o['case_id'], o['normalized_sha256']))
    out = {'arms': {}, 'comparisons': {}, 'unique_outputs_evaluated': len(ev),
           'evaluation_failures': sum(not r.get('ok') for r in ev.values())}
    for key, per_case in arms.items():
        ms = [r['metrics'] for r in per_case.values() if r and r.get('ok') and r.get('metrics')]
        cand = [r['source_law_in_candidates'] for r in per_case.values() if r and r.get('ok') and 'source_law_in_candidates' in r]
        out['arms'][key] = {'questions': len(per_case), 'ok': sum(bool(r and r.get('ok')) for r in per_case.values()),
                            'labeled_law': len(ms), 'metrics': means(ms),
                            'source_law_in_candidates': statistics.fmean(cand) if cand else None,
                            'search_s_mean': statistics.fmean(r['search_s'] for r in per_case.values() if r and r.get('ok'))}
    ref = arms[args.reference]
    for key, per_case in arms.items():
        if key == args.reference:
            continue
        mrr, hit1, law_hit1, law_cand, same = [], [], [], [], []
        for cid, r in per_case.items():
            b = ref.get(cid)
            if not (r and b and r.get('ok') and b.get('ok')):
                continue
            same.append(r['normalized_sha256'] == b['normalized_sha256'])
            if r.get('metrics') and b.get('metrics'):
                if b['metrics']['chunk_mrr20'] is not None:
                    mrr.append((b['metrics']['chunk_mrr20'], r['metrics']['chunk_mrr20']))
                    hit1.append((b['metrics']['chunk_hit1'], r['metrics']['chunk_hit1']))
                law_hit1.append((b['metrics']['law_hit1'], r['metrics']['law_hit1']))
                law_cand.append((b['source_law_in_candidates'], r['source_law_in_candidates']))
        out['comparisons'][f'{args.reference}->{key}'] = {
            'questions_compared': len(same), 'identical_preparation_output': statistics.fmean(same) if same else None,
            'chunk_mrr20_change': paired_difference_ci(mrr), 'chunk_hit1_mcnemar': mcnemar_exact(hit1),
            'law_hit1_mcnemar': mcnemar_exact(law_hit1), 'law_in_candidates_mcnemar': mcnemar_exact(law_cand)}
    args.output.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    for k, v in out['arms'].items():
        m = v['metrics']
        print(k, 'q', v['questions'], 'law_n', v['labeled_law'], {kk: round(m[kk], 4) for kk in ['chunk_hit1', 'chunk_hit5', 'chunk_hit20', 'chunk_mrr20', 'law_hit1', 'law_hit20'] if m.get(kk) is not None}, 'law_cand', round(v['source_law_in_candidates'] or 0, 4))
    for k, v in out['comparisons'].items():
        print(k, 'identical', round(v['identical_preparation_output'] or 0, 4), 'hit1', v['chunk_hit1_mcnemar'], 'law_hit1', v['law_hit1_mcnemar'], 'mrr_delta', round(v['chunk_mrr20_change'].get('delta') or 0, 4), v['chunk_mrr20_change'].get('delta_95ci'))


if __name__ == '__main__':
    main()

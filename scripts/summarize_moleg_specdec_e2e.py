"""Summarize the E2E streaming rows (orig / direct / spec app processes)."""
from __future__ import annotations
import argparse, json, statistics
from collections import defaultdict, Counter
from pathlib import Path
from moleg_paper_metrics import paired_cluster_ci, paired_difference_ci, mcnemar_exact, id_recall, timing


def read_rows(paths):
    rows = []
    for p in paths:
        rows += [json.loads(l) for l in Path(p).read_text().splitlines()]
    return rows


def ids_of(row):
    return [str(x.get('item_id') or x.get('id') or '') for x in (row.get('result') or {}).get('laws', [])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=Path, action='append', required=True)
    ap.add_argument('--cases', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    cases = {c['case_id']: c for c in json.loads(args.cases.read_text())}
    rows = read_rows(args.rows)
    groups = defaultdict(list)
    for r in rows:
        groups[r['variant']].append(r)
    out = {'rows': len(rows), 'variants': {}, 'comparisons': {}}
    queries = defaultdict(lambda: defaultdict(list))
    for name, records in groups.items():
        inc = []
        for r in records:
            queries[name][r['case_id']].append(r)
            case = cases[r['case_id']]
            if case.get('source_id'):
                ids = ids_of(r)
                inc.append({'source_law_in_returned_evidence': float(any(x.split('_')[0] == case['source_law_id'] for x in ids)),
                            'source_chunk_in_returned_evidence': float(case['source_id'] in ids) if case.get('reference_answer') else None})
        kinds = {}
        for kind in sorted({r['kind'] for r in records}):
            sel = [r for r in records if r['kind'] == kind]
            kinds[kind] = {'n': len(sel), 'success': sum(r['ok'] for r in sel), 'latency': timing([r['elapsed_s'] for r in sel]),
                           'ttft': timing([r.get('ttft_s') for r in sel])}
        out['variants'][name] = {
            'n': len(records), 'success': sum(r['ok'] for r in records), 'status_counts': dict(Counter(r.get('status') for r in records)),
            'latency_including_failures': timing([r['elapsed_s'] for r in records]),
            'ttft': timing([r.get('ttft_s') for r in records]),
            'source_law_in_returned_evidence': statistics.fmean(x['source_law_in_returned_evidence'] for x in inc) if inc else None,
            'source_law_n': len(inc),
            'source_chunk_in_returned_evidence': statistics.fmean(v) if (v := [x['source_chunk_in_returned_evidence'] for x in inc if x['source_chunk_in_returned_evidence'] is not None]) else None,
            'source_chunk_n': len(v),
            'mean_answer_chars': statistics.fmean(len((r.get('result') or {}).get('comment') or '') for r in records),
            'empty_answer_count': sum(not (r.get('result') or {}).get('comment') for r in records),
            'hitl_count': sum(bool((r.get('result') or {}).get('hitl')) for r in records),
            'by_kind': kinds}
    for ref in groups:
        for name in groups:
            if name == ref:
                continue
            pairs, ttft, recalls, exact, src_pairs, src_bin, chunk_pairs = [], [], [], [], [], [], []
            for q, refs in queries[ref].items():
                cands = queries[name].get(q, [])
                if not cands:
                    continue
                pairs.append((statistics.fmean(r['elapsed_s'] for r in refs), statistics.fmean(r['elapsed_s'] for r in cands)))
                bt = [r['ttft_s'] for r in refs if r.get('ttft_s') is not None]
                ct = [r['ttft_s'] for r in cands if r.get('ttft_s') is not None]
                if bt and ct:
                    ttft.append((statistics.fmean(bt), statistics.fmean(ct)))
                case = cases[q]
                if case.get('source_id'):
                    def inc(row, chunk=False):
                        ids = ids_of(row)
                        return float(case['source_id'] in ids) if chunk else float(any(x.split('_')[0] == case['source_law_id'] for x in ids))
                    src_pairs.append((statistics.fmean(inc(r) for r in refs), statistics.fmean(inc(r) for r in cands)))
                    src_bin.append((inc(refs[0]), inc(cands[0])))
                    if case.get('reference_answer'):
                        chunk_pairs.append((statistics.fmean(inc(r, True) for r in refs), statistics.fmean(inc(r, True) for r in cands)))
                for b in refs:
                    c = next((x for x in cands if x['repeat'] == b['repeat']), None)
                    if c is None:
                        continue
                    a, z = ids_of(b), ids_of(c)
                    rec = id_recall(a, z)
                    if rec is not None:
                        recalls.append(rec); exact.append(a == z)
            out['comparisons'][f'{ref}->{name}'] = {
                'latency': paired_cluster_ci(pairs), 'ttft': paired_cluster_ci(ttft),
                'source_law_in_returned_evidence_change': paired_difference_ci(src_pairs),
                'source_chunk_in_returned_evidence_change': paired_difference_ci(chunk_pairs),
                'source_law_mcnemar': mcnemar_exact(src_bin),
                'document_recall_vs_reference': statistics.fmean(recalls) if recalls else None,
                'exact_returned_ids': statistics.fmean(exact) if exact else None, 'nonempty_reference_pairs': len(recalls)}
    args.output.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    for name, v in out['variants'].items():
        print(name, 'n', v['n'], 'ok', v['success'], 'mean', round(v['latency_including_failures']['mean_s'], 3),
              'p95', round(v['latency_including_failures']['p95_s'], 3), 'ttft', round(v['ttft']['mean_s'] or 0, 3),
              'law_inc', v['source_law_in_returned_evidence'], 'chunk_inc', v['source_chunk_in_returned_evidence'])
    for k, v in out['comparisons'].items():
        print(k, 'speedup', round(v['latency'].get('speedup') or 0, 3), 'red%', round(v['latency'].get('reduction_pct') or 0, 2),
              'ci', [round(x, 2) for x in (v['latency'].get('reduction_95ci_pct') or [])], 'ttft_speedup', round(v['ttft'].get('speedup') or 0, 3),
              'docrecall', v['document_recall_vs_reference'], 'exact', v['exact_returned_ids'])


if __name__ == '__main__':
    main()

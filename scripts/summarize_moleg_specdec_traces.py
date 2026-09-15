"""Per-call breakdown from the isolated app traces: orchestrator call positions
(1st = intent classification, 2nd = search preparation, 3rd = selection for
law-search requests), worker streaming, guardrail, retrieval and rerank."""
from __future__ import annotations
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path
from moleg_paper_metrics import timing, paired_cluster_ci


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--trace', action='append', required=True, help='name=path')
    ap.add_argument('--e2e', type=Path, action='append', required=True, help='e2e rows to map session -> case')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    session_case = {}
    for p in args.e2e:
        for line in p.read_text().splitlines():
            r = json.loads(line); session_case[r['session_id']] = (r['case_id'], r['variant'], r['repeat'])
    out = {'arms': {}, 'paired_orchestrator_total': {}}
    per_case_orch = defaultdict(dict)
    for spec in args.trace:
        name, path = spec.split('=', 1)
        rows = [json.loads(l) for l in Path(path).read_text().splitlines()]
        rows = [r for r in rows if r['session_id'] in session_case]
        pos = defaultdict(list); tokens = defaultdict(list); other = defaultdict(list); orch_total = []
        for r in rows:
            calls = [c for c in r.get('llm', []) if c.get('model') == 'orchestrator' and not c.get('stream')]
            for i, c in enumerate(calls):
                pos[f'orchestrator_call_{i+1}'].append(c['elapsed_s'])
                if c.get('usage'):
                    tokens[f'orchestrator_call_{i+1}'].append(c['usage'].get('completion_tokens', 0))
            orch_total.append(sum(c['elapsed_s'] for c in calls))
            per_case_orch[session_case[r['session_id']][0]][name] = sum(c['elapsed_s'] for c in calls)
            for c in r.get('llm', []):
                if c.get('stream'):
                    other['worker_stream'].append(c['elapsed_s'])
                elif c.get('model') != 'orchestrator':
                    other['other_llm'].append(c['elapsed_s'])
            for kind in ['guardrail', 'retrieval', 'rerank']:
                for c in r.get(kind, []):
                    other[kind].append(c['elapsed_s'])
        out['arms'][name] = {'requests': len(rows), 'orchestrator_calls_per_request': statistics.fmean(len([c for c in r.get('llm', []) if c.get('model') == 'orchestrator' and not c.get('stream')]) for r in rows) if rows else None,
                             'orchestrator_total_per_request': timing(orch_total),
                             'by_position': {k: {'latency': timing(v), 'mean_completion_tokens': statistics.fmean(tokens[k]) if tokens[k] else None,
                                                 'tokens_per_second': (sum(tokens[k]) / sum(v)) if tokens[k] and sum(v) else None} for k, v in pos.items()},
                             'other': {k: timing(v) for k, v in other.items()}}
    names = list(out['arms'])
    for ref in names:
        for cand in names:
            if ref == cand:
                continue
            pairs = [(v[ref], v[cand]) for v in per_case_orch.values() if ref in v and cand in v]
            out['paired_orchestrator_total'][f'{ref}->{cand}'] = paired_cluster_ci(pairs)
    args.output.write_text(json.dumps(out, indent=1) + '\n')
    for name, a in out['arms'].items():
        print(name, 'req', a['requests'], 'orch_total_mean', round(a['orchestrator_total_per_request']['mean_s'] or 0, 3),
              {k: (round(v['latency']['mean_s'], 3), round(v['tokens_per_second'] or 0, 1)) for k, v in a['by_position'].items()})
    for k, v in out['paired_orchestrator_total'].items():
        print(k, 'speedup', round(v.get('speedup') or 0, 3), 'red%', round(v.get('reduction_pct') or 0, 2), v.get('reduction_95ci_pct'))


if __name__ == '__main__':
    main()

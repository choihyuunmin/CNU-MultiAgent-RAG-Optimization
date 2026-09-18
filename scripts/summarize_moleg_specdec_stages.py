"""Aggregate stage-replay rows: paired per-question latency comparison between
variants, output agreement, and speculative-decoding acceptance from counters.
"""
from __future__ import annotations
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path
from moleg_paper_metrics import paired_cluster_ci, timing


def load(path):
    return [json.loads(l) for l in Path(path).read_text().splitlines()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=Path, required=True)
    ap.add_argument('--reference', required=True, help='reference variant name')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    rows = load(args.rows)
    blocks = load(args.rows.with_suffix('.blocks.jsonl')) if args.rows.with_suffix('.blocks.jsonl').exists() else []
    variants = sorted({r['variant'] for r in rows})
    stages = sorted({r['stage'] for r in rows})
    out = {'rows': len(rows), 'variants': variants, 'stages': stages, 'reference': args.reference,
           'http_ok': {v: sum(r['ok'] for r in rows if r['variant'] == v) for v in variants},
           'valid_json': {v: sum(bool(r.get('valid_json')) for r in rows if r['variant'] == v) for v in variants},
           'finish_reasons': {v: dict(defaultdict(int, {fr: sum(1 for r in rows if r['variant'] == v and r.get('finish_reason') == fr)
                              for fr in {r.get('finish_reason') for r in rows}})) for v in variants},
           'latency': {}, 'paired': {}, 'agreement': {}, 'tokens': {}, 'spec': {}}
    by = defaultdict(list)
    for r in rows:
        by[(r['variant'], r['stage'], r['case_id'])].append(r)
    for v in variants:
        for s in stages + ['all']:
            vals = [r['elapsed_s'] for r in rows if r['variant'] == v and (s == 'all' or r['stage'] == s)]
            out['latency'][f'{v}/{s}'] = timing(vals)
            toks = [r['usage']['completion_tokens'] for r in rows if r['variant'] == v and r.get('usage') and (s == 'all' or r['stage'] == s)]
            out['tokens'][f'{v}/{s}'] = {'mean_completion_tokens': statistics.fmean(toks) if toks else None}
    ref = args.reference
    for v in variants:
        if v == ref:
            continue
        for s in stages + ['all']:
            pairs = []
            for (vv, ss, cid), rs in by.items():
                if vv != ref or (s != 'all' and ss != s):
                    continue
                cand = by.get((v, ss, cid))
                if not cand:
                    continue
                pairs.append((statistics.fmean(r['elapsed_s'] for r in rs), statistics.fmean(r['elapsed_s'] for r in cand)))
            out['paired'][f'{ref}->{v}/{s}'] = paired_cluster_ci(pairs)
    # agreement: normalized JSON equality between variants (repeat 0 vs repeat 0) and within-variant repeats
    def hashes(v, rep):
        return {(r['stage'], r['case_id']): r.get('normalized_sha256') for r in rows if r['variant'] == v and r['repeat'] == rep and r['ok']}
    reps = sorted({r['repeat'] for r in rows})
    for v in variants:
        for s in stages:
            if len(reps) >= 2:
                a, b = hashes(v, reps[0]), hashes(v, reps[1])
                common = [k for k in a if k in b and k[0] == s]
                out['agreement'][f'{v} repeat{reps[0]} vs repeat{reps[1]}/{s}'] = {'n': len(common), 'identical': sum(a[k] == b[k] for k in common)}
            if v != ref:
                for rep in reps:
                    a, b = hashes(ref, rep), hashes(v, rep)
                    common = [k for k in a if k in b and k[0] == s]
                    out['agreement'][f'{ref} vs {v} repeat{rep}/{s}'] = {'n': len(common), 'identical': sum(a[k] == b[k] for k in common)}
    # speculative counters per variant (sum of deltas over blocks)
    for v in variants:
        d = defaultdict(float)
        for b in blocks:
            if b['variant'] != v:
                continue
            cb, ca = b.get('counters_before', {}), b.get('counters_after', {})
            for k in ca:
                if k in cb and isinstance(ca[k], (int, float)) and isinstance(cb[k], (int, float)):
                    d[k] += ca[k] - cb[k]
        drafts, dtoks, acc = d.get('vllm:spec_decode_num_drafts_total', 0), d.get('vllm:spec_decode_num_draft_tokens_total', 0), d.get('vllm:spec_decode_num_accepted_tokens_total', 0)
        gen = d.get('vllm:generation_tokens_total', 0)
        out['spec'][v] = {'drafts': drafts, 'draft_tokens': dtoks, 'accepted_tokens': acc,
                          'acceptance_rate': acc / dtoks if dtoks else None,
                          'mean_accepted_per_draft': acc / drafts if drafts else None,
                          'generation_tokens': gen, 'preemptions': d.get('vllm:num_preemptions_total', 0),
                          'prefix_hit_rate': (d.get('vllm:prefix_cache_hits_total', 0) / d['vllm:prefix_cache_queries_total']) if d.get('vllm:prefix_cache_queries_total') else None,
                          'accepted_per_position': {k: d[k] for k in sorted(d) if k.startswith('accepted_pos_')}}
    args.output.write_text(json.dumps(out, indent=1, ensure_ascii=False) + '\n')
    for k, v in out['paired'].items():
        print(k, 'speedup', round(v.get('speedup') or 0, 3), 'reduction%', round(v.get('reduction_pct') or 0, 2), 'ci', [round(x, 2) for x in (v.get('reduction_95ci_pct') or [])])
    for k, v in out['agreement'].items():
        print(k, v)
    for k, v in out['spec'].items():
        print(k, {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items() if kk != 'accepted_per_position'})


if __name__ == '__main__':
    main()

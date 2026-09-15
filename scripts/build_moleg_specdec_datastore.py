"""Build a per-stage response-history datastore for the experimental proposer
from replay rows of a *different* question fold (leakage-free by construction).
"""
from __future__ import annotations
import argparse, json, re
from pathlib import Path


def fold_of(case_id):
    return 'even' if int(re.sub(r'\D', '', case_id)) % 2 == 0 else 'odd'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--rows', type=Path, required=True)
    ap.add_argument('--requests', type=Path, required=True)
    ap.add_argument('--variant', required=True)
    ap.add_argument('--repeat', type=int, default=0)
    ap.add_argument('--source-fold', choices=['even', 'odd'], required=True, help='fold whose outputs go into the datastore')
    ap.add_argument('--tokenizer', required=True)
    ap.add_argument('--min-n', type=int, default=1)
    ap.add_argument('--max-n', type=int, default=4)
    ap.add_argument('--fingerprint-len', type=int, default=48)
    ap.add_argument('--fingerprint-offset', type=int, default=4, help='skip this many leading tokens (bos/turn markers or a shared chat-template header)')
    ap.add_argument('--eos-id', type=int, default=106, help='end-of-turn token appended to each history sequence; -1 to omit')
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    reqs = json.loads(args.requests.read_text())
    stage_fp = {}
    for r in reqs:
        if r['stage'] in stage_fp:
            continue
        text = tok.apply_chat_template(r['payload']['messages'], tokenize=False, add_generation_prompt=True,
                                       **r['payload'].get('chat_template_kwargs', {}))
        ids = tok.encode(text, add_special_tokens=False)
        # skip the very first tokens (bos/turn markers) so an off-by-one in
        # special-token handling on the server cannot break the fingerprint
        stage_fp[r['stage']] = ids[args.fingerprint_offset:args.fingerprint_offset + args.fingerprint_len]
    stages = {}
    counts = {}
    for line in args.rows.read_text().splitlines():
        r = json.loads(line)
        if r['variant'] != args.variant or r['repeat'] != args.repeat or not r.get('ok'):
            continue
        if fold_of(r['case_id']) != args.source_fold:
            continue
        fp = stage_fp[r['stage']]
        key = str(hash(tuple(fp)))
        stages.setdefault(key, {'name': r['stage'], 'fingerprint': fp, 'seqs': []})
        stages[key]['seqs'].append(tok.encode(r['answer'], add_special_tokens=False) + ([args.eos_id] if args.eos_id >= 0 else []))
        counts[r['stage']] = counts.get(r['stage'], 0) + 1
    args.output.write_text(json.dumps({'fingerprint_len': args.fingerprint_len + args.fingerprint_offset, 'min_n': args.min_n, 'max_n': args.max_n,
                                       'source_fold': args.source_fold, 'variant': args.variant, 'repeat': args.repeat,
                                       'stages': stages}))
    print(json.dumps({'source_fold': args.source_fold, 'sequences': counts}))


if __name__ == '__main__':
    main()

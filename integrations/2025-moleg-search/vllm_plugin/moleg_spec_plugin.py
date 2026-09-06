"""vLLM general plugin: stage-aware prompt-lookup speculation for structured
agent stages. Loaded only in an isolated experimental server via PYTHONPATH;
never installed into the production environment.

Policies (environment variables, read once per process):
  MOLEG_NGRAM_ORDER   = earliest (stock) | latest  -> which occurrence of the
                        longest matching n-gram in the prompt to copy from.
  MOLEG_SPEC_DATASTORE= path to JSON produced by build_moleg_specdec_datastore.py
                        (response history of *other* questions, per stage).
                        When set, requests whose prompt lookup finds no match
                        fall back to the stage datastore (ctx_then_ds).
  MOLEG_SPEC_MODE     = ctx_then_ds (datastore only when the prompt has no
                        match) | longest_any (the source with the longer
                        matching n-gram wins; ties go to the prompt).
  MOLEG_SPEC_STATS    = path to write proposal-source counters periodically.
"""
from __future__ import annotations
import json, os, time


def register():
    import numpy as np
    from numba import jit
    from vllm.v1.spec_decode import ngram_proposer as mod

    order = os.environ.get('MOLEG_NGRAM_ORDER', 'earliest')
    ds_path = os.environ.get('MOLEG_SPEC_DATASTORE')
    stats_path = os.environ.get('MOLEG_SPEC_STATS')
    mode = os.environ.get('MOLEG_SPEC_MODE', 'ctx_then_ds')
    if mode not in ('ctx_then_ds', 'longest_any'):
        raise ValueError('MOLEG_SPEC_MODE must be ctx_then_ds or longest_any')
    if order not in ('earliest', 'latest'):
        raise ValueError('MOLEG_NGRAM_ORDER must be earliest or latest')

    if order == 'latest':
        @jit(nopython=True)
        def _latest(origin_tokens, min_ngram, max_ngram, max_model_len, k):
            total_token = origin_tokens.shape[0]
            if total_token < min_ngram:
                return np.empty((0,), dtype=origin_tokens.dtype)
            k = min(k, max_model_len - total_token)
            if k <= 0:
                return np.empty((0,), dtype=origin_tokens.dtype)
            tokens = origin_tokens[::-1]
            lps = np.zeros(max_ngram, dtype=np.int32)
            longest_ngram = 0
            position = 0
            prev_lps = 0
            i = 1
            while i < total_token:
                if tokens[prev_lps] == tokens[i]:
                    prev_lps += 1
                    # strict '>' keeps the first hit in reversed order,
                    # i.e. the most recent occurrence in the original order.
                    if prev_lps > longest_ngram:
                        longest_ngram = prev_lps
                        position = i
                    if i < max_ngram:
                        lps[i] = prev_lps
                    if prev_lps == max_ngram:
                        prev_lps = lps[max_ngram - 1]
                    i += 1
                elif prev_lps != 0:
                    prev_lps = lps[prev_lps - 1]
                else:
                    i += 1
            if longest_ngram < min_ngram:
                return np.empty((0,), dtype=origin_tokens.dtype)
            start_position = total_token - 1 - position + longest_ngram
            k = min(k, total_token - start_position)
            return origin_tokens[start_position:start_position + k]
        mod._find_longest_matched_ngram_and_propose_tokens = _latest

    if not ds_path:
        return
    data = json.loads(open(ds_path).read())
    fp_len = data['fingerprint_len']
    stages = {}
    for fp_key, st in data['stages'].items():
        index = {}
        seqs = [list(map(int, s)) for s in st['seqs']]
        for src, seq in enumerate(seqs):
            for j in range(len(seq)):
                for n in range(data['min_n'], data['max_n'] + 1):
                    if j - n + 1 < 0:
                        continue
                    key = (n, tuple(seq[j - n + 1:j + 1]))
                    if key not in index:  # earliest occurrence in history order
                        index[key] = (src, j - n + 1)
        stages[tuple(map(int, st['fingerprint']))] = {'name': st['name'], 'seqs': seqs, 'index': index}
    fps = list(stages)
    counters = {'calls': 0, 'ctx_drafts': 0, 'ds_drafts': 0, 'ds_over_ctx': 0, 'no_draft': 0, 'unknown_stage': 0, 'stage_hits': {}, 'mode': mode}

    @jit(nopython=True)
    def _longest_ctx_match(origin_tokens, min_ngram, max_ngram):
        # Same KMP scan as the stock proposer, returning only the length of the
        # longest suffix n-gram (<= max_ngram) that re-occurs earlier.
        total_token = origin_tokens.shape[0]
        if total_token < min_ngram:
            return 0
        tokens = origin_tokens[::-1]
        lps = np.zeros(max_ngram, dtype=np.int32)
        longest_ngram = 0
        prev_lps = 0
        i = 1
        while i < total_token:
            if tokens[prev_lps] == tokens[i]:
                prev_lps += 1
                if prev_lps >= longest_ngram:
                    longest_ngram = prev_lps
                if i < max_ngram:
                    lps[i] = prev_lps
                if prev_lps == max_ngram:
                    prev_lps = lps[max_ngram - 1]
                i += 1
            elif prev_lps != 0:
                prev_lps = lps[prev_lps - 1]
            else:
                i += 1
        return longest_ngram if longest_ngram >= min_ngram else 0
    _longest_ctx_match(np.array([1, 2, 3, 1, 2], dtype=np.int32), 1, 4)
    original = mod.NgramProposer.batch_propose

    def find_stage(row, num_tokens):
        head = row[:min(num_tokens, fp_len + 16)].tolist()
        for fp in fps:
            L = len(fp)
            for off in range(0, max(1, len(head) - L + 1)):
                if head[off:off + L] == list(fp):
                    return stages[fp]
        return None

    def batch_propose(self, num_requests, valid_ngram_requests, num_tokens_no_spec, token_ids_cpu):
        drafts = original(self, num_requests, valid_ngram_requests, num_tokens_no_spec, token_ids_cpu)
        counters['calls'] += 1
        for i in valid_ngram_requests:
            n_tok = int(num_tokens_no_spec[i])
            row = token_ids_cpu[i]
            if drafts[i] and mode == 'ctx_then_ds':
                counters['ctx_drafts'] += 1
                continue
            k = min(self.k, self.max_model_len - n_tok)
            if k <= 0:
                continue
            # longest_any: only a strictly longer datastore match may override
            # the prompt-lookup draft; ties keep the prompt draft.
            floor = self.min_n - 1
            if drafts[i]:
                floor = max(floor, int(_longest_ctx_match(row[:n_tok], self.min_n, self.max_n)))
            st = find_stage(row, n_tok)
            if st is None:
                counters['unknown_stage'] += 1
                if drafts[i]:
                    counters['ctx_drafts'] += 1
                else:
                    counters['no_draft'] += 1
                continue
            found = None
            for n in range(min(self.max_n, n_tok), floor, -1):
                key = (n, tuple(int(x) for x in row[n_tok - n:n_tok]))
                hit = st['index'].get(key)
                if hit is not None:
                    src, pos = hit
                    found = st['seqs'][src][pos + n:pos + n + k]
                    break
            if found:
                if drafts[i]:
                    counters['ds_over_ctx'] += 1
                drafts[i] = [int(x) for x in found]
                counters['ds_drafts'] += 1
                counters['stage_hits'][st['name']] = counters['stage_hits'].get(st['name'], 0) + 1
            elif drafts[i]:
                counters['ctx_drafts'] += 1
            else:
                counters['no_draft'] += 1
        if stats_path and counters['calls'] % 200 == 0:
            try:
                with open(stats_path, 'w') as f:
                    json.dump({**counters, 'utc': time.time()}, f)
            except OSError:
                pass
        return drafts
    mod.NgramProposer.batch_propose = batch_propose

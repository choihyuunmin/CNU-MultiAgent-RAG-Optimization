"""CPU checks for the offline speculative-decoding oracle simulation."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from simulate_moleg_specdec_policies import Context, propose, simulate, build_index


def reference(tokens, low, high, k):
    """Earliest-occurrence longest-n-gram lookup (vLLM semantics, validated 14,400x)."""
    for n in range(min(high, len(tokens)), low - 1, -1):
        suffix = tokens[-n:]
        for start in range(len(tokens) - n):
            if tokens[start:start + n] == suffix:
                return tokens[start + n:start + n + k]
    return []


def test_context_lookup_matches_reference_earliest():
    import random
    rng = random.Random(7)
    for _ in range(300):
        seq = [rng.randint(0, 5) for _ in range(rng.randint(3, 40))]
        low, high, k = 2, 4, 5
        ctx = Context(seq, low, high)
        draft, _ = propose(ctx, k, 'earliest')
        assert draft == reference(seq, low, high, k)


def test_full_copy_output_needs_few_steps():
    prompt = list(range(100, 130)) + [1, 2, 3]
    output = list(range(100, 130))  # verbatim copy of a prompt span after the 3-gram [1,2,3]? no: first token is sampled
    # make the copy start after a matchable suffix: append the trigger so lookup finds it
    prompt = [1, 2, 3] + list(range(100, 130)) + [9, 9, 1, 2, 3]
    r = simulate(prompt, output, k=8, min_n=2, max_n=4, order='earliest')
    # first token from prefill, then [3, 100] suffix matches -> copies 8 per step
    assert r['tokens'] == 30
    assert r['steps'] <= 1 + 4 + 1
    assert r['accepted'] > 0


def test_datastore_used_only_without_prompt_match():
    prompt = [5, 6, 7, 8]
    output = [50, 51, 52, 53, 54, 55]
    seqs = {'other': [50, 51, 52, 53, 54, 55, 56]}
    index = {}
    for cid, seq in seqs.items():
        build_index(seq, 2, 4, index, 0, cid)
    ds = {'seqs': seqs, 'index': index}
    plain = simulate(prompt, output, 5, 2, 4, 'earliest')
    with_ds = simulate(prompt, output, 5, 2, 4, 'earliest', ds, 'self')
    assert plain['steps'] == len(output)  # nothing to copy from the prompt
    assert with_ds['steps'] < plain['steps'] and with_ds['ds_steps'] >= 1
    # leave-one-out: excluding the only datastore source removes the gain
    excluded = simulate(prompt, output, 5, 2, 4, 'earliest', ds, 'other')
    assert excluded['steps'] == plain['steps']

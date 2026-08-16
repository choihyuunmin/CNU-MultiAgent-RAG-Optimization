# Accuracy-first server experiments

## Goal

Reduce inference latency without removing retrieval evidence, LLM stages, prompt content, or output tokens. Every variant uses same model weights, sampling settings, query IDs, and request payloads.

## Variants

| ID | Server change | One-line example | Expected target |
|---|---|---|---|
| V0 | Control | Run current vLLM arguments unchanged. | Reference |
| V1 | Automatic prefix caching | Reuse KV blocks for repeated static agent instructions. | TTFT, GPU work |
| V2 | Prefix cache + chunked prefill 8192 | Mix prefill and decode work under fixed token budget. | TTFT, throughput, p95 |
| V3 | Prefix cache + n-gram speculation 3 | Verify three proposed tokens with target model. | Decode time |
| V4 | Prefix cache + n-gram speculation 5 | Verify five proposed tokens with target model. | Decode time |

Profiles are exported by `cnu_rag_optimization.VLLM_PROFILES`. Append each profile's `server_args` to existing `vllm serve` command. Test one isolated deployment at a time; never switch production traffic during validation.

## Measurement protocol

1. Pin model revision, tokenizer revision, dtype, tensor parallelism, GPU clocks/power policy, sampling parameters, seed, concurrency, and question order.
2. Start isolated endpoint; wait until model health and GPU memory stabilize.
3. Run three excluded warm-ups.
4. Run same 400 queries: 20 source groups, four difficulty levels, 100 questions per level, concurrency 4.
5. Store request latency, TTFT, inter-token latency, prompt/decode tokens, vLLM queue time, prefix-cache hit rate, speculative acceptance, and response payload.
6. Compare against control with `scripts/evaluate_exact_regression.py`.
7. Reject variant on missing query, failure, retrieval regression, response mismatch above gate, or infrastructure timeout.

Example gate:

```bash
python scripts/evaluate_exact_regression.py \
  control.jsonl candidate.jsonl \
  --response-field response \
  --id-field law_ids
```

Default thresholds are 1.00 for request success, exact response, document-ID recall, and Top-1 agreement. If backend numerical nondeterminism prevents exact text repetition, first measure control-versus-control variance; do not relax retrieval thresholds below domain safety requirements.

## Non-applicable methods

- Persistent answer caching: excluded from primary 400-query experiment because unique questions get no legitimate cold-cache benefit and preloading pseudo-gold would leak evaluation answers.
- Smaller model, quantization, prompt truncation, stage removal, or lower output budgets: excluded from accuracy-preserving track; evaluate only as explicit accuracy/latency trade-off ablations.
- In-flight single-flight: valid production network optimization for duplicate concurrent requests, but expected gain is zero on 400 unique questions.

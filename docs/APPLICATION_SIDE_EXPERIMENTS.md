# Accuracy-first application-side experiments

## Constraint

Remote inference engine, model, scheduler, batching, decoding, and deployment settings remained unchanged. Only application routing and typed tool dispatch changed.

## Accepted variant

| Component | Behavior | Safety fallback |
|---|---|---|
| Confidence-gated local routing | Route an explicit retrieval request without an LLM classifier when entity, domain, action, and unique-route evidence all match. | Existing LLM router handles ambiguity, excluded intents, and conversation-dependent requests. |
| Typed single-tool dispatch | Send validated structured arguments directly when exactly one tool is available and already selected. | Existing LLM tool router handles multiple tools, missing arguments, mismatches, and failed validation. |

No retrieval, reranking, evidence, or final-answer stage is removed.

## One-line example

An explicit “find entity X records” request takes deterministic retrieval route; parsed typed arguments call sole search tool directly instead of asking two LLMs to repeat already-known decisions.

## Same-query protocol

1. Freeze question IDs, order, seed, concurrency, and warm-ups.
2. Run control and candidate against same remote inference deployment.
3. Store status, latency, selected document IDs, ranked results, and answer text.
4. Compare document-ID Recall, nDCG@10, Top-1 agreement, answer embedding similarity, and aggregate fidelity.
5. Reject candidate if any predefined aggregate gate fails.

Case-study gates: success 1.00, Recall >= 0.98, nDCG@10 >= 0.97, Top-1 >= 0.95, answer similarity >= 0.97, fidelity >= 0.97.

## Result

400 unique questions, 20 source groups, four difficulty levels with 100 questions each, concurrency 4, three excluded warm-ups, seed 20250809:

| Variant | Success | LLM calls | Mean | p50 | p95 | p99 | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| Seeded control | 400/400 | 2,007 | 16.525 s | 16.150 s | 37.775 s | 52.504 s | 0.240 req/s |
| Local route + typed dispatch | 400/400 | 1,375 | 14.850 s | 15.453 s | 35.620 s | 43.998 s | 0.268 req/s |

Candidate reduced LLM calls 31.5%, mean latency 10.1%, p95 5.7%, and p99 16.2%. Throughput rose 11.3%.

| Regression metric | Candidate |
|---|---:|
| Document-ID Recall | 0.9804 |
| nDCG@10 | 0.9778 |
| Top-1 agreement | 0.9650 |
| Answer similarity | 0.9756 |
| Aggregate fidelity | 0.9763 |

All predefined aggregate gates passed. Pseudo-gold regression proves preservation relative to current output, not domain-expert correctness.

Unchanged control rerun against an earlier control produced Recall 0.9756, nDCG@10 0.9746, Top-1 0.9650, answer similarity 0.9711, and aggregate fidelity 0.9726. Candidate metrics were equal or higher on every measure, so observed differences did not exceed measured repeat-run variability.

By difficulty, candidate mean latency changed from 1.254 to 1.309 seconds for simple single-agent lookup, 19.385 to 15.768 for conditional retrieval, 24.494 to 24.460 for cross-source comparison, and 20.965 to 17.863 for complex analysis. Optimization targets redundant multi-agent calls; already-single-call queries receive no benefit.

## Rejected application variants

| Variant | One-line behavior | Reason rejected |
|---|---|---|
| Direct endpoint routing | Bypass gateway and call existing role endpoint. | Slower in measured sample. |
| Verified search prefetch | Start predicted retrieval while LLM chooses arguments; reuse exact matches only. | Small speed gain, unstable small-sample regression. |
| Verified entity plan cache | Reuse a plan only after entity-independent equivalence is verified. | 5/20 hits; insufficient gain. |
| Alternate synthesis role | Move final synthesis to another existing model role. | Latency increased. |

Final publication should interleave control and candidate blocks and repeat each question at least three times to estimate load variance.

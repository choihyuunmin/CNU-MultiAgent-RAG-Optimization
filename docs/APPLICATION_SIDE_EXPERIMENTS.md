# Accuracy-first application-side experiments

## Constraint

Remote inference engine, model, scheduler, batching, decoding, and deployment settings remained unchanged. Only application routing and typed tool dispatch changed.

## Accepted foundation

| Component | Behavior | Safety fallback |
|---|---|---|
| Confidence-gated local routing | Route an explicit retrieval request without an LLM classifier when entity, domain, action, and unique-route evidence all match. | Existing LLM router handles ambiguity, excluded intents, and conversation-dependent requests. |
| Typed single-tool dispatch | Send validated structured arguments directly when exactly one tool is available and already selected. | Existing LLM tool router handles multiple tools, missing arguments, mismatches, and failed validation. |

No retrieval, reranking, evidence, or final-answer stage is removed.

## Latest combined variant

Safe hybrid adds three application-side techniques to accepted foundation:

| Component | Behavior | Safety fallback |
|---|---|---|
| Persistent bounded transport | Reuse HTTP/TCP connections for remote inference and retrieval calls. | Existing timeouts and error handling remain unchanged. |
| Parallel final enrichment | Start independent metadata enrichment beside final generation. | Await normal enrichment when prefetched result is unavailable. |
| Evidence-constrained routing | Use deterministic ranked selection only when candidates converge on one parent and pass scope, score, and keyword gates. | Existing LLM selector handles every uncertain or multi-source case. |

One-parent limit matters: a two-parent ablation changed ranked evidence and was rejected.

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

## 2026-08-23 rerun

Same-query validation used 400 unique questions: 100 at each of four difficulty levels. Concurrency was 4; three warm-ups were excluded. Indexed snapshot contained 3,203,485 source chunks, 456,428 translated chunks, and 19,583 title records.

| Variant | Success | LLM calls | Mean | p50 | p95 | p99 | Throughput |
|---|---:|---:|---:|---:|---:|---:|---:|
| Current control | 400/400 | 1,657 | 22.360 s | 22.846 s | 51.772 s | 107.571 s | 0.178 req/s |
| Safe hybrid | 400/400 | 1,650 | 19.938 s | 21.201 s | 47.850 s | 59.834 s | 0.199 req/s |
| Safe hybrid + bounded tail hedge | 400/400 | 1,668 | 19.762 s | 21.523 s | 46.998 s | 56.157 s | 0.201 req/s |

| Regression metric | Safe hybrid | Hybrid + hedge |
|---|---:|---:|
| Document-ID Recall | 0.9827 | 0.9816 |
| nDCG@10 | 0.9798 | 0.9766 |
| Top-1 agreement | 0.9700 | 0.9625 |
| Answer similarity | 0.9750 | 0.9766 |
| Aggregate fidelity | 0.9769 | 0.9766 |

Evidence routing fired on 16/400 safe-hybrid requests. Fired subgroup preserved document Recall, nDCG, and Top-1 at 1.000 while reducing paired latency 26.2%. Tail hedge fired on 17/400 requests, added 18 calls relative to safe hybrid, and improved mean only 0.9%. Safe hybrid is default candidate; tail hedge remains rejected ablation.

Unchanged control reruns previously produced aggregate fidelity around 0.9726 against an earlier control. Latest candidate fidelity stays inside measured generative repeatability. This is pseudo-gold preservation, not proof of domain-expert correctness.

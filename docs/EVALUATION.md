# Evaluation summary

## Case-study corpus snapshot

One production-scale domain corpus was used to validate the domain-independent optimization logic. OpenSearch `_count` snapshot collected on 2026-08-14:

| Index role | Documents |
|---|---:|
| Primary-source content chunks | 3,201,230 |
| Secondary-language content chunks | 457,873 |
| Title metadata documents | 19,583 |
| Total indexed documents | 3,678,686 |

Counts describe index documents, not manually deduplicated source entities. Other domains may substitute product records, security events, papers, manuals, tickets, or enterprise documents.

## Evaluation design

| Item | Method |
|---|---|
| Queries | 400 domain questions |
| Source groups | 20; each group appears 25 times |
| Difficulty | Simple lookup, conditional explanation, cross-source comparison, complex analysis; 100 each |
| Control | Same 400 question IDs for every variant |
| Load | Concurrency 4, three warm-ups excluded |
| Latency | mean, p50, p95, p99, throughput |
| Retrieval regression | Document-ID Recall, nDCG@10, Top-1 agreement |
| Answer regression | BGE-M3 cosine similarity |
| Reference | Stored current-system output used as pseudo-gold |

Pseudo-gold regression measures behavioral preservation. It is not domain-expert correctness.

## Acceptance gates

Application-level candidates require: success rate 1.00, document-ID Recall at least 0.98, nDCG@10 at least 0.97, Top-1 agreement at least 0.95, answer similarity at least 0.97, and aggregate pseudo-gold fidelity at least 0.97. Server-only candidates use stricter exact-output gates where deterministic execution permits it.

## Aggregate result: accepted application-side method

| Variant | Mean | p50 | p95 | p99 | Throughput | Fidelity | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| Seeded control | 16.525 s | 16.150 s | 37.775 s | 52.504 s | 0.240 req/s | Reference | Control |
| Confidence route + typed dispatch | 14.850 s | 15.453 s | 35.620 s | 43.998 s | 0.268 req/s | 0.9763 | Accepted |

Candidate reduced LLM calls from 2,007 to 1,375. Document-ID Recall 0.9804, nDCG@10 0.9778, Top-1 agreement 0.9650, and answer similarity 0.9756 passed predefined aggregate gates.

Unchanged control rerun against an earlier control reached Recall 0.9756, nDCG@10 0.9746, Top-1 0.9650, answer similarity 0.9711, and fidelity 0.9726. Candidate was no worse on these repeat-variance indicators. This does not replace expert-labeled correctness evaluation.

Application queue p95 stayed near 0.1 ms. Main remaining bottleneck was remote inference. Inference engine, model, scheduler, and decoding settings were not changed.

## Transfer to other domains

An integrating system supplies four neutral inputs:

1. ranked retrieval documents;
2. stable document IDs;
3. query features such as direct lookup and analysis intent;
4. number of source groups involved in the query.

No statute, country, language, vector database, LLM provider, or agent framework is assumed by the optimization package.

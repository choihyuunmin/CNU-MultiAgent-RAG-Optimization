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

## Aggregate result

| Variant | Mean | p50 | p95 | Throughput | Fidelity |
|---|---:|---:|---:|---:|---:|
| Current control recheck | 15.67 s | 17.20 s | 35.79 s | 0.254 req/s | 0.975 self-repeat |
| Token/input budget | 15.65 s | 16.32 s | 35.89 s | 0.254 req/s | 0.968 |
| Global selector + budget | 11.27 s | 12.39 s | 24.62 s | 0.353 req/s | 0.931 |
| Multi-source selector + budget | 13.45 s | 15.72 s | 27.59 s | 0.296 req/s | 0.958 |

Global selector + budget reduced mean latency by 28.1% (95% bootstrap CI 24.9% to 31.1%) and increased throughput by 39.1%. Prompt tokens fell 33.4%, total vLLM inference time fell 32.9%, and prefix-cache hit rate rose from 57.75% to 82.10%.

Queue time fell from 2.255 ms/request to 0.880 ms/request but was already negligible. Main bottleneck was inference, so reducing calls and tokens produced most gain.

## Transfer to other domains

An integrating system supplies four neutral inputs:

1. ranked retrieval documents;
2. stable document IDs;
3. query features such as direct lookup and analysis intent;
4. number of source groups involved in the query.

No statute, country, language, vector database, LLM provider, or agent framework is assumed by the optimization package.

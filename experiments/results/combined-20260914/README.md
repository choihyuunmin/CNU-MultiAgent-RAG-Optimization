# combined-harness-20260914 — public artifacts

Two-arm combined-harness experiment (2026-09-14). Metrics and hashes only; raw
questions, answers, and per-request texts are not included (kept in the private
server-side experiment directory). Silver source labels are not expert gold.

- `plan.json` — frozen plan: 200 stratified questions, arms, loads {1,4,16},
  2 repeats, seed, timeouts, `law_search_nodes_sha256`, runtime file SHA-256.
- `selected.json` — the 200 selected case ids, kinds, and question SHA-256.
- `summary.json` — per-trial latency metrics + shared-instance vLLM telemetry.
- `combined-analysis.json` — headline table, paired-bootstrap latency CIs per
  load, source-recall and evidence-exact changes, arm-level stage/reasoning
  breakdown, failure rows.
- `answer_judge_summary.json` — blinded phi-4 judge (65 reference-answer
  questions), relevance/support with paired CIs. Automated, not expert.
- `model-inventory-before.json` / `model-inventory-after.json` — the 6 relevant
  vLLM processes' ports and argv SHA-256; identical before and after (read-only
  backend, production untouched).

## Headline (paired, repetitions averaged, n=200/load)

| users | baseline mean s | combined mean s | reduction | 95% CI | evidence exact | law-recall Δ |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.68 | 8.61 | 19.3% | 12.5–29.0% | 0.967 | +0.004 |
| 4 | 13.99 | 12.38 | 11.5% | 8.3–14.8% | 0.935 | 0.000 |
| 16 | 32.26 | 30.52 | 5.4% | 1.5–9.9% | 0.952 | 0.000 |

All reductions significant. Judge: relevance 1.815→1.831, support 1.80→1.831
(both CIs include 0 → non-inferior). SSE success 2398/2400 (2 baseline tail
timeouts; combined 0 failures). Gain decays with load: bounded by the
orchestrator prepare+select ceiling, not the worker or search path.

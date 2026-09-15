# combined-sweep-20260915 — public artifacts

Method exploration + concurrency capacity ramp (2026-09-15). Metrics and hashes
only; raw questions/answers are not included. Silver source labels, not expert gold.

- `explore-summary.json` — Phase A: baseline vs reasoning-low vs
  reasoning-low+no-ontology at concurrency 4, 48 questions.
- `summary.json` — Phase B ramp: per-trial latency, success, goodput, and
  shared-instance vLLM telemetry for concurrency 1→300 (both arms). Field
  `hardware_limit_level` = 300.
- `campaign-status.json` — run status, levels, reference arm, hardware limit.
- `plan.json` — frozen 200-question plan + runtime SHA-256.
- `model-inventory-after.json` — the 6 relevant vLLM ports and argv SHA-256 after
  the run; identical to the 2026-09-14 baseline inventory (production unchanged).

## Findings

Method exploration: worker `reasoning_effort=low` is the winner (−7.9% at C=4,
recall 1.00). Dropping ontology search gave no latency or recall benefit, so it
was kept.

Capacity ramp — three limits on the shared 2×H200 orchestrator-bound stack:
1. **SLO-30 useful capacity knee ≈ 5–10 concurrent** (goodput peaks ~0.38 req/s at C=10).
2. **Raw throughput saturates ≈ 50–100 concurrent** (~0.6–0.66 req/s; orchestrator
   KV cache pinned 99–100% from C=20; preemptions from C=50).
3. **Hard collapse at 300 concurrent** (success 40–44%, mass 240s timeouts, ~130
   preemptions). Auto-stopped there; C=500 not run to spare the live service.

The reasoning-low arm is faster or equal at every load up to C=100; under KV
saturation the gain converges to noise. Source recall holds ≥0.985 through C=100;
the C=300 recall drop is a timeout side effect, not degraded retrieval.

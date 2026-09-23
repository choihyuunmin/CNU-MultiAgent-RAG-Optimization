# Combined comparison (2026-09-21): original path vs direct dispatch vs selection-input reduction vs both

This study runs the four conditions the earlier separate trials could not compare under one clock:

| Arm | Search-boundary path | Selection-input reduction |
|---|---|---|
| `original` | worker model forms the tool call | off |
| `direct` | checked direct dispatch | off |
| `reduce` | worker model forms the tool call | on (fields filtered, `content` cut at 300 characters) |
| `combined` | checked direct dispatch | on |

Same 200 legal-search questions, same frozen application image, models, prompts, and shared inference
engines. Each arm runs in its own application Pod (2 CPU / 8 GiB limits, admission capacity 100); a
separate client Pod drives closed-loop batches of 200 requests at 20, 50, and 100 concurrent requests.
Each load level has four rounds, and the four arms rotate through the four positions in a Latin square
(round 1: original, direct, reduce, combined; round 2: direct, reduce, combined, original; and so on),
so every arm runs once in every position at every load. Total: 48 cells, 9,600 requests, request
timeout 600 s, one preliminary smoke request per arm excluded. All 9,600 requests returned a
format-valid response; no failures or timeouts.

Law accuracy is evaluated with pooled relevance judgments (see "Law accuracy" below). The question set
carries no reference law IDs, so these judgments come from two models, not from legal experts.

## Results at a glance

- **Direct dispatch alone does not change end-to-end time** (+0.7%, +0.4%, +1.6% at loads 20/50/100; all
  intervals include zero), repeating the earlier finding.
- **Selection-input reduction alone** cuts the mean by 22.5%, 16.5%, and 14.4%.
- **The combination** cuts the mean by 30.0%, 23.1%, and 16.9% against the original path
  (18.5 s, 28.2 s, 28.4 s), and by 29.6%, 22.9%, and 15.6% against direct dispatch. At every load the
  95% lower bound of the relative mean gain exceeds the 5% release target and the p95-ratio upper bound
  is below 1.05.
- **The two changes interact.** Reduction saves more time on the direct path than on the original path:
  +4.2 s [2.4, 6.1] at load 20 and +7.7 s [4.1, 11.3] at load 50 (not significant at 100). Direct dispatch
  becomes useful once the orchestration engine is relieved.
- **Law accuracy (model judges):** precision of the returned laws is the same in all four arms at all
  loads (differences within ±0.02, intervals include zero). Recall against the pooled relevant set is
  unchanged at load 20 and higher for the reduced arms at loads 50 and 100, because those arms keep
  returning evidence where the original path returns empty or fallback answers (no-law responses at load
  100: original 323/800, direct 274, reduce 216, combined 120).
- **Law sets still differ.** Exact final law-set agreement between original and combined is 77.0%,
  71.0%, and 54.9% (same-arm repeats: 93.4%, 87.5%, 75.8%). The selection stage picks a different set in
  about 40–48% of requests when the input is reduced. The judges rate the swapped laws as equally relevant,
  which is evidence of preserved relevance, not proof of unchanged legal correctness.

## End-to-end response time (pooled over four rounds, 800 requests per row)

| Load | Arm | Mean (s) | Median (s) | p95 (s) | Format/s | Law-ID/s | No laws /800 |
|---|---|---|---|---|---|---|---|
| 20 | original | 61.71 | 61.14 | 108.45 | 0.2978 | 0.2587 | 105 |
| 20 | direct | 61.29 | 58.90 | 112.00 | 0.3047 | 0.2651 | 104 |
| 20 | reduce | 47.83 | 46.09 | 87.67 | 0.3974 | 0.3447 | 106 |
| 20 | combined | 43.17 | 40.62 | 80.79 | 0.4273 | 0.3712 | 105 |
| 50 | original | 121.74 | 127.27 | 193.17 | 0.3458 | 0.2779 | 157 |
| 50 | direct | 121.29 | 129.97 | 185.90 | 0.3455 | 0.2838 | 143 |
| 50 | reduce | 101.69 | 107.29 | 164.84 | 0.4058 | 0.3490 | 112 |
| 50 | combined | 93.57 | 97.53 | 153.65 | 0.4464 | 0.3867 | 107 |
| 100 | original | 167.75 | 170.38 | 255.33 | 0.4719 | 0.2813 | 323 |
| 100 | direct | 165.04 | 167.93 | 261.34 | 0.4624 | 0.3040 | 274 |
| 100 | reduce | 143.58 | 155.39 | 216.42 | 0.5268 | 0.3845 | 216 |
| 100 | combined | 139.35 | 147.02 | 221.91 | 0.5353 | 0.4550 | 120 |

Format/s divides format-valid responses by the sum of the four run times; Law-ID/s counts only responses
containing at least one law ID. Percentiles use linear interpolation.

## Paired comparisons (same question, same round; 800 pairs, 200 question clusters)

Saving is A minus B. Intervals are 95% question-cluster bootstrap intervals (4,000 resamples). "Questions
+/−" counts questions whose four-round average saving is positive/negative. "Gain lower bound" is the 2.5%
bootstrap quantile of the relative mean reduction. "Exact set" is the share of pairs with identical final
law-ID sets; "Recall of A's laws" is the macro recall of A's law IDs in B's response over nonempty A.

| Load | Comparison (A vs B) | Mean A → B (s) | Saving (s) | Saving | 95% CI (s) | Questions +/− | Gain lower bound | p95 ratio (upper) | Recall of A's laws | Exact set |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | original vs direct | 61.71 → 61.29 | +0.42 | +0.7% | [−1.19, +2.02] | 97/103 | −2.0% | 1.033 (1.114) | 97.1% | 88.8% |
| 20 | original vs reduce | 61.71 → 47.83 | +13.88 | +22.5% | [+11.51, +16.23] | 173/27 | +19.1% | 0.808 (0.910) | 93.4% | 76.1% |
| 20 | original vs combined | 61.71 → 43.17 | +18.54 | +30.0% | [+16.45, +20.64] | 190/10 | +27.4% | 0.745 (0.817) | 93.8% | 77.0% |
| 20 | direct vs combined | 61.29 → 43.17 | +18.12 | +29.6% | [+15.82, +20.31] | 188/12 | +26.5% | 0.721 (0.802) | 94.2% | 77.8% |
| 20 | reduce vs combined | 47.83 → 43.17 | +4.66 | +9.7% | [+3.47, +5.94] | 147/53 | +7.4% | 0.921 (1.005) | 98.5% | 92.8% |
| 50 | original vs direct | 121.74 → 121.29 | +0.45 | +0.4% | [−2.60, +3.68] | 107/93 | −2.2% | 0.962 (1.030) | 93.9% | 83.0% |
| 50 | original vs reduce | 121.74 → 101.69 | +20.05 | +16.5% | [+17.25, +22.91] | 163/37 | +14.4% | 0.853 (0.937) | 93.8% | 71.4% |
| 50 | original vs combined | 121.74 → 93.57 | +28.17 | +23.1% | [+24.33, +32.50] | 169/31 | +20.5% | 0.795 (0.851) | 93.8% | 71.0% |
| 50 | direct vs combined | 121.29 → 93.57 | +27.72 | +22.9% | [+24.51, +31.08] | 180/20 | +20.5% | 0.826 (0.861) | 94.1% | 72.0% |
| 50 | reduce vs combined | 101.69 → 93.57 | +8.12 | +8.0% | [+5.46, +10.92] | 118/82 | +5.5% | 0.932 (0.991) | 97.7% | 90.8% |
| 100 | original vs direct | 167.75 → 165.04 | +2.71 | +1.6% | [−0.77, +6.01] | 124/76 | −0.5% | 1.024 (1.060) | 88.1% | 76.6% |
| 100 | original vs reduce | 167.75 → 143.58 | +24.17 | +14.4% | [+19.75, +28.60] | 151/49 | +12.0% | 0.848 (0.896) | 81.8% | 56.9% |
| 100 | original vs combined | 167.75 → 139.35 | +28.40 | +16.9% | [+22.52, +33.93] | 144/56 | +13.8% | 0.869 (0.937) | 91.2% | 54.9% |
| 100 | direct vs combined | 165.04 → 139.35 | +25.70 | +15.6% | [+20.44, +31.03] | 140/60 | +12.7% | 0.849 (0.899) | 91.7% | 57.4% |
| 100 | reduce vs combined | 143.58 → 139.35 | +4.23 | +2.9% | [+0.32, +8.27] | 125/75 | +0.2% | 1.025 (1.090) | 96.1% | 77.6% |

Same-arm comparisons across rounds (rounds 1–2, 3–4, 1–3, 2–4; 800 pairs) show the size of drift between
runs of the same condition: −0.95 s [−1.80, −0.16], −0.21, +3.66 [+2.25, +5.26], +0.43 at load 20;
−1.24, +6.89 [+4.61, +9.20], +5.03 [+3.41, +6.67], −3.00 [−4.33, −1.69] at load 50; +1.35, +0.32, −1.43,
+1.43 at load 100 (original, direct, reduce, combined). Drift reaches 3–7 s at load 50; the reduction and
combination effects are three to nine times larger.

### Interaction between the two changes (2×2 factorial)

For each question and round, the saving from reduction on the direct path, (direct − combined), minus the
saving from reduction on the original path, (original − reduce):

| Load | Pairs | Extra saving on the direct path (s) | 95% CI |
|---|---|---|---|
| 20 | 800 | +4.25 | [+2.39, +6.13] |
| 50 | 800 | +7.67 | [+4.07, +11.25] |
| 100 | 800 | +1.53 | [−2.54, +5.83] |

Equivalently, the combined saving exceeds the sum of the two single savings by the same amounts. The
single-change savings therefore cannot be added; the combination must be measured, as it is here.

## Where the time goes (engine counters and boundary timing)

Shared inference engines, counters differenced over each cell and pooled per arm and load (orchestration
endpoint unless stated):

| Load | Arm | Orchestration requests | Prompt tokens/request | Queue/request (s) | Prefill/request (s) | Decode/request (s) | Running (mean) | KV max | Preemptions | Worker requests |
|---|---|---|---|---|---|---|---|---|---|---|
| 20 | original | 2,989 | 4,006 | 0.62 | 1.40 | 11.49 | 13.8 | 1.00 | 0 | 2,043 |
| 20 | direct | 2,980 | 4,027 | 0.38 | 1.32 | 12.49 | 15.2 | 1.00 | 0 | 1,173 |
| 20 | reduce | 2,984 | 3,144 | 0.01 | 0.46 | 5.91 | 9.3 | 0.97 | 0 | 2,049 |
| 20 | combined | 2,988 | 3,139 | 0.00 | 0.44 | 6.77 | 11.3 | 0.95 | 0 | 1,181 |
| 50 | original | 2,967 | 4,038 | 14.87 | 1.36 | 10.17 | 14.0 | 1.00 | 6 | 2,018 |
| 50 | direct | 2,965 | 4,038 | 16.82 | 1.76 | 15.11 | 20.2 | 1.00 | 12 | 1,156 |
| 50 | reduce | 2,957 | 3,162 | 3.74 | 0.81 | 10.44 | 16.1 | 1.00 | 12 | 2,010 |
| 50 | combined | 2,961 | 3,161 | 4.80 | 0.94 | 13.56 | 22.8 | 1.00 | 18 | 1,194 |
| 100 | original | 2,991 | 4,017 | 30.63 | 1.20 | 11.09 | 20.1 | 1.00 | 32 | 1,841 |
| 100 | direct | 2,952 | 4,043 | 31.68 | 1.18 | 13.12 | 22.1 | 1.00 | 26 | 1,020 |
| 100 | reduce | 2,866 | 3,238 | 7.53 | 0.74 | 10.24 | 19.5 | 1.00 | 22 | 1,905 |
| 100 | combined | 2,869 | 3,241 | 10.86 | 0.85 | 13.69 | 26.1 | 1.00 | 24 | 1,075 |

- Reduction cuts the selection input from about 11,900 to 5,030 characters per selection call (−58%) and the
  orchestration prompt from about 4,000 to 3,150 tokens per request; queue time per request falls from
  15–31 s to 4–11 s at loads 50 and 100, and decode time per request halves at load 20.
- Direct dispatch removes 42–43% of worker-endpoint requests and the 4–45 s of per-branch dispatch
  overhead on the original path (branch-weighted mean overhead outside the handler: original 4.25 s,
  22.7 s, 28.1 s; reduce 7.8 s, 24.3 s, 45.5 s; direct and combined about 1 ms at all loads), but on its own
  it only moves waiting into the orchestration queue (queue/request 0.62 → 0.38 s at load 20, 14.9 → 16.8 s
  at 50, 30.6 → 31.7 s at 100).
- The orchestration KV cache is at its ceiling in every cell; preemptions appear from load 50.

## Selection decisions and law-set agreement

Selection-stage decisions are compared by the hash of the selected set (pairs where both arms reached the
selection stage):

| Load | original vs direct | original vs reduce | original vs combined | reduce vs combined | same-arm repeats |
|---|---|---|---|---|---|
| 20 | 84.9% | 59.2% | 58.8% | 87.2% | 92.7–94.4% |
| 50 | 81.6% | 57.8% | 55.7% | 84.7% | 87.8–92.4% |
| 100 | 79.7% | 51.3% | 52.0% | 81.9% | 82.2–88.9% |

Direct dispatch leaves the selection decision within repeat variation; reduction changes it in about
40–48% of requests, whichever dispatch path is used. The "has relevant laws" flag agrees in over 99% of
pairs at loads 20 and 50 and in 81–83% at load 100 (repeats 93–100%).

## Law accuracy (pooled relevance judgments)

Method. All distinct law provisions returned by any arm in any round were pooled per question (174
questions with at least one returned provision; 1,395 question–provision pairs). Two models judged each
pair on the question text and the provision's country, title, subject, and first 600 characters, with a
0/1/2 rubric (0 unrelated or wrong country, 1 same subject but not the issue asked, 2 directly on point),
in shuffled chunks of at most eight provisions, temperature 0, thinking disabled. Judges: the orchestration
model (`master`, 31B) and the worker model (`tool_synthesis`, 20B). A 20% sample was re-judged in reverse
order to measure repeatability. Per response: precision = share of returned provisions judged ≥1;
recall = share of the question's pooled relevant provisions (judged ≥1) that the response contains;
"any relevant" = at least one relevant provision returned. Strict variants use score 2 only.

| Judge | Pairs | Score 2 / 1 / 0 | Repeat agreement (exact / relevant flag) | Agreement with other judge (exact / flag / kappa) |
|---|---|---|---|---|
| master | 1,395 | 617 / 559 / 219 | 80.7% / 96.0% | 53.9% / 75.3% / 0.37 |
| tool_synthesis | 1,395 | 413 / 499 / 483 | 67.9% / 83.5% | |

Per arm and load (means over 800 responses; master judge; second judge in parentheses):

| Load | Arm | Precision | Recall | Any relevant | Strict precision | Strict recall | Provisions returned | Empty |
|---|---|---|---|---|---|---|---|---|
| 20 | original | 0.843 (0.680) | 0.847 (0.868) | 0.985 (0.990) | 0.453 | 0.860 | 5.67 | 13.1% |
| 20 | direct | 0.848 (0.684) | 0.852 (0.872) | 0.988 (0.993) | 0.458 | 0.862 | 5.66 | 13.0% |
| 20 | reduce | 0.850 (0.677) | 0.839 (0.849) | 0.987 (0.990) | 0.462 | 0.865 | 5.57 | 13.3% |
| 20 | combined | 0.846 (0.674) | 0.839 (0.848) | 0.988 (0.991) | 0.460 | 0.868 | 5.58 | 13.1% |
| 50 | original | 0.837 (0.682) | 0.771 (0.794) | 0.904 (0.914) | 0.463 | 0.790 | 5.16 | 19.6% |
| 50 | direct | 0.843 (0.682) | 0.791 (0.810) | 0.926 (0.930) | 0.463 | 0.810 | 5.22 | 17.9% |
| 50 | reduce | 0.845 (0.672) | 0.835 (0.844) | 0.980 (0.979) | 0.462 | 0.861 | 5.53 | 14.0% |
| 50 | combined | 0.846 (0.671) | 0.840 (0.846) | 0.985 (0.985) | 0.457 | 0.865 | 5.59 | 13.4% |
| 100 | original | 0.835 (0.652) | 0.539 (0.557) | 0.654 (0.652) | 0.447 | 0.568 | 3.65 | 40.4% |
| 100 | direct | 0.839 (0.663) | 0.577 (0.597) | 0.721 (0.717) | 0.456 | 0.610 | 3.86 | 34.3% |
| 100 | reduce | 0.828 (0.659) | 0.688 (0.692) | 0.821 (0.815) | 0.450 | 0.704 | 4.49 | 27.0% |
| 100 | combined | 0.842 (0.673) | 0.810 (0.819) | 0.967 (0.967) | 0.459 | 0.834 | 5.37 | 15.0% |

Paired differences against the original path (master judge; question-cluster bootstrap; B minus A):

| Load | B | Precision | Recall | Any relevant |
|---|---|---|---|---|
| 20 | direct | +0.006 [−0.002, +0.015] | +0.005 [−0.006, +0.016] | +0.003 [+0.000, +0.007] |
| 20 | reduce | +0.005 [−0.009, +0.023] | −0.009 [−0.036, +0.019] | +0.001 [−0.016, +0.019] |
| 20 | combined | +0.003 [−0.012, +0.020] | −0.008 [−0.036, +0.018] | +0.003 [−0.015, +0.019] |
| 50 | direct | +0.007 [−0.003, +0.018] | +0.021 [−0.007, +0.050] | +0.022 [−0.007, +0.052] |
| 50 | reduce | +0.007 [−0.008, +0.026] | +0.064 [+0.029, +0.101] | +0.076 [+0.041, +0.115] |
| 50 | combined | +0.007 [−0.009, +0.025] | +0.069 [+0.031, +0.110] | +0.081 [+0.045, +0.121] |
| 100 | direct | +0.008 [−0.003, +0.020] | +0.038 [+0.014, +0.062] | +0.067 [+0.039, +0.094] |
| 100 | reduce | +0.019 [−0.007, +0.049] | +0.149 [+0.093, +0.206] | +0.167 [+0.105, +0.230] |
| 100 | combined | +0.020 [−0.004, +0.047] | +0.271 [+0.221, +0.319] | +0.312 [+0.257, +0.369] |

The second judge gives the same signs and similar sizes (for example, load 100 original vs combined:
recall +0.262 [+0.211, +0.315], any relevant +0.315 [+0.257, +0.375]). Precision differences between
reduce and combined are within ±0.005 at every load.

Reading. Among the provisions that are returned, the share judged relevant is the same in all arms, so the
changed law sets are exchanges between provisions of similar judged relevance rather than a drop in
relevance. The recall and "any relevant" gains at loads 50 and 100 come from the reduced arms answering
requests that the original path ends with an empty or fallback response under queueing. These judgments are
model opinions with fair inter-judge agreement (kappa 0.37 on the relevant/not-relevant split) and are not
a substitute for expert labels; they do not assess answer text, only the returned provisions.

## Time series within a run (engine gauges sampled about once per second)

The load client scraped the engine metrics endpoints about every 1.1 s during every cell. The
`timeseries/` folder holds one table per cell in 1-second bins (orchestration and worker engines: KV cache
usage, running and waiting requests, cumulative queue time, completed engine requests, preemptions; client:
cumulative started and completed requests, requests in flight), and `figures/` holds the plots produced
from them by `harness/timeseries_cc.py`. Time is seconds since the first request of each run; the arms ran
one after another on the shared engines, so the comparison aligns runs by run-relative time.

- `figures/timeseries-C100-r4-original-vs-combined.pdf`: original vs combined at concurrency 100, round 4
  (adjacent positions in the rotation). Panels: (a) orchestration KV cache usage, (b) running and waiting
  requests at the orchestration engine, (c) cumulative queue time at the orchestration engine, (d) client
  requests started and completed.
- `figures/timeseries-C100-r4-worker-original-vs-combined.pdf`: the worker engine during the same two runs.
- `figures/timeseries-C100-all-rounds-original-vs-combined.pdf` and `-four-arms.pdf`: all four rounds per
  arm (thin lines) with their per-second mean (thick), and the same for all four arms.
- `figures/timeseries-C50-...` and `timeseries-C20-...`: the two lower loads.

What the traces show at concurrency 100 (ranges over the four rounds; `timeseries/timeseries-summary.json`):

| Indicator (orchestration engine unless stated) | Original | Combined |
|---|---|---|
| Run length (s) | 400–458 | 350–388 |
| KV cache first at ≥99% (s after start) | 50–69 (207 in round 4 after an early stall) | 26–40 |
| Waiting-request peak (requests, at s) | 103–144 (at 143–197 s) | 67–86 (at 53–184 s) |
| Seconds with a non-empty waiting queue | 317–336 | 196–257 |
| Mean running requests | 18.7–21.4 | 24.8–27.5 |
| Total queue time accumulated in the run (s) | 18,300–26,000 | 6,900–8,500 |
| 100th client completion (s after start) | 225–244 | 176–192 |

- The original path shows two waves. Right after the first batch of 100 requests is classified and prepared,
  all of them ask the worker engine to form the tool call; that engine admits four requests at a time, its
  waiting queue climbs to 105–142, and for about 20 s (t ≈ 25–45 s) the orchestration engine has nothing to
  do (KV usage and running requests fall to zero). The same stall repeats at t ≈ 250–290 s for the second
  batch. The combined path never idles the orchestration engine: the worker queue stays at 20–40
  (synthesis calls only) and the search results reach selection immediately.
- Once the search results arrive, the original path's selection calls carry the full 12K-character JSON and
  the orchestration queue builds to 100–145 waiting requests for most of the run; queue time accumulates
  fastest between t ≈ 180 and 250 s. With the reduced input the queue peaks at 67–86 and drains earlier, so
  the cumulative queue time levels off at about a third of the original's.
- At concurrency 50 the original path's waiting queue peaks at 51–54 (combined 29–38) and accumulates
  10,000–11,700 s of queue time (combined 2,600–5,000 s). At concurrency 20 the original path still queues
  up to 10–11 requests for about 150 s of each run (430–470 s of queue time in total), while the combined
  path's orchestration queue is essentially empty and its KV cache stays around 31–37%.

## Limitations and data notes

- Shared engines, one instance per arm, finite 200-request batches; four rounds per load.
- Model judges, no expert labels; the stronger judge is the same model that performs selection.
- Application-side events (boundary timing, selection hashes, per-call statistics) were captured from the
  streamed Pod logs and, after the streams stopped at the first log rotation, from the rotated log files
  kept on the node. One cell (load 50, round 3, original) lost about 2.5 minutes of application-side
  events to rotation and has 93% coverage; client-side records and engine counters are complete for
  every cell.
- The client, questions, and application source hashes are recorded in `review-inventory.json` and
  `protocol.json`.

## Files

- `analysis.json`, `analysis.requests.csv`: all statistics above and per-request response time, law count,
  and selection-input size (question IDs and numbers only).
- `judge-summary.json`: judge call counts, score distributions, repeat and inter-judge agreement.
- `protocol.json`, `status.json`: client protocol and completion record; `review-inventory.json`: package
  and script hashes.
- `timeseries/`: per-cell 1-second tables of engine gauges and client progress, plus `timeseries-summary.json`.
- `figures/`: time-series plots (PDF and PNG) described above.
- `harness/`: server entry point (`serve_cc.py`), manifest generator, remote supervisor, load client,
  metrics collector, analysis (`analyze_cc.py`), the judge (`judge_laws.py`), and the time-series
  extraction and plotting script (`timeseries_cc.py`).
- Raw responses, Pod logs, per-pair judgments, and the question file are not kept in this repository.

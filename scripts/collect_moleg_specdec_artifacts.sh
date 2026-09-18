#!/usr/bin/env bash
# Pull experiment artifacts from the GPU host: public summaries/metrics into experiments/results,
# raw rows with prompts/answers into the git-ignored experiments/private directory.
set -euo pipefail
# SSH53 may be a wrapper that already includes the remote host (then REMOTE must be empty)
SSH=${SSH53:-ssh}; REMOTE=${REMOTE53-}  # e.g. user@gpu-host; empty when SSH53 wraps the host
R=/data/project/vllm/fine-tune/experiments/specdec-20260906
PUB=experiments/results/moleg-specdec-20260906; PRIV=experiments/private/moleg-specdec-20260906
mkdir -p $PUB $PRIV/logs
$SSH $REMOTE "cd $R && tar czf - results/*.summary.json results/*.traces.json results/sim_draft_policies.json results/sim_draft_ds.json results/retrieval_eval.summary.json results/*.manifest.json restore/gpu_before.csv restore/gpu_phase1.csv restore/gpu_after.csv restore/ps_before.txt restore/ps_after.txt run/*.argv.json 2>/dev/null" | tar xzf - -C $PUB --strip-components=0
$SSH $REMOTE "cd $R && tar czf - results/*.jsonl results/*.json ds/*.json logs/*.log 2>/dev/null" | tar xzf - -C $PRIV
# never keep credentials-bearing environment captures outside the server
rm -f $PRIV/restore/*.environ 2>/dev/null || true
echo "public: $(find $PUB -type f | wc -l) files; private: $(find $PRIV -type f | wc -l) files"

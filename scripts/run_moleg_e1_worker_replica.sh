#!/usr/bin/env bash
# E1: worker-replica load balancing under concurrency (200 odd-fold questions).
#   1. production orchestrator restarted with lossless speculative decoding (+ history plugin, larger KV)
#   2. phi-4 stopped; second gpt-oss worker replica launched on GPU1 (:8106)
#   3. gateways: 28150 (single worker) / 28151 (two workers, EWMA-cost routing); apps w1/w2
#   4. e2e at concurrency 4, 8, 16 for both arms; traces + summaries
#   5. restore: apps, gateways, replica stopped; production orchestrator and phi-4 relaunched from saved argv/env
# Run on the GPU host inside the experiment directory. Stops on first error, leaving state for review.
set -euo pipefail
cd /data/project/vllm/fine-tune/experiments/specdec-20260906
P=/data/project/vllm/.venv/bin/python
NUMBA=$PWD/../paper-20260905.OemN3l/numba-compat; PLUGIN=$PWD/vllm_plugin
log(){ echo "[$(date +%T)] $*"; }
GEMMA=$(pgrep -f "vllm serve google/gemma-4-31B-it --port 8000" | head -1)
GPTOSS=$(pgrep -f "vllm serve openai/gpt-oss-20b --port 8006" | head -1)
PHI4=$(pgrep -f "vllm serve microsoft/phi-4 --port 8002" | head -1)
log "recapture environments (gemma $GEMMA, gpt-oss $GPTOSS, phi-4 $PHI4)"
for p in $GEMMA $GPTOSS $PHI4; do tr "\0" "\n" < /proc/$p/environ > restore/$p.environ; tr "\0" "\n" < /proc/$p/cmdline > restore/$p.cmdline; readlink /proc/$p/cwd > restore/$p.cwd; chmod 600 restore/$p.environ; done
nvidia-smi --query-compute-apps=pid,used_memory,gpu_uuid --format=csv > restore/gpu_before_e1.csv
log "restart production orchestrator with speculative decoding (+plugin, --max-num-seqs 64)"
ARGV=$($P - <<PY
import json
argv=[x for x in open('restore/$GEMMA.cmdline').read().split('\n') if x]
argv+=['--speculative-config',json.dumps({"method":"ngram","num_speculative_tokens":16,"prompt_lookup_max":8,"prompt_lookup_min":2}),'--max-num-seqs','64']
print(json.dumps(argv))
PY
)
$P moleg_gpu_ctl.py stop --pid $GEMMA
$P moleg_gpu_ctl.py start-custom --name prod_gemma_spec --argv-json "$ARGV" --gpu 0 --template-pid $GEMMA --pythonpath $PLUGIN:$NUMBA --env MOLEG_SPEC_DATASTORE=$PWD/ds/ds_even.json --env MOLEG_SPEC_STATS=$PWD/results/spec_stats_prod.json --env MOLEG_SPEC_MODE=longest_any
grep -h "KV cache size" logs/custom_prod_gemma_spec.log | cut -c1-120
log "stop phi-4 and launch a second gpt-oss replica on :8106"
$P moleg_gpu_ctl.py stop --pid $PHI4
ARGV2=$($P - <<PY
import json
argv=[x for x in open('restore/$GPTOSS.cmdline').read().split('\n') if x]
argv[argv.index('--port')+1]='8106'
print(json.dumps(argv))
PY
)
$P moleg_gpu_ctl.py start-custom --name gptoss_replica --argv-json "$ARGV2" --gpu 1 --template-pid $GPTOSS
log "gateways and app arms"
setsid nohup $P moleg_lb_gateway.py --port 28150 --policy ewma_cost --upstream http://127.0.0.1:8006 --trace results/e1_gw_single.trace.jsonl > logs/e1_gw_single.log 2>&1 < /dev/null &
setsid nohup $P moleg_lb_gateway.py --port 28151 --policy ewma_cost --upstream http://127.0.0.1:8006 --upstream http://127.0.0.1:8106 --trace results/e1_gw_dual.trace.jsonl > logs/e1_gw_dual.log 2>&1 < /dev/null &
sleep 4; curl -sf http://127.0.0.1:28150/health >/dev/null && curl -sf http://127.0.0.1:28151/health >/dev/null && echo gateways up
/data/project/vllm/fine-tune/2025-moleg-rag/.venv/bin/python launch_moleg_worker_apps.py
sleep 25; for port in 28160 28161; do curl -sf -o /dev/null http://127.0.0.1:$port/docs && echo "app $port up"; done
for c in 4 8 16; do
  log "e2e concurrency $c (w1 single worker vs w2 two workers)"
  $P evaluate_moleg_paper_e2e.py --cases results/cases_odd.json --output results/e1_c${c}.jsonl \
     --variant w1=http://127.0.0.1:28160 --variant w2=http://127.0.0.1:28161 --repeats 1 --concurrency $c --block-size $(( c > 8 ? 16 : 8 )) --limit 400 > logs/e1_c${c}.log 2>&1
  $P summarize_moleg_specdec_e2e.py --rows results/e1_c${c}.jsonl --cases results/cases.json --output results/e1_c${c}.summary.json | tail -4
  $P summarize_moleg_specdec_traces.py --trace w1=results/e1_w1.trace.jsonl --trace w2=results/e1_w2.trace.jsonl --e2e results/e1_c${c}.jsonl --output results/e1_c${c}.traces.json | tail -3
done
log "restore"
for pid in $($P -c "import json;print(' '.join(str(x['pid']) for x in json.load(open('run/worker_apps.json'))))"); do kill -TERM $pid 2>/dev/null || true; done
pkill -f "moleg_lb_gateway.py" || true
$P moleg_gpu_ctl.py stop-named --name gptoss_replica
$P moleg_gpu_ctl.py stop-named --name prod_gemma_spec
$P moleg_gpu_ctl.py restart-saved --pid $GEMMA --name restored_gemma_e1
$P moleg_gpu_ctl.py restart-saved --pid $PHI4 --name restored_phi4_e1
rm -f restore/$GEMMA.environ restore/$GPTOSS.environ restore/$PHI4.environ
$P moleg_gpu_ctl.py status | tail -11
log "E1 COMPLETE AND RESTORED"

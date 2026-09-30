#!/usr/bin/env bash
# Run Phase F only — TP=2 tensor parallelism sweeps (both GPUs on one engine).
# Requires Phase A-E results already on GitHub / in ./results/
#
# Usage:
#   chmod +x run_phase_f.sh
#   MODEL=Qwen/Qwen3-8B ./run_phase_f.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SGLANG_URL="http://localhost:30000"
VLLM_URL="http://localhost:8000"
OUT="${OUT:-./results}"

SGLANG_PID=""
VLLM_PID=""
LOGS_DIR="$OUT/server_logs"
mkdir -p "$LOGS_DIR"

GREEN="\033[0;32m"; YELLOW="\033[1;33m"; CYAN="\033[0;36m"; RED="\033[0;31m"; RESET="\033[0m"
banner() { echo -e "\n${CYAN}════════════════════════════════════════════════════${RESET}\n${CYAN}  $1${RESET}\n${CYAN}════════════════════════════════════════════════════${RESET}\n"; }
ok()     { echo -e "${GREEN}✓ $1${RESET}"; }
info()   { echo -e "${YELLOW}▶ $1${RESET}"; }
err()    { echo -e "${RED}✗ $1${RESET}"; }

start_sglang() {
    local gpu="${1:-0}"; local tp="${2:-1}"; local extra="${3:-}"
    info "Starting SGLang on GPU(s) $gpu (tp=$tp)..."
    CUDA_VISIBLE_DEVICES="$gpu" nohup python -m sglang.launch_server \
        --model-path "$MODEL" \
        --port 30000 \
        ${tp:+--tp-size $tp} \
        $extra \
        > "$LOGS_DIR/sglang.log" 2>&1 &
    SGLANG_PID=$!
    echo "  SGLang PID: $SGLANG_PID  (log: $LOGS_DIR/sglang.log)"
}

start_vllm() {
    local gpu="${1:-0}"; local tp="${2:-1}"; local extra="${3:-}"
    info "Starting vLLM on GPU(s) $gpu (tp=$tp)..."
    CUDA_VISIBLE_DEVICES="$gpu" nohup vllm serve "$MODEL" \
        --port 8000 \
        --override-generation-config '{"enable_thinking": false}' \
        ${tp:+--tensor-parallel-size $tp} \
        $extra \
        > "$LOGS_DIR/vllm.log" 2>&1 &
    VLLM_PID=$!
    echo "  vLLM PID: $VLLM_PID  (log: $LOGS_DIR/vllm.log)"
}

wait_for_server() {
    local url="$1"; local name="$2"; local max=120; local n=0
    info "Waiting for $name at $url ..."
    while ! curl -sf "$url/health" > /dev/null 2>&1 && \
          ! curl -sf "$url/v1/models" > /dev/null 2>&1; do
        sleep 3; n=$((n+1))
        if [ $n -ge $max ]; then
            err "Timed out waiting for $name"
            tail -20 "$LOGS_DIR/sglang.log" 2>/dev/null || true
            tail -20 "$LOGS_DIR/vllm.log"   2>/dev/null || true
            exit 1
        fi
        (( n % 10 == 0 )) && { echo ""; info "Still waiting..."; \
            tail -3 "$LOGS_DIR/vllm.log" 2>/dev/null || \
            tail -3 "$LOGS_DIR/sglang.log" 2>/dev/null || true; } || echo -n "."
    done
    echo ""; ok "$name is ready"
}

kill_sglang() {
    if [ -n "$SGLANG_PID" ]; then
        info "Stopping SGLang (PID $SGLANG_PID)..."
        kill "$SGLANG_PID" 2>/dev/null || true
        wait "$SGLANG_PID" 2>/dev/null || true
        SGLANG_PID=""; sleep 3; ok "SGLang stopped"
    fi
}

kill_vllm() {
    if [ -n "$VLLM_PID" ]; then
        info "Stopping vLLM (PID $VLLM_PID)..."
        kill "$VLLM_PID" 2>/dev/null || true
        wait "$VLLM_PID" 2>/dev/null || true
        VLLM_PID=""; sleep 3; ok "vLLM stopped"
    fi
}

kill_all() { kill_sglang; kill_vllm; }
trap kill_all EXIT INT TERM

START_TIME=$(date +%s)
mkdir -p "$OUT"

echo ""
echo "  inference-optimizer-lab2 — Phase F only (TP=2)"
echo "  Model:   $MODEL"
echo "  Output:  $OUT"
echo "  Started: $(date)"

# =============================================================================
# PHASE F — Tensor Parallelism TP=2
# =============================================================================

banner "PHASE F: Tensor Parallelism TP=2"
echo "  Both GPUs → single engine. Expected: ~1.95x throughput (NVLink)"

# F1-F3: SGLang TP=2
start_sglang "0,1" "2"
wait_for_server "$SGLANG_URL" "SGLang TP=2"

banner "F1: SGLang TP=2 — Quick sweep"
python -m inference_optimizer sweep \
    --quick --engine sglang --model "$MODEL" \
    --sglang-url "$SGLANG_URL" --output-dir "$OUT/sglang_tp2_quick"
ok "F1 complete → $OUT/sglang_tp2_quick/charts/"

banner "F2: SGLang TP=2 — Peak sweep"
python -m inference_optimizer sweep \
    --peak --engine sglang --model "$MODEL" \
    --sglang-url "$SGLANG_URL" --output-dir "$OUT/sglang_tp2_peak"
ok "F2 complete → $OUT/sglang_tp2_peak/charts/"

banner "F3: SGLang TP=2 — Full default sweep"
python -m inference_optimizer sweep \
    --engine sglang --model "$MODEL" \
    --sglang-url "$SGLANG_URL" --output-dir "$OUT/sglang_tp2_default"
ok "F3 complete → $OUT/sglang_tp2_default/charts/"

kill_sglang

# F4-F6: vLLM TP=2
start_vllm "0,1" "2"
wait_for_server "$VLLM_URL" "vLLM TP=2"

banner "F4: vLLM TP=2 — Quick sweep"
python -m inference_optimizer sweep \
    --quick --engine vllm --model "$MODEL" \
    --vllm-url "$VLLM_URL" --output-dir "$OUT/vllm_tp2_quick"
ok "F4 complete → $OUT/vllm_tp2_quick/charts/"

banner "F5: vLLM TP=2 — Peak sweep"
python -m inference_optimizer sweep \
    --peak --engine vllm --model "$MODEL" \
    --vllm-url "$VLLM_URL" --output-dir "$OUT/vllm_tp2_peak"
ok "F5 complete → $OUT/vllm_tp2_peak/charts/"

banner "F6: vLLM TP=2 — Full default sweep"
python -m inference_optimizer sweep \
    --engine vllm --model "$MODEL" \
    --vllm-url "$VLLM_URL" --output-dir "$OUT/vllm_tp2_default"
ok "F6 complete → $OUT/vllm_tp2_default/charts/"

kill_vllm

# =============================================================================
# SUMMARY
# =============================================================================

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

banner "Phase F Complete"
echo "  Total time: ${ELAPSED} minutes"
echo "  Results:"
for dir in "$OUT"/sglang_tp2_* "$OUT"/vllm_tp2_*/; do
    [ -d "$dir" ] || continue
    n_charts=$(ls "$dir/charts/"*.png 2>/dev/null | wc -l || echo 0)
    n_json=$(ls "$dir"/*.json 2>/dev/null | wc -l || echo 0)
    echo "  $(basename "$dir"):  ${n_json} result(s), ${n_charts} chart(s)"
done
echo ""
echo "  Zip results:  python3 -c \"import zipfile,os; z=zipfile.ZipFile('results_phase_f.zip','w',zipfile.ZIP_DEFLATED); [z.write(os.path.join(r,f)) for r,d,fs in os.walk('results') for f in fs]; z.close(); print('Done')\""

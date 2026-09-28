#!/usr/bin/env bash
# Run phases B-F only (skip Phase A SGLang sweeps).
# Use when Phase A results already exist in $OUT/sglang_*/
#
# Usage:
#   chmod +x run_from_b.sh
#   MODEL=Qwen/Qwen3-8B TP2=1 ./run_from_b.sh

set -euo pipefail

MODEL="${MODEL:-Qwen/Qwen3-8B}"
SGLANG_URL="http://localhost:30000"
VLLM_URL="http://localhost:8000"
OUT="${OUT:-./results}"
SKIP_LONG="${SKIP_LONG:-0}"
TP2="${TP2:-0}"

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
    local gpu="${1:-1}"; local tp="${2:-1}"; local extra="${3:-}"
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
        (( n % 10 == 0 )) && { echo ""; info "Still waiting... last log lines:"; \
            tail -3 "$LOGS_DIR/sglang.log" 2>/dev/null || \
            tail -3 "$LOGS_DIR/vllm.log"   2>/dev/null || true; } || echo -n "."
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
echo "  inference-optimizer-lab2 — Phases B-F"
echo "  Model:   $MODEL"
echo "  Output:  $OUT"
echo "  Started: $(date)"

# =============================================================================
# PHASE B — vLLM (GPU 1)
# =============================================================================

banner "PHASE B: vLLM Sweeps (GPU 1)"

start_vllm "1" "1"
wait_for_server "$VLLM_URL" "vLLM"

banner "B1: Quick Sweep — vLLM validation (12 cells)"
python -m inference_optimizer sweep \
    --quick --engine vllm --model "$MODEL" \
    --vllm-url "$VLLM_URL" --output-dir "$OUT/vllm_quick"
ok "B1 complete → $OUT/vllm_quick/charts/"

banner "B2: Peak Sweep — vLLM saturation cliff (28 cells)"
python -m inference_optimizer sweep \
    --peak --engine vllm --model "$MODEL" \
    --vllm-url "$VLLM_URL" --output-dir "$OUT/vllm_peak"
ok "B2 complete → $OUT/vllm_peak/charts/"

banner "B3: Default Sweep — vLLM full grid (112 cells)"
python -m inference_optimizer sweep \
    --engine vllm --model "$MODEL" \
    --vllm-url "$VLLM_URL" --output-dir "$OUT/vllm_default"
ok "B3 complete → $OUT/vllm_default/charts/"

kill_vllm

# =============================================================================
# PHASE C — Head-to-head (SGLang GPU 0, vLLM GPU 1)
# =============================================================================

banner "PHASE C: Head-to-Head — SGLang vs vLLM"

start_sglang "0" "1"
start_vllm   "1" "1"
wait_for_server "$SGLANG_URL" "SGLang"
wait_for_server "$VLLM_URL"   "vLLM"

banner "C1: Quick Head-to-Head (12 cells × 2 engines)"
python -m inference_optimizer sweep \
    --quick --engine sglang vllm --model "$MODEL" \
    --sglang-url "$SGLANG_URL" --vllm-url "$VLLM_URL" \
    --output-dir "$OUT/headtohead_quick"
ok "C1 complete → $OUT/headtohead_quick/charts/"

banner "C2: Full Head-to-Head (112 cells × 2 engines)"
python -m inference_optimizer sweep \
    --engine sglang vllm --model "$MODEL" \
    --sglang-url "$SGLANG_URL" --vllm-url "$VLLM_URL" \
    --output-dir "$OUT/headtohead_full"
ok "C2 complete → $OUT/headtohead_full/charts/"

kill_all

# =============================================================================
# PHASE D — Long-context (SGLang, GPU 0, FP8 KV)
# =============================================================================

if [ "$SKIP_LONG" = "1" ]; then
    info "Skipping Phase D (SKIP_LONG=1)"
else
    banner "PHASE D: Long-Context Sweep (64K / 128K)"
    start_sglang "0" "1" "--context-length 131072 --kv-cache-dtype fp8_e5m2"
    wait_for_server "$SGLANG_URL" "SGLang (long-context)"

    banner "D1: Long-Context Sweep — 64K / 128K (8 cells)"
    python -m inference_optimizer sweep \
        --long-context --engine sglang --model "$MODEL" \
        --sglang-url "$SGLANG_URL" --output-dir "$OUT/sglang_longctx"
    ok "D1 complete → $OUT/sglang_longctx/charts/"

    kill_sglang
fi

# =============================================================================
# PHASE E — Compare SGLang default vs vLLM default
# =============================================================================

banner "PHASE E: Compare SGLang vs vLLM"

SGLANG_JSON=$(ls -t "$OUT/sglang_default"/*.json 2>/dev/null | head -1 || echo "")
VLLM_JSON=$(ls -t "$OUT/vllm_default"/*.json 2>/dev/null | head -1 || echo "")

if [ -n "$SGLANG_JSON" ] && [ -n "$VLLM_JSON" ]; then
    python -m inference_optimizer compare \
        --baseline "$SGLANG_JSON" --optimized "$VLLM_JSON"
    ok "E1 complete"
else
    info "Skipping compare — result files not found"
fi

# =============================================================================
# PHASE F — Tensor Parallelism TP=2
# =============================================================================

if [ "$TP2" = "1" ]; then
    banner "PHASE F: Tensor Parallelism TP=2"

    start_sglang "0,1" "2"
    wait_for_server "$SGLANG_URL" "SGLang TP=2"

    banner "F1: SGLang TP=2 — Quick sweep"
    python -m inference_optimizer sweep \
        --quick --engine sglang --model "$MODEL" \
        --sglang-url "$SGLANG_URL" --output-dir "$OUT/sglang_tp2_quick"
    ok "F1 complete"

    banner "F2: SGLang TP=2 — Peak sweep"
    python -m inference_optimizer sweep \
        --peak --engine sglang --model "$MODEL" \
        --sglang-url "$SGLANG_URL" --output-dir "$OUT/sglang_tp2_peak"
    ok "F2 complete"

    banner "F3: SGLang TP=2 — Full default sweep"
    python -m inference_optimizer sweep \
        --engine sglang --model "$MODEL" \
        --sglang-url "$SGLANG_URL" --output-dir "$OUT/sglang_tp2_default"
    ok "F3 complete"

    kill_sglang

    start_vllm "0,1" "2"
    wait_for_server "$VLLM_URL" "vLLM TP=2"

    banner "F4: vLLM TP=2 — Quick sweep"
    python -m inference_optimizer sweep \
        --quick --engine vllm --model "$MODEL" \
        --vllm-url "$VLLM_URL" --output-dir "$OUT/vllm_tp2_quick"
    ok "F4 complete"

    banner "F5: vLLM TP=2 — Peak sweep"
    python -m inference_optimizer sweep \
        --peak --engine vllm --model "$MODEL" \
        --vllm-url "$VLLM_URL" --output-dir "$OUT/vllm_tp2_peak"
    ok "F5 complete"

    banner "F6: vLLM TP=2 — Full default sweep"
    python -m inference_optimizer sweep \
        --engine vllm --model "$MODEL" \
        --vllm-url "$VLLM_URL" --output-dir "$OUT/vllm_tp2_default"
    ok "F6 complete"

    kill_vllm

    banner "F7: Compare SGLang 1-GPU vs TP=2"
    S1=$(ls -t "$OUT/sglang_default"/*.json 2>/dev/null | head -1 || echo "")
    S2=$(ls -t "$OUT/sglang_tp2_default"/*.json 2>/dev/null | head -1 || echo "")
    [ -n "$S1" ] && [ -n "$S2" ] && python -m inference_optimizer compare \
        --baseline "$S1" --optimized "$S2" && ok "F7 complete"

    banner "F8: Compare vLLM 1-GPU vs TP=2"
    V1=$(ls -t "$OUT/vllm_default"/*.json 2>/dev/null | head -1 || echo "")
    V2=$(ls -t "$OUT/vllm_tp2_default"/*.json 2>/dev/null | head -1 || echo "")
    [ -n "$V1" ] && [ -n "$V2" ] && python -m inference_optimizer compare \
        --baseline "$V1" --optimized "$V2" && ok "F8 complete"

else
    info "Skipping Phase F. To run TP=2: TP2=1 MODEL=$MODEL ./run_from_b.sh"
fi

# =============================================================================
# SUMMARY
# =============================================================================

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

banner "Phases B-F Complete"
echo "  Total time: ${ELAPSED} minutes"
echo "  Results:"
for dir in "$OUT"/*/; do
    [ -d "$dir" ] || continue
    n_charts=$(ls "$dir/charts/"*.png 2>/dev/null | wc -l || echo 0)
    n_json=$(ls "$dir"/*.json 2>/dev/null | wc -l || echo 0)
    echo "  $(basename "$dir"):  ${n_json} result(s), ${n_charts} chart(s)"
done
echo ""
echo "  Copy charts: scp -r root@<ip>:$(pwd)/$OUT/*/charts/ ./local-results/"

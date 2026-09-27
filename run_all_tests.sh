#!/usr/bin/env bash
# =============================================================================
# inference-optimizer-lab2 — Full Test Suite
# Runs every preset, both engines, optimize loop, long-context, and comparison.
#
# Usage:
#   chmod +x run_all_tests.sh
#   MODEL=/workspace/Qwen3-8B ./run_all_tests.sh
#
# Defaults (override with env vars):
#   MODEL      — model path  (default: /workspace/Qwen3-8B)
#   SGLANG_URL — SGLang URL  (default: http://localhost:30000)
#   VLLM_URL   — vLLM URL    (default: http://localhost:8000)
#   OUT        — output dir  (default: ./results)
#   SKIP_LONG  — set to 1 to skip 64K/128K long-context tests
#   TP2        — set to 1 to run Phase F (tensor parallelism TP=2 on both GPUs)
#
# Phase F usage:
#   TP2=1 MODEL=/workspace/Qwen3-8B ./run_all_tests.sh
#   Run after Phases A-E complete. Kills both engines and runs TP=2 on each.
# =============================================================================

set -euo pipefail

MODEL="${MODEL:-/workspace/Qwen3-8B}"
SGLANG_URL="${SGLANG_URL:-http://localhost:30000}"
VLLM_URL="${VLLM_URL:-http://localhost:8000}"
OUT="${OUT:-./results}"
SKIP_LONG="${SKIP_LONG:-0}"
TP2="${TP2:-0}"

GREEN="\033[0;32m"
YELLOW="\033[1;33m"
CYAN="\033[0;36m"
RED="\033[0;31m"
RESET="\033[0m"

banner() { echo -e "\n${CYAN}════════════════════════════════════════════════════${RESET}"; echo -e "${CYAN}  $1${RESET}"; echo -e "${CYAN}════════════════════════════════════════════════════${RESET}\n"; }
ok()     { echo -e "${GREEN}✓ $1${RESET}"; }
info()   { echo -e "${YELLOW}▶ $1${RESET}"; }
pause()  {
    echo -e "\n${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    echo -e "${YELLOW}  ACTION REQUIRED:${RESET}"
    echo -e "  $1"
    echo -e "${YELLOW}  Press ENTER when ready...${RESET}"
    echo -e "${YELLOW}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}"
    read -r
}

wait_for_server() {
    local url="$1"
    local name="$2"
    local max=60
    local n=0
    info "Waiting for $name to be ready at $url ..."
    while ! curl -sf "$url/health" > /dev/null 2>&1 && \
          ! curl -sf "$url/v1/models" > /dev/null 2>&1; do
        sleep 3
        n=$((n+1))
        if [ $n -ge $max ]; then
            echo -e "${RED}Timed out waiting for $name${RESET}"; exit 1
        fi
        echo -n "."
    done
    echo ""
    ok "$name is ready"
}

START_TIME=$(date +%s)
mkdir -p "$OUT"

echo ""
echo "  inference-optimizer-lab2 — Full Test Suite"
echo "  Model:   $MODEL"
echo "  Output:  $OUT"
echo "  Started: $(date)"

# =============================================================================
# PHASE A — SGLang
# =============================================================================

banner "PHASE A: SGLang Sweeps"

pause "Start SGLang in another terminal:

  python -m sglang.launch_server \\
    --model-path $MODEL \\
    --port 30000 \\
    --chat-template qwen3

Wait until you see 'Server is ready', then press ENTER."

wait_for_server "$SGLANG_URL" "SGLang"

# A1 — Quick (validation)
banner "A1: Quick Sweep — validation (12 cells)"
python -m inference_optimizer sweep \
    --quick \
    --engine sglang \
    --model "$MODEL" \
    --sglang-url "$SGLANG_URL" \
    --output-dir "$OUT/sglang_quick"
ok "A1 complete → $OUT/sglang_quick/charts/"

# A2 — Peak (saturation cliff)
banner "A2: Peak Sweep — saturation cliff (28 cells, ~30 min)"
python -m inference_optimizer sweep \
    --peak \
    --engine sglang \
    --model "$MODEL" \
    --sglang-url "$SGLANG_URL" \
    --output-dir "$OUT/sglang_peak"
ok "A2 complete → $OUT/sglang_peak/charts/"

# A3 — Default (full grid)
banner "A3: Default Sweep — full grid (112 cells, ~2 hrs)"
python -m inference_optimizer sweep \
    --engine sglang \
    --model "$MODEL" \
    --sglang-url "$SGLANG_URL" \
    --output-dir "$OUT/sglang_default"
ok "A3 complete → $OUT/sglang_default/charts/"

# A4 — Optimize loop
banner "A4: Optimization Loop — Measure → Diagnose → Optimize → Re-benchmark"
python -m inference_optimizer optimize \
    --engine sglang \
    --model "$MODEL" \
    --sglang-url "$SGLANG_URL" \
    --output-dir "$OUT/sglang_optimized" \
    --max-iterations 5
ok "A4 complete → $OUT/sglang_optimized/charts/"

# =============================================================================
# PHASE B — vLLM
# =============================================================================

banner "PHASE B: vLLM Sweeps"

pause "Stop SGLang (Ctrl+C in its terminal), then start vLLM:

  vllm serve $MODEL \\
    --port 8000 \\
    --override-generation-config '{\"enable_thinking\": false}'

Wait until you see 'Application startup complete', then press ENTER."

wait_for_server "$VLLM_URL" "vLLM"

# B1 — Quick
banner "B1: Quick Sweep — vLLM validation (12 cells)"
python -m inference_optimizer sweep \
    --quick \
    --engine vllm \
    --model "$MODEL" \
    --vllm-url "$VLLM_URL" \
    --output-dir "$OUT/vllm_quick"
ok "B1 complete → $OUT/vllm_quick/charts/"

# B2 — Peak
banner "B2: Peak Sweep — vLLM saturation cliff (28 cells)"
python -m inference_optimizer sweep \
    --peak \
    --engine vllm \
    --model "$MODEL" \
    --vllm-url "$VLLM_URL" \
    --output-dir "$OUT/vllm_peak"
ok "B2 complete → $OUT/vllm_peak/charts/"

# B3 — Default
banner "B3: Default Sweep — vLLM full grid (112 cells)"
python -m inference_optimizer sweep \
    --engine vllm \
    --model "$MODEL" \
    --vllm-url "$VLLM_URL" \
    --output-dir "$OUT/vllm_default"
ok "B3 complete → $OUT/vllm_default/charts/"

# =============================================================================
# PHASE C — Head-to-head (both engines running)
# =============================================================================

banner "PHASE C: Head-to-Head — SGLang vs vLLM"

pause "Now start BOTH servers simultaneously (two terminals):

  Terminal 1 — SGLang:
    python -m sglang.launch_server \\
      --model-path $MODEL \\
      --port 30000 \\
      --chat-template qwen3

  Terminal 2 — vLLM:
    vllm serve $MODEL \\
      --port 8000 \\
      --override-generation-config '{\"enable_thinking\": false}'

  NOTE: Qwen3-8B (~16 GB) × 2 = ~32 GB — fits comfortably on H100 NVL (93.6 GB).
  Wait for both to be ready, then press ENTER."

wait_for_server "$SGLANG_URL" "SGLang"
wait_for_server "$VLLM_URL" "vLLM"

# C1 — Quick head-to-head
banner "C1: Quick Head-to-Head (12 cells × 2 engines)"
python -m inference_optimizer sweep \
    --quick \
    --engine sglang vllm \
    --model "$MODEL" \
    --sglang-url "$SGLANG_URL" \
    --vllm-url "$VLLM_URL" \
    --output-dir "$OUT/headtohead_quick"
ok "C1 complete → $OUT/headtohead_quick/charts/"

# C2 — Full head-to-head
banner "C2: Full Head-to-Head (112 cells × 2 engines, ~3 hrs)"
python -m inference_optimizer sweep \
    --engine sglang vllm \
    --model "$MODEL" \
    --sglang-url "$SGLANG_URL" \
    --vllm-url "$VLLM_URL" \
    --output-dir "$OUT/headtohead_full"
ok "C2 complete → $OUT/headtohead_full/charts/"

# =============================================================================
# PHASE D — Long-context (64K / 128K)
# =============================================================================

if [ "$SKIP_LONG" = "1" ]; then
    info "Skipping Phase D (SKIP_LONG=1)"
else
    banner "PHASE D: Long-Context Sweep (64K / 128K)"

    pause "Stop both servers. Start SGLang with FP8 KV cache and 128K context window:

  python -m sglang.launch_server \\
    --model-path $MODEL \\
    --port 30000 \\
    --chat-template qwen3 \\
    --context-length 131072 \\
    --kv-cache-dtype fp8_e5m2

  ⚠  This uses ~57 GB of KV cache at ctx=128K c=1.
     All 93.6 GB VRAM is in use — single request only.
  Press ENTER when ready."

    wait_for_server "$SGLANG_URL" "SGLang (long-context)"

    banner "D1: Long-Context Sweep — 64K / 128K (8 cells)"
    python -m inference_optimizer sweep \
        --long-context \
        --engine sglang \
        --model "$MODEL" \
        --sglang-url "$SGLANG_URL" \
        --output-dir "$OUT/sglang_longctx"
    ok "D1 complete → $OUT/sglang_longctx/charts/"
fi

# =============================================================================
# PHASE E — Compare saved results
# =============================================================================

banner "PHASE E: Compare SGLang Default vs vLLM Default"

# Find the latest JSON files from the two default sweeps
SGLANG_JSON=$(ls -t "$OUT/sglang_default"/*.json 2>/dev/null | head -1 || echo "")
VLLM_JSON=$(ls -t "$OUT/vllm_default"/*.json 2>/dev/null | head -1 || echo "")

if [ -n "$SGLANG_JSON" ] && [ -n "$VLLM_JSON" ]; then
    python -m inference_optimizer compare \
        --baseline "$SGLANG_JSON" \
        --optimized "$VLLM_JSON"
    ok "E1 complete"
else
    info "Skipping compare — one or both default sweep result files not found"
fi

# =============================================================================
# PHASE F — Tensor Parallelism TP=2 (both GPUs on one engine)
# =============================================================================

if [ "$TP2" = "1" ]; then
    banner "PHASE F: Tensor Parallelism TP=2"
    echo "  Both GPUs are now used by a single engine."
    echo "  Expected: ~1.95x throughput vs single GPU (NVLink 4.0 on H100 SXM5/NVL)"
    echo ""

    # F1 — SGLang TP=2
    pause "Stop all running servers. Start SGLang with TP=2 across both GPUs:

  CUDA_VISIBLE_DEVICES=0,1 python -m sglang.launch_server \\
    --model-path $MODEL \\
    --port 30000 \\
    --chat-template qwen3 \\
    --tp-size 2

  Wait until you see 'Server is ready', then press ENTER."

    wait_for_server "$SGLANG_URL" "SGLang TP=2"

    banner "F1: SGLang TP=2 — Quick sweep (validation)"
    python -m inference_optimizer sweep \
        --quick \
        --engine sglang \
        --model "$MODEL" \
        --sglang-url "$SGLANG_URL" \
        --output-dir "$OUT/sglang_tp2_quick"
    ok "F1 complete → $OUT/sglang_tp2_quick/charts/"

    banner "F2: SGLang TP=2 — Peak sweep (saturation cliff)"
    python -m inference_optimizer sweep \
        --peak \
        --engine sglang \
        --model "$MODEL" \
        --sglang-url "$SGLANG_URL" \
        --output-dir "$OUT/sglang_tp2_peak"
    ok "F2 complete → $OUT/sglang_tp2_peak/charts/"

    banner "F3: SGLang TP=2 — Full default sweep"
    python -m inference_optimizer sweep \
        --engine sglang \
        --model "$MODEL" \
        --sglang-url "$SGLANG_URL" \
        --output-dir "$OUT/sglang_tp2_default"
    ok "F3 complete → $OUT/sglang_tp2_default/charts/"

    # F4 — vLLM TP=2
    pause "Stop SGLang. Now start vLLM with TP=2 across both GPUs:

  CUDA_VISIBLE_DEVICES=0,1 vllm serve $MODEL \\
    --port 8000 \\
    --override-generation-config '{\"enable_thinking\": false}' \\
    --tensor-parallel-size 2

  Wait until you see 'Application startup complete', then press ENTER."

    wait_for_server "$VLLM_URL" "vLLM TP=2"

    banner "F4: vLLM TP=2 — Quick sweep (validation)"
    python -m inference_optimizer sweep \
        --quick \
        --engine vllm \
        --model "$MODEL" \
        --vllm-url "$VLLM_URL" \
        --output-dir "$OUT/vllm_tp2_quick"
    ok "F4 complete → $OUT/vllm_tp2_quick/charts/"

    banner "F5: vLLM TP=2 — Peak sweep (saturation cliff)"
    python -m inference_optimizer sweep \
        --peak \
        --engine vllm \
        --model "$MODEL" \
        --vllm-url "$VLLM_URL" \
        --output-dir "$OUT/vllm_tp2_peak"
    ok "F5 complete → $OUT/vllm_tp2_peak/charts/"

    banner "F6: vLLM TP=2 — Full default sweep"
    python -m inference_optimizer sweep \
        --engine vllm \
        --model "$MODEL" \
        --vllm-url "$VLLM_URL" \
        --output-dir "$OUT/vllm_tp2_default"
    ok "F6 complete → $OUT/vllm_tp2_default/charts/"

    # F7 — Compare 1GPU vs TP=2 for each engine
    banner "F7: Compare 1-GPU vs TP=2 — SGLang"
    SGLANG_1GPU_JSON=$(ls -t "$OUT/sglang_default"/*.json 2>/dev/null | head -1 || echo "")
    SGLANG_TP2_JSON=$(ls -t "$OUT/sglang_tp2_default"/*.json 2>/dev/null | head -1 || echo "")
    if [ -n "$SGLANG_1GPU_JSON" ] && [ -n "$SGLANG_TP2_JSON" ]; then
        python -m inference_optimizer compare \
            --baseline "$SGLANG_1GPU_JSON" \
            --optimized "$SGLANG_TP2_JSON"
        ok "F7 SGLang 1GPU vs TP=2 complete"
    fi

    banner "F8: Compare 1-GPU vs TP=2 — vLLM"
    VLLM_1GPU_JSON=$(ls -t "$OUT/vllm_default"/*.json 2>/dev/null | head -1 || echo "")
    VLLM_TP2_JSON=$(ls -t "$OUT/vllm_tp2_default"/*.json 2>/dev/null | head -1 || echo "")
    if [ -n "$VLLM_1GPU_JSON" ] && [ -n "$VLLM_TP2_JSON" ]; then
        python -m inference_optimizer compare \
            --baseline "$VLLM_1GPU_JSON" \
            --optimized "$VLLM_TP2_JSON"
        ok "F8 vLLM 1GPU vs TP=2 complete"
    fi

else
    info "Skipping Phase F (TP2 not set). To run: TP2=1 MODEL=$MODEL ./run_all_tests.sh"
fi

# =============================================================================
# SUMMARY
# =============================================================================

END_TIME=$(date +%s)
ELAPSED=$(( (END_TIME - START_TIME) / 60 ))

banner "All Tests Complete"
echo "  Total time: ${ELAPSED} minutes"
echo "  Results:"
echo ""
for dir in "$OUT"/*/; do
    n_charts=$(ls "$dir/charts/"*.png 2>/dev/null | wc -l || echo 0)
    n_json=$(ls "$dir"/*.json 2>/dev/null | wc -l || echo 0)
    echo "  $(basename "$dir"):  ${n_json} result file(s), ${n_charts} chart(s)"
done
echo ""
echo "  All charts are in their respective results/*/charts/ directories."
echo "  Copy them off the GPU machine with:"
echo "    scp -r user@gpu-machine:$(pwd)/$OUT/*/charts/ ./local-results/"

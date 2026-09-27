"""
GPU memory requirements for LLM inference sweeps.

Computes KV cache memory costs and emits per-cell warnings so you know
before running which cells need more than one GPU.

Math basis — Qwen3-8B:
  num_layers = 28
  num_kv_heads = 8   (GQA)
  head_dim = 128
  BF16 = 2 bytes/element, FP8 = 1 byte/element

Per-token KV cache:
  BF16: 28 × 2 (K+V) × 8 × 128 × 2 bytes = 917,504 bytes ≈ 0.875 MB/token
  FP8:  28 × 2 (K+V) × 8 × 128 × 1 byte  = 458,752 bytes ≈ 0.4375 MB/token

Single H100 NVL (93.6 GB total):
  ~16 GB  — Qwen3-8B BF16 weights
  ~6 GB   — CUDA context, CUDA graphs, OS
  ~71 GB  — available for KV cache (generous estimate)

H100 NVL pool capacity:
  BF16: 71 GB / 0.875 MB ≈ 81,143 tokens total
  FP8:  71 GB / 0.4375 MB ≈ 162,286 tokens total

This is the total active-token budget across ALL concurrent sequences.
Individual requests > pool capacity will fail with a context-too-long error
UNLESS the server was started with --max-model-len set high enough AND
the KV pool was sized to match.
"""

from __future__ import annotations

from dataclasses import dataclass

# --------------------------------------------------------------------------- #
# Model-specific KV constants (Qwen3-8B)
# --------------------------------------------------------------------------- #

_NUM_LAYERS = 28
_NUM_KV_HEADS = 8
_HEAD_DIM = 128
_BYTES_BF16 = 2
_BYTES_FP8 = 1

KV_BYTES_PER_TOKEN_BF16: int = _NUM_LAYERS * 2 * _NUM_KV_HEADS * _HEAD_DIM * _BYTES_BF16
KV_BYTES_PER_TOKEN_FP8: int = _NUM_LAYERS * 2 * _NUM_KV_HEADS * _HEAD_DIM * _BYTES_FP8

# Single H100 NVL budget (bytes)
_H100_NVL_TOTAL_GB = 93.6
_WEIGHTS_GB = 16.0          # Qwen3-8B BF16
_SYSTEM_OVERHEAD_GB = 6.0   # CUDA ctx, graphs, OS
_KV_BUDGET_GB = _H100_NVL_TOTAL_GB - _WEIGHTS_GB - _SYSTEM_OVERHEAD_GB
_KV_BUDGET_BYTES = _KV_BUDGET_GB * (1024 ** 3)

# Token capacity of the KV pool on one H100 NVL
SINGLE_GPU_KV_TOKENS_BF16 = int(_KV_BUDGET_BYTES / KV_BYTES_PER_TOKEN_BF16)
SINGLE_GPU_KV_TOKENS_FP8 = int(_KV_BUDGET_BYTES / KV_BYTES_PER_TOKEN_FP8)


# --------------------------------------------------------------------------- #
# Requirement dataclass
# --------------------------------------------------------------------------- #

@dataclass
class GPURequirement:
    context_tokens: int
    concurrency: int
    kv_gb_bf16: float      # KV memory needed (worst case: all c requests fill ctx)
    kv_gb_fp8: float
    min_gpus_bf16: int     # GPUs needed to hold this in BF16
    min_gpus_fp8: int      # GPUs needed to hold this in FP8
    safe_on_single_gpu: bool
    warning: str           # human-readable note
    server_flags_required: list[str]  # flags needed to serve this at all


def check_cell(context_tokens: int, concurrency: int) -> GPURequirement:
    """
    Estimate GPU requirements for one sweep cell.

    'concurrency' is the client-side concurrency bound, not the scheduler's
    batch size. We use it as an upper bound on simultaneous active sequences.
    """
    # Worst case: all concurrent requests have full context in-flight at once
    max_active_tokens = context_tokens * concurrency

    kv_bytes_bf16 = max_active_tokens * KV_BYTES_PER_TOKEN_BF16
    kv_bytes_fp8 = max_active_tokens * KV_BYTES_PER_TOKEN_FP8

    kv_gb_bf16 = kv_bytes_bf16 / (1024 ** 3)
    kv_gb_fp8 = kv_bytes_fp8 / (1024 ** 3)

    budget = _KV_BUDGET_BYTES
    min_gpus_bf16 = max(1, int(kv_bytes_bf16 / budget) + (1 if kv_bytes_bf16 % budget else 0))
    min_gpus_fp8 = max(1, int(kv_bytes_fp8 / budget) + (1 if kv_bytes_fp8 % budget else 0))

    safe = min_gpus_bf16 == 1
    flags: list[str] = []

    if context_tokens > 8192:
        flags.append(f"--max-model-len {context_tokens}")
    if context_tokens > 32768:
        flags.append("--kv-cache-dtype fp8  (recommended)")
    if context_tokens > 65536:
        flags.append("--tensor-parallel-size 2  (required for BF16)")

    # Build human-readable warning
    if safe:
        if context_tokens > 16384:
            warning = (
                f"ctx={context_tokens:,} c={concurrency}: "
                f"KV={kv_gb_bf16:.1f} GB BF16 / {kv_gb_fp8:.1f} GB FP8 — "
                f"fits on 1 GPU but is large; add --max-model-len {context_tokens}"
            )
        else:
            warning = ""
    else:
        if min_gpus_fp8 == 1:
            warning = (
                f"⚠  ctx={context_tokens:,} c={concurrency}: "
                f"KV={kv_gb_bf16:.1f} GB in BF16 → needs {min_gpus_bf16} GPUs in BF16 "
                f"but fits on 1 GPU with FP8 KV ({kv_gb_fp8:.1f} GB). "
                f"Add --kv-cache-dtype fp8"
            )
        else:
            warning = (
                f"⚠⚠ ctx={context_tokens:,} c={concurrency}: "
                f"KV={kv_gb_bf16:.1f} GB BF16 / {kv_gb_fp8:.1f} GB FP8 — "
                f"needs {min_gpus_bf16} GPUs (BF16) or {min_gpus_fp8} GPUs (FP8). "
                f"Add --tensor-parallel-size {min_gpus_fp8}"
            )

    return GPURequirement(
        context_tokens=context_tokens,
        concurrency=concurrency,
        kv_gb_bf16=round(kv_gb_bf16, 2),
        kv_gb_fp8=round(kv_gb_fp8, 2),
        min_gpus_bf16=min_gpus_bf16,
        min_gpus_fp8=min_gpus_fp8,
        safe_on_single_gpu=safe,
        warning=warning,
        server_flags_required=flags,
    )


def print_requirements_table(context_lengths: list[int], concurrencies: list[int]) -> None:
    """
    Print a GPU requirements table for the planned sweep so the user knows
    which cells need more hardware before starting.
    """
    print("\n  GPU Requirements (Qwen3-8B, single H100 NVL 93.6 GB)")
    print(f"  KV budget: {_KV_BUDGET_GB:.0f} GB  |  "
          f"BF16 capacity: {SINGLE_GPU_KV_TOKENS_BF16:,} tokens  |  "
          f"FP8 capacity: {SINGLE_GPU_KV_TOKENS_FP8:,} tokens")
    print()

    # Table: rows = context, cols = concurrency
    col_w = 22
    header = f"  {'Context':>8}"
    for c in concurrencies:
        header += f"  {'C='+str(c):>{col_w}}"
    print(header)
    print("  " + "-" * (10 + len(concurrencies) * (col_w + 2)))

    needs_multi: list[GPURequirement] = []

    for ctx in context_lengths:
        row = f"  {ctx:>8}"
        for c in concurrencies:
            req = check_cell(ctx, c)
            if req.safe_on_single_gpu:
                cell_str = f"{req.kv_gb_bf16:.1f}GB ok"
            elif req.min_gpus_fp8 == 1:
                cell_str = f"fp8→1GPU"
                needs_multi.append(req)
            else:
                cell_str = f"{req.min_gpus_fp8}GPU(fp8)"
                needs_multi.append(req)
            row += f"  {cell_str:>{col_w}}"
        print(row)

    print()
    print("  Legend: '<N> GB ok' = fits on 1 GPU in BF16  |  "
          "'fp8→1GPU' = needs FP8 KV flag  |  "
          "'NGP(fp8)' = needs N GPUs even in FP8")

    if needs_multi:
        print()
        print("  Cells requiring action:")
        seen: set[tuple] = set()
        for req in needs_multi:
            key = (req.context_tokens, req.concurrency)
            if key in seen:
                continue
            seen.add(key)
            if req.warning:
                print(f"    {req.warning}")
            if req.server_flags_required:
                print(f"      Required server flags: {' '.join(req.server_flags_required)}")
    else:
        print()
        print("  All cells fit on a single H100 NVL in BF16.")

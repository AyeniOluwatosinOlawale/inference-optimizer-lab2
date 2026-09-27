# inference-optimizer-lab2

LLM inference benchmarking, bottleneck classification, and optimization loop for SGLang and vLLM.

Part of the AI Inference Engineering Roadmap 2026 — Phase 9 onwards.

## Install

```bash
pip install -e .
pip install -e ".[gpu]"   # adds pynvml for GPU monitoring
```

## Commands

```bash
# Default sweep: ctx [512..32K] x c [1..128]
python -m inference_optimizer sweep --engine sglang --model /path/to/model

# Quick validation (3x4 grid)
python -m inference_optimizer sweep --quick --engine vllm --model /path/to/model

# Peak throughput — find saturation cliff (c ramps to 128)
python -m inference_optimizer sweep --peak --engine sglang --model /path/to/model

# Long-context (64K / 128K) — needs FP8 KV or 2 GPUs
python -m inference_optimizer sweep --long-context --engine vllm --model /path/to/model

# Full optimization loop (Measure → Diagnose → Optimize → Re-benchmark)
python -m inference_optimizer optimize --engine sglang --model /path/to/model

# Compare two saved result files
python -m inference_optimizer compare --baseline results/sweep_*.json --optimized results/optimized_*.json
```

## Concurrency limits

| Preset | Max concurrency | Notes |
|---|---|---|
| default | 128 | Practical ceiling for H100 NVL + Qwen3-8B |
| --peak | 128 | Ramps 1→128 on short contexts |
| --long-context | 8 | KV pool fills fast at 64K+ |
| CLI override | up to 256 | `--concurrencies 1 4 8 16 32 64 128 256` — over-subscription test |

## GPU requirements

A GPU requirements table is printed automatically before each sweep showing which
cells need FP8 KV cache (`--kv-cache-dtype fp8`) or multiple GPUs.

General rules for Qwen3-8B on a single H100 NVL (93.6 GB):
- ctx ≤ 8K: all concurrencies safe in BF16
- ctx = 16K–32K: needs `--max-model-len` + FP8 KV at high concurrency
- ctx = 64K: fits on 1 GPU with FP8 KV at c=1 only
- ctx = 128K: needs 2 GPUs in BF16; 1 GPU possible with FP8 KV at c=1

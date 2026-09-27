from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SweepConfig:
    context_lengths: list[int]       # token counts for context
    concurrencies: list[int]         # concurrent request counts
    output_tokens: int               # constant output length
    n_requests_per_cell: int         # total requests per cell
    workloads: list[str]             # ["random"] or ["random", "shared_prefix"]
    shared_prefix_tokens: int        # for shared_prefix workload
    engines: list[str]               # ["sglang"] or ["vllm"] or ["sglang", "vllm"]
    sglang_url: str
    vllm_url: str
    model: str
    temperature: float = 0.0
    output_dir: str = "./results"


def default_config() -> SweepConfig:
    """
    Standard sweep: 7 context lengths × 8 concurrencies × 2 workloads.
    All cells fit on a single H100 NVL in BF16 up to ctx=8K.
    Cells at ctx=16K and 32K require --max-model-len and may need FP8 KV
    at high concurrency — the GPU requirements table printed at startup
    will show exactly which cells need flags.
    """
    return SweepConfig(
        context_lengths=[512, 1024, 2048, 4096, 8192, 16384, 32768],
        concurrencies=[1, 2, 4, 8, 16, 32, 64, 128],
        output_tokens=256,
        n_requests_per_cell=20,
        workloads=["random", "shared_prefix"],
        shared_prefix_tokens=1024,
        engines=["sglang", "vllm"],
        sglang_url="http://localhost:30000",
        vllm_url="http://localhost:8000",
        model="Qwen/Qwen3-8B",
    )


def quick_config() -> SweepConfig:
    """Fast validation: 3 context lengths × 4 concurrencies × 1 workload."""
    cfg = default_config()
    cfg.context_lengths = [512, 2048, 8192]
    cfg.concurrencies = [1, 8, 32, 128]
    cfg.workloads = ["random"]
    cfg.n_requests_per_cell = 10
    return cfg


def long_context_config() -> SweepConfig:
    """
    Long-context sweep: 64K and 128K contexts.

    GPU requirements (Qwen3-8B, H100 NVL 93.6 GB):
      ctx=64K  c=1:  57 GB BF16 — fits on 1 GPU, use --kv-cache-dtype fp8
      ctx=64K  c=2+: needs 2 GPUs in BF16; 1 GPU with FP8 KV at c≤2
      ctx=128K c=1:  114 GB BF16 — needs 2 GPUs; 1 GPU only with FP8 KV
      ctx=128K c=2+: needs 2 GPUs even with FP8 KV

    Minimum server flags to run at all:
      vllm:   --max-model-len 131072 --kv-cache-dtype fp8
      sglang: --context-length 131072 --kv-cache-dtype fp8_e5m2

    Concurrency is kept low intentionally — at 64K+ context the KV pool
    fills with a single sequence; high concurrency just causes queuing.
    """
    cfg = default_config()
    cfg.context_lengths = [65536, 131072]
    cfg.concurrencies = [1, 2, 4, 8]
    cfg.workloads = ["random"]
    cfg.n_requests_per_cell = 10
    return cfg


def peak_config() -> SweepConfig:
    """
    Peak throughput sweep: finds the saturation point for a single H100 NVL.

    Concurrencies ramp from 1 → 128. On H100 NVL with Qwen3-8B:
      - Throughput (tok/s) rises steeply up to ~c=32-64
      - c=64-128: plateau region — adds <10% more throughput but TPOT climbs
      - c=128: the practical ceiling; beyond this, scheduling overhead and
        KV cache pressure dominate and throughput may actually decrease

    Context lengths are kept short (≤8K) so the KV cache pool is not the
    limiting factor — this isolates pure compute/bandwidth saturation.

    To deliberately test over-subscription (where the server queue fills),
    add --concurrencies 1 4 8 16 32 64 128 256 on the CLI.
    """
    cfg = default_config()
    cfg.context_lengths = [512, 2048, 4096, 8192]
    cfg.concurrencies = [1, 4, 8, 16, 32, 64, 128]
    cfg.workloads = ["random"]
    cfg.n_requests_per_cell = 30
    return cfg

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Optimization:
    id: str
    name: str
    bottleneck: str
    description: str
    expected_improvement: str
    metrics_to_watch: list[str]

    # How to apply
    vllm_server_flags: dict[str, str]
    sglang_server_flags: dict[str, str]
    per_request_params: dict[str, str]
    requires_restart: bool

    # Side effects / constraints
    side_effects: str
    applies_to_engines: list[str]
    applies_when: str

    # Priority (lower = try first)
    priority: int


OPTIMIZATION_REGISTRY: dict[str, list[Optimization]] = {
    "prefill_compute": [
        Optimization(
            id="chunked_prefill",
            name="Chunked Prefill",
            bottleneck="prefill_compute",
            description=(
                "Break long prompt prefills into smaller chunks so that prefill and decode "
                "steps can be interleaved. Reduces TTFT spikes under concurrent load by "
                "preventing a single large prefill from blocking all decode steps."
            ),
            expected_improvement="TTFT p95 -20% to -40% on context >2K under concurrent load",
            metrics_to_watch=["ttft_p50_ms", "ttft_p95_ms", "tpot_p50_ms"],
            vllm_server_flags={
                "--enable-chunked-prefill": "",
                "--max-num-batched-tokens": "512",
            },
            sglang_server_flags={"--chunked-prefill-size": "512"},
            per_request_params={},
            requires_restart=True,
            side_effects="Slight increase in scheduling overhead; decode latency may improve.",
            applies_to_engines=["vllm", "sglang"],
            applies_when="TTFT p95 > 2x TTFT p50, or TTFT grows >2x vs c=1 baseline",
            priority=1,
        ),
        Optimization(
            id="cuda_graphs",
            name="CUDA Graph Capture",
            bottleneck="prefill_compute",
            description=(
                "Capture CUDA graphs for decode kernel sequences to reduce per-step kernel "
                "launch overhead. vLLM enables this by default; ensure --enforce-eager is "
                "NOT set. SGLang uses graph capture automatically."
            ),
            expected_improvement="Decode throughput +30-50%, engine init +4s",
            metrics_to_watch=["tpot_p50_ms", "output_tps_total"],
            vllm_server_flags={},
            sglang_server_flags={},
            per_request_params={},
            requires_restart=True,
            side_effects=(
                "Increases GPU memory usage ~2GB for graph capture. "
                "Remove --enforce-eager flag if present."
            ),
            applies_to_engines=["vllm", "sglang"],
            applies_when="Server started with --enforce-eager or cuda graphs disabled",
            priority=2,
        ),
        Optimization(
            id="flash_attention",
            name="FlashAttention Kernel",
            bottleneck="prefill_compute",
            description=(
                "Enable FlashAttention 2/3 kernel for attention computation. "
                "Dramatically reduces HBM reads during prefill by fusing attention ops. "
                "Both vLLM and SGLang use it by default when flash-attn is installed."
            ),
            expected_improvement="Prefill time -40-60% vs naive attention, especially on long contexts",
            metrics_to_watch=["ttft_p50_ms", "ttft_p95_ms"],
            vllm_server_flags={},
            sglang_server_flags={},
            per_request_params={},
            requires_restart=True,
            side_effects="Requires flash-attn package installed. Verify in server startup logs.",
            applies_to_engines=["vllm", "sglang"],
            applies_when="FlashAttention not installed or server using naive attention fallback",
            priority=3,
        ),
        Optimization(
            id="quantize_w8a8",
            name="FP8 Weight+Activation Quantization",
            bottleneck="prefill_compute",
            description=(
                "Apply FP8 quantization to both weights and activations. "
                "Reduces memory bandwidth requirements and enables faster matrix multiply "
                "on H100 NVL tensor cores (FP8 GEMM is 2x faster than BF16)."
            ),
            expected_improvement="Prefill compute -30-50%, marginal quality impact (<1% on most benchmarks)",
            metrics_to_watch=["ttft_p50_ms", "ttft_p95_ms", "output_tps_total"],
            vllm_server_flags={"--quantization": "fp8"},
            sglang_server_flags={"--quantization": "fp8"},
            per_request_params={},
            requires_restart=True,
            side_effects="Model quality may degrade slightly. Best with FP8-calibrated checkpoints.",
            applies_to_engines=["vllm", "sglang"],
            applies_when="Model is BF16/FP16 and GPU supports FP8 (H100, A100)",
            priority=4,
        ),
    ],
    "decode_bandwidth": [
        Optimization(
            id="kv_cache_fp8",
            name="FP8 KV Cache Quantization",
            bottleneck="decode_bandwidth",
            description=(
                "Store KV cache in FP8 instead of BF16/FP16 to halve HBM bandwidth "
                "required for attention computation during decode. Each token's KV "
                "cache takes 2x less memory, allowing more sequences in cache."
            ),
            expected_improvement="TPOT -20-35%, HBM bandwidth -40%, slight quality impact on very long contexts",
            metrics_to_watch=["tpot_p50_ms", "tpot_p95_ms", "output_tps_total"],
            vllm_server_flags={"--kv-cache-dtype": "fp8"},
            sglang_server_flags={"--kv-cache-dtype": "fp8_e5m2"},
            per_request_params={},
            requires_restart=True,
            side_effects="Slight quality degradation on contexts >8K tokens. Negligible for most use cases.",
            applies_to_engines=["vllm", "sglang"],
            applies_when="TPOT rising with high HBM bandwidth utilization (>2000 GB/s)",
            priority=1,
        ),
        Optimization(
            id="prefix_caching",
            name="Radix/Prefix KV Cache",
            bottleneck="decode_bandwidth",
            description=(
                "Enable radix-tree-based prefix caching to reuse KV cache blocks "
                "for shared prompt prefixes. SGLang enables this by default. "
                "For vLLM, requires --enable-prefix-caching flag. "
                "Dramatically reduces TTFT on shared-prefix workloads."
            ),
            expected_improvement="TTFT -30-70% on shared-prefix workloads; no impact on random workloads",
            metrics_to_watch=["ttft_p50_ms", "ttft_p95_ms", "kv_cache_hit_rate"],
            vllm_server_flags={"--enable-prefix-caching": ""},
            sglang_server_flags={},
            per_request_params={},
            requires_restart=True,
            side_effects=(
                "Increases KV cache memory usage. May evict KV blocks for other sequences "
                "under high load. SGLang has it on by default."
            ),
            applies_to_engines=["vllm", "sglang"],
            applies_when="Workload has shared system prompts or repeated context prefixes",
            priority=2,
        ),
        Optimization(
            id="gqa_variant",
            name="GQA/MQA Model Architecture",
            bottleneck="decode_bandwidth",
            description=(
                "Use a model variant with Grouped Query Attention (GQA) or Multi-Query "
                "Attention (MQA). GQA reduces KV cache size by sharing key-value heads "
                "across query heads, directly reducing HBM bandwidth per decode step. "
                "Qwen3-8B already uses GQA — verify configuration."
            ),
            expected_improvement="KV cache size -4-8x vs MHA, TPOT -30-50% (model-level choice)",
            metrics_to_watch=["tpot_p50_ms", "output_tps_total"],
            vllm_server_flags={},
            sglang_server_flags={},
            per_request_params={},
            requires_restart=False,
            side_effects="Requires a model that supports GQA/MQA. Cannot be applied at runtime.",
            applies_to_engines=["vllm", "sglang"],
            applies_when="Current model uses full MHA; switching to GQA variant is possible",
            priority=3,
        ),
    ],
    "kv_cache_capacity": [
        Optimization(
            id="increase_gpu_memory_utilization",
            name="Increase GPU Memory Utilization for KV Cache",
            bottleneck="kv_cache_capacity",
            description=(
                "Allow vLLM/SGLang to use more of the available GPU VRAM for the KV cache "
                "pool. Default is 0.90 for vLLM; pushing to 0.95 can add 5-15% more "
                "KV cache blocks without affecting model weights."
            ),
            expected_improvement="More KV cache blocks, fewer evictions, +10-20% throughput at high concurrency",
            metrics_to_watch=["kv_cache_utilization", "tpot_p50_ms", "output_tps_total"],
            vllm_server_flags={"--gpu-memory-utilization": "0.95"},
            sglang_server_flags={"--mem-fraction-static": "0.85"},
            per_request_params={},
            requires_restart=True,
            side_effects="Risk of OOM if model is large. Monitor memory carefully after change.",
            applies_to_engines=["vllm", "sglang"],
            applies_when="KV cache utilization >80% and GPU has headroom",
            priority=1,
        ),
        Optimization(
            id="kv_cache_quantization",
            name="FP8 KV Cache to Fit More Sequences",
            bottleneck="kv_cache_capacity",
            description=(
                "Store KV cache in FP8 to fit 2x more sequences in the same VRAM. "
                "Directly addresses KV cache capacity bottleneck by halving the "
                "per-token KV footprint."
            ),
            expected_improvement="2x KV cache capacity, supports 2x more concurrent sequences",
            metrics_to_watch=["kv_cache_utilization", "num_queued", "output_tps_total"],
            vllm_server_flags={"--kv-cache-dtype": "fp8"},
            sglang_server_flags={"--kv-cache-dtype": "fp8_e5m2"},
            per_request_params={},
            requires_restart=True,
            side_effects="Minor quality degradation on very long contexts (>8K tokens).",
            applies_to_engines=["vllm", "sglang"],
            applies_when="KV cache utilization consistently >80% causing evictions",
            priority=2,
        ),
        Optimization(
            id="reduce_max_model_len",
            name="Cap Maximum Context Window",
            bottleneck="kv_cache_capacity",
            description=(
                "Reduce the maximum supported context length to free up KV cache VRAM "
                "that was pre-allocated for maximum-length sequences. "
                "If most requests use <4K tokens, capping at 4096 can save significant VRAM."
            ),
            expected_improvement="KV cache capacity +20-50% depending on reduction; direct VRAM savings",
            metrics_to_watch=["kv_cache_utilization", "num_queued"],
            vllm_server_flags={"--max-model-len": "4096"},
            sglang_server_flags={"--context-length": "4096"},
            per_request_params={},
            requires_restart=True,
            side_effects="Requests exceeding the cap will be rejected with an error.",
            applies_to_engines=["vllm", "sglang"],
            applies_when="Most requests use short contexts; max-len is much larger than p99 context",
            priority=3,
        ),
    ],
    "kv_cache_memory": [
        Optimization(
            id="kv_offload_cpu",
            name="CPU KV Cache Offloading",
            bottleneck="kv_cache_memory",
            description=(
                "Offload evicted KV cache blocks to CPU DRAM instead of discarding them. "
                "Allows serving requests with contexts that exceed GPU KV cache capacity "
                "by swapping blocks between GPU and CPU memory."
            ),
            expected_improvement="Supports 2-4x more concurrent sequences at cost of higher TPOT",
            metrics_to_watch=["kv_cache_utilization", "num_queued", "tpot_p50_ms"],
            vllm_server_flags={"--swap-space": "4"},
            sglang_server_flags={},
            per_request_params={},
            requires_restart=True,
            side_effects="CPU-GPU memory transfer adds latency. TPOT increases for swapped sequences.",
            applies_to_engines=["vllm"],
            applies_when="Serving long-context requests where GPU KV cache is insufficient",
            priority=1,
        ),
        Optimization(
            id="block_size_tuning",
            name="KV Cache Block Size Tuning",
            bottleneck="kv_cache_memory",
            description=(
                "Adjust the KV cache block size. Larger blocks reduce fragmentation overhead "
                "but waste memory for short sequences. Smaller blocks improve utilization "
                "for short-context workloads."
            ),
            expected_improvement="KV cache utilization +5-15% depending on workload distribution",
            metrics_to_watch=["kv_cache_utilization", "output_tps_total"],
            vllm_server_flags={"--block-size": "32"},
            sglang_server_flags={},
            per_request_params={},
            requires_restart=True,
            side_effects="Minimal; tune based on median context length in your workload.",
            applies_to_engines=["vllm"],
            applies_when="Mixed short/long context workloads with high KV fragmentation",
            priority=2,
        ),
        Optimization(
            id="enable_sliding_window",
            name="Sliding Window Attention",
            bottleneck="kv_cache_memory",
            description=(
                "Enable sliding window attention to limit KV cache growth for very long "
                "sequences. Only attends to the last W tokens, capping KV cache per sequence."
            ),
            expected_improvement="KV cache per sequence capped at window size; supports very long contexts",
            metrics_to_watch=["kv_cache_utilization", "tpot_p50_ms"],
            vllm_server_flags={},
            sglang_server_flags={},
            per_request_params={},
            requires_restart=False,
            side_effects="Loses attention to tokens outside window; quality impact on tasks requiring full context.",
            applies_to_engines=["vllm", "sglang"],
            applies_when="Model supports sliding window attention (Mistral, Phi-3-mini)",
            priority=3,
        ),
    ],
    "scheduler_queue": [
        Optimization(
            id="hrrn_scheduler",
            name="HRRN (Highest Response Ratio Next) Scheduler",
            bottleneck="scheduler_queue",
            description=(
                "Switch SGLang scheduler from FCFS (First Come First Served) to HRRN "
                "(Highest Response Ratio Next). HRRN prioritizes requests that have been "
                "waiting longest relative to their expected service time, dramatically "
                "reducing starvation and mean TTFT."
            ),
            expected_improvement="Mean TTFT -69%, p99 TTFT -8% vs FCFS on mixed workloads",
            metrics_to_watch=["ttft_p50_ms", "ttft_p95_ms", "ttft_p99_ms"],
            vllm_server_flags={},
            sglang_server_flags={"--schedule-policy": "hrrn"},
            per_request_params={},
            requires_restart=True,
            side_effects="Slightly higher overhead per scheduling decision. No quality impact.",
            applies_to_engines=["sglang"],
            applies_when="TTFT p99/p50 ratio >3x, indicating head-of-line blocking",
            priority=1,
        ),
        Optimization(
            id="chunked_prefill_scheduler",
            name="Chunked Prefill to Reduce HOL Blocking",
            bottleneck="scheduler_queue",
            description=(
                "Enable chunked prefill to interleave prefill and decode steps. "
                "Prevents a single long prefill from blocking all decode steps (head-of-line "
                "blocking), reducing TPOT variance for concurrent requests."
            ),
            expected_improvement="TTFT p99 -20-40%, TPOT variance -30%; decode requests not blocked",
            metrics_to_watch=["ttft_p99_ms", "tpot_p95_ms", "queue_ms"],
            vllm_server_flags={"--enable-chunked-prefill": ""},
            sglang_server_flags={"--chunked-prefill-size": "512"},
            per_request_params={},
            requires_restart=True,
            side_effects="Minor throughput overhead from interleaving. Overall efficiency may drop slightly.",
            applies_to_engines=["vllm", "sglang"],
            applies_when="Mixed short/long prompt workloads with TTFT p99 >> p50",
            priority=2,
        ),
        Optimization(
            id="increase_max_num_seqs",
            name="Increase Maximum Concurrent Sequences",
            bottleneck="scheduler_queue",
            description=(
                "Increase the maximum number of sequences the scheduler will run concurrently. "
                "Default is 256 for vLLM. Increasing to 512+ reduces queue depth at "
                "high concurrency."
            ),
            expected_improvement="Queue depth -50%, req/s +20-40% at high concurrency",
            metrics_to_watch=["num_queued", "req_per_sec", "ttft_p50_ms"],
            vllm_server_flags={"--max-num-seqs": "512"},
            sglang_server_flags={"--max-running-requests": "512"},
            per_request_params={},
            requires_restart=True,
            side_effects="Higher memory pressure; may cause OOM if KV cache is nearly full.",
            applies_to_engines=["vllm", "sglang"],
            applies_when="num_queued > 0 consistently and KV cache utilization is <60%",
            priority=3,
        ),
    ],
    "cpu_overhead": [
        Optimization(
            id="async_engine",
            name="Async Engine Client",
            bottleneck="cpu_overhead",
            description=(
                "Ensure vLLM is running with the async engine client (default in v0.4+). "
                "The async engine decouples request submission from generation, allowing "
                "the scheduler to batch more efficiently and reducing per-request CPU cost."
            ),
            expected_improvement="CPU scheduling overhead -30-50%, throughput +10-20%",
            metrics_to_watch=["req_per_sec", "output_tps_total"],
            vllm_server_flags={},
            sglang_server_flags={},
            per_request_params={},
            requires_restart=False,
            side_effects="None. Async engine is the default; verify not using sync API.",
            applies_to_engines=["vllm"],
            applies_when="Server running with sync API client; low GPU util with high request rate",
            priority=1,
        ),
        Optimization(
            id="increase_num_scheduler_steps",
            name="Increase Continuous Decode Steps",
            bottleneck="cpu_overhead",
            description=(
                "Increase the number of continuous decode steps SGLang runs per scheduler "
                "call. Default is 1; setting to 10 amortizes scheduler overhead across "
                "more decode steps, reducing CPU-GPU synchronization frequency."
            ),
            expected_improvement="Decode throughput +15-25%, scheduler CPU overhead -60%",
            metrics_to_watch=["output_tps_total", "tpot_p50_ms"],
            vllm_server_flags={},
            sglang_server_flags={"--num-continuous-decode-steps": "10"},
            per_request_params={},
            requires_restart=True,
            side_effects="Slightly higher TTFT as more decode steps run before checking for new prefills.",
            applies_to_engines=["sglang"],
            applies_when="GPU util <50% with high request rate; CPU is the bottleneck",
            priority=2,
        ),
        Optimization(
            id="disable_logging_overhead",
            name="Reduce Logging and Metrics Overhead",
            bottleneck="cpu_overhead",
            description=(
                "Disable verbose per-request logging and reduce Prometheus metrics "
                "scrape frequency. On high-throughput servers, logging every request "
                "can add measurable CPU overhead."
            ),
            expected_improvement="CPU overhead -5-15% at very high request rates (>1000 req/s)",
            metrics_to_watch=["req_per_sec", "output_tps_total"],
            vllm_server_flags={"--disable-log-requests": ""},
            sglang_server_flags={},
            per_request_params={},
            requires_restart=True,
            side_effects="Reduces observability; harder to debug individual request failures.",
            applies_to_engines=["vllm"],
            applies_when="Very high request rates (>500 req/s) with CPU overhead confirmed",
            priority=3,
        ),
    ],
    "saturation": [
        Optimization(
            id="tensor_parallelism",
            name="Tensor Parallelism (Multi-GPU)",
            bottleneck="saturation",
            description=(
                "Spread model weights and computation across multiple GPUs using tensor "
                "parallelism. Increases total compute and memory bandwidth proportionally "
                "to the number of GPUs, directly scaling throughput."
            ),
            expected_improvement="Throughput ~2x with 2 GPUs, ~4x with 4 GPUs (sub-linear due to comms)",
            metrics_to_watch=["output_tps_total", "ttft_p50_ms", "tpot_p50_ms"],
            vllm_server_flags={"--tensor-parallel-size": "2"},
            sglang_server_flags={"--tp-size": "2"},
            per_request_params={},
            requires_restart=True,
            side_effects="Requires NVLink or fast interconnect between GPUs. Adds ~5% comms overhead.",
            applies_to_engines=["vllm", "sglang"],
            applies_when="Single GPU is fully saturated; multiple GPUs available in the node",
            priority=1,
        ),
        Optimization(
            id="data_parallelism",
            name="Data Parallelism (Multiple Replicas)",
            bottleneck="saturation",
            description=(
                "Run multiple independent model replicas on separate GPUs or nodes, "
                "with a load balancer distributing requests across replicas. "
                "Ideal when each GPU can handle the model independently."
            ),
            expected_improvement="Throughput scales linearly with replicas (perfect scaling)",
            metrics_to_watch=["output_tps_total", "req_per_sec"],
            vllm_server_flags={},
            sglang_server_flags={},
            per_request_params={},
            requires_restart=True,
            side_effects="No KV cache sharing between replicas; higher total memory footprint.",
            applies_to_engines=["vllm", "sglang"],
            applies_when="Multiple GPUs/nodes available; model fits on single GPU",
            priority=2,
        ),
        Optimization(
            id="speculative_decoding",
            name="Speculative Decoding (ngram Draft)",
            bottleneck="saturation",
            description=(
                "Use speculative decoding with an ngram draft model to generate multiple "
                "tokens per forward pass. The draft model proposes N tokens, the target "
                "model verifies them all in one pass, achieving >1 accepted token per step."
            ),
            expected_improvement="Decode throughput +1.5-3x on low-entropy outputs; TPOT -30-50%",
            metrics_to_watch=["output_tps_total", "tpot_p50_ms"],
            vllm_server_flags={
                "--speculative-config": '{"method": "ngram", "num_speculative_tokens": 5}'
            },
            sglang_server_flags={"--speculative-algorithm": "EAGLE"},
            per_request_params={},
            requires_restart=True,
            side_effects=(
                "Higher GPU memory usage for draft model. "
                "Throughput gain depends heavily on output entropy (lower = better)."
            ),
            applies_to_engines=["vllm", "sglang"],
            applies_when="Decode-bound workload; output is relatively predictable (code, structured data)",
            priority=3,
        ),
    ],
    "unknown": [
        Optimization(
            id="baseline_profile",
            name="Profile with Detailed Metrics",
            bottleneck="unknown",
            description=(
                "Run a targeted profiling sweep with GPU profiler (Nsight Systems) to "
                "identify the actual bottleneck. Collect detailed kernel-level timing, "
                "memory access patterns, and compute utilization per operation."
            ),
            expected_improvement="Provides ground-truth bottleneck data for targeted optimization",
            metrics_to_watch=["ttft_p50_ms", "tpot_p50_ms", "output_tps_total"],
            vllm_server_flags={},
            sglang_server_flags={},
            per_request_params={},
            requires_restart=False,
            side_effects="Profiling adds overhead; run on representative workload sample.",
            applies_to_engines=["vllm", "sglang"],
            applies_when="No clear bottleneck pattern; need ground-truth profiling data",
            priority=1,
        ),
        Optimization(
            id="network_optimization",
            name="Network / Load Balancer Tuning",
            bottleneck="unknown",
            description=(
                "Check network path between client and server. High e2e latency with "
                "low TTFT may indicate network overhead. Ensure client and server are "
                "co-located; use Unix socket or InfiniBand if available."
            ),
            expected_improvement="E2E latency -5-15% when network is the bottleneck",
            metrics_to_watch=["e2e_p50_ms", "queue_ms"],
            vllm_server_flags={},
            sglang_server_flags={},
            per_request_params={},
            requires_restart=False,
            side_effects="None.",
            applies_to_engines=["vllm", "sglang"],
            applies_when="queue_ms is high but GPU utilization and KV cache are normal",
            priority=2,
        ),
        Optimization(
            id="batch_size_sweep",
            name="Max Batch Size Sweep",
            bottleneck="unknown",
            description=(
                "Sweep max-num-batched-tokens to find the batch size that maximizes "
                "throughput for your workload. Too small = under-utilizes GPU; "
                "too large = increases latency."
            ),
            expected_improvement="Throughput +10-30% by finding optimal batch size",
            metrics_to_watch=["output_tps_total", "ttft_p50_ms", "tpot_p50_ms"],
            vllm_server_flags={"--max-num-batched-tokens": "4096"},
            sglang_server_flags={},
            per_request_params={},
            requires_restart=True,
            side_effects="Larger batches increase latency; smaller batches reduce throughput.",
            applies_to_engines=["vllm"],
            applies_when="Unknown bottleneck with sub-optimal throughput",
            priority=3,
        ),
    ],
}


def get_optimizations(bottleneck: str, engine: str) -> list[Optimization]:
    """Return optimizations for this bottleneck, filtered by engine, sorted by priority."""
    opts = OPTIMIZATION_REGISTRY.get(bottleneck, [])
    return sorted(
        [
            o
            for o in opts
            if not o.applies_to_engines or engine in o.applies_to_engines
        ],
        key=lambda o: o.priority,
    )


def generate_server_command(
    engine: str,
    model: str,
    url_port: int,
    base_flags: dict[str, str],
    optimizations: list[Optimization],
) -> str:
    """Generate the server start command with optimization flags applied."""
    all_flags: dict[str, str] = dict(base_flags)

    for opt in optimizations:
        if engine == "vllm":
            all_flags.update(opt.vllm_server_flags)
        elif engine == "sglang":
            all_flags.update(opt.sglang_server_flags)

    if engine == "vllm":
        cmd_parts = [
            f"python -m vllm.entrypoints.openai.api_server",
            f"--model {model}",
            f"--port {url_port}",
        ]
    else:
        cmd_parts = [
            f"python -m sglang.launch_server",
            f"--model-path {model}",
            f"--port {url_port}",
        ]

    for flag, value in all_flags.items():
        if value:
            cmd_parts.append(f"{flag} {value}")
        else:
            cmd_parts.append(flag)

    return " \\\n  ".join(cmd_parts)

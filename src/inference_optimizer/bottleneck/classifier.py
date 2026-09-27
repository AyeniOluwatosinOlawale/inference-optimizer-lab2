from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..types import CellResult


def classify(
    cell: CellResult,
    baseline_cell: CellResult | None = None,
    prev_concurrency_cell: CellResult | None = None,
) -> tuple[str, str, str]:
    """
    Classify the primary bottleneck for a cell result.

    Returns (bottleneck, confidence, reason).

    Rules applied in priority order — first match wins.
    """
    gpu = cell.gpu
    snap_post = cell.server_post

    good = [r for r in cell.requests if r.succeeded]
    queue_ms_mean = (
        sum(r.queue_ms for r in good) / len(good) if good else 0.0
    )

    # --- Rule 1: TTFT growing due to prefill compute (GPU busy) ---
    if baseline_cell is not None and baseline_cell.ttft_p50_ms > 0:
        ttft_ratio = cell.ttft_p50_ms / baseline_cell.ttft_p50_ms
        if ttft_ratio > 2.0 and gpu.util_mean_pct > 80:
            return (
                "prefill_compute",
                "high",
                f"TTFT grew {ttft_ratio:.1f}x vs baseline (c=1) and GPU util "
                f"is {gpu.util_mean_pct:.0f}% — prefill saturating compute.",
            )

    # --- Rule 2: TTFT growing but GPU not saturated ---
    if baseline_cell is not None and baseline_cell.ttft_p50_ms > 0:
        ttft_ratio = cell.ttft_p50_ms / baseline_cell.ttft_p50_ms
        if ttft_ratio > 2.0 and gpu.util_mean_pct <= 80:
            return (
                "prefill_compute",
                "medium",
                f"TTFT grew {ttft_ratio:.1f}x vs baseline but GPU util "
                f"is only {gpu.util_mean_pct:.0f}% — likely prefill contention.",
            )

    # --- Rule 3: Requests spending most of their time queued ---
    if cell.e2e_p50_ms > 0:
        queue_fraction = queue_ms_mean / cell.e2e_p50_ms
        if queue_fraction > 0.3 and gpu.util_mean_pct < 70:
            return (
                "scheduler_queue",
                "high",
                f"Mean queue wait {queue_ms_mean:.0f}ms is "
                f"{queue_fraction * 100:.0f}% of e2e latency "
                f"with GPU at {gpu.util_mean_pct:.0f}% — scheduler queueing bottleneck.",
            )

    # --- Rule 4: KV cache full causing evictions ---
    if snap_post.kv_cache_utilization > 0.80:
        if prev_concurrency_cell is not None and prev_concurrency_cell.tpot_p50_ms > 0:
            tpot_ratio = cell.tpot_p50_ms / prev_concurrency_cell.tpot_p50_ms
            if tpot_ratio > 1.0:
                return (
                    "kv_cache_capacity",
                    "high",
                    f"KV cache utilization {snap_post.kv_cache_utilization:.0%} "
                    f"and TPOT rose {tpot_ratio:.1f}x — cache evictions slowing decode.",
                )

    # --- Rule 5: TPOT rising due to HBM bandwidth saturation ---
    if prev_concurrency_cell is not None and prev_concurrency_cell.tpot_p50_ms > 0:
        tpot_ratio = cell.tpot_p50_ms / prev_concurrency_cell.tpot_p50_ms
        if tpot_ratio > 1.2 and gpu.hbm_bw_mean_gbps > 2000:
            return (
                "decode_bandwidth",
                "high",
                f"TPOT rose {tpot_ratio:.1f}x vs lower concurrency and "
                f"HBM BW is {gpu.hbm_bw_mean_gbps:.0f} GB/s — decode bandwidth bound.",
            )

    # --- Rule 6: TPOT rising due to GPU compute ---
    if prev_concurrency_cell is not None and prev_concurrency_cell.tpot_p50_ms > 0:
        tpot_ratio = cell.tpot_p50_ms / prev_concurrency_cell.tpot_p50_ms
        if tpot_ratio > 1.2 and gpu.util_mean_pct > 80:
            return (
                "decode_compute",
                "medium",
                f"TPOT rose {tpot_ratio:.1f}x and GPU util "
                f"is {gpu.util_mean_pct:.0f}% — decode compute saturated.",
            )

    # --- Rule 7: Low GPU util but high latency suggests CPU overhead ---
    if gpu.available and gpu.util_mean_pct < 50:
        if cell.ttft_p50_ms > 200 or cell.tpot_p50_ms > 20:
            return (
                "cpu_overhead",
                "medium",
                f"GPU util is only {gpu.util_mean_pct:.0f}% but "
                f"TTFT p50={cell.ttft_p50_ms:.0f}ms / TPOT p50={cell.tpot_p50_ms:.1f}ms — "
                f"likely CPU-bound scheduling or tokenization overhead.",
            )

    # --- Rule 8: Throughput saturated — adding concurrency buys nothing ---
    if prev_concurrency_cell is not None and prev_concurrency_cell.output_tps_total > 0:
        tps_gain = (
            cell.output_tps_total / prev_concurrency_cell.output_tps_total - 1
        )
        if tps_gain < 0.05:
            return (
                "saturation",
                "high",
                f"Throughput gain from increasing concurrency is only "
                f"{tps_gain * 100:.1f}% — system is saturated.",
            )

    # --- Rule 9: Shared-prefix workload with high cache hit rate ---
    if cell.key.workload == "shared_prefix" and snap_post.kv_cache_hit_rate > 0.7:
        if baseline_cell is not None and baseline_cell.ttft_p50_ms > 0:
            ttft_ratio = cell.ttft_p50_ms / baseline_cell.ttft_p50_ms
            if ttft_ratio < 0.7:
                return (
                    "cache_effective",
                    "high",
                    f"KV cache hit rate {snap_post.kv_cache_hit_rate:.0%} on "
                    f"shared-prefix workload; TTFT is {ttft_ratio:.1f}x of random baseline — "
                    f"prefix caching working effectively.",
                )

    # --- Rule 10: Fallthrough ---
    return (
        "unknown",
        "low",
        "No dominant bottleneck pattern detected from available metrics.",
    )

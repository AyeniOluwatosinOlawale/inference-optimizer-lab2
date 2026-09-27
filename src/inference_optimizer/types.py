from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class RequestRecord:
    request_id: str
    engine: str          # "sglang" | "vllm"
    model: str
    context_tokens: int
    output_tokens_requested: int
    output_tokens_actual: int
    concurrency: int
    workload: str        # "random" | "shared_prefix"

    ttft_ms: float
    tpot_ms: float
    itl_ms_list: list[float]    # gap between each consecutive content chunk
    itl_mean_ms: float
    e2e_ms: float
    queue_ms: float             # time waiting for semaphore (client-side queue proxy)
    output_tps: float           # output_tokens / (e2e_ms / 1000)

    error: str | None = None

    @property
    def succeeded(self) -> bool:
        return self.error is None


@dataclass
class GPUSummary:
    util_mean_pct: float = 0.0
    util_peak_pct: float = 0.0
    memory_used_mean_gb: float = 0.0
    memory_used_peak_gb: float = 0.0
    power_mean_watts: float = 0.0
    temperature_mean_c: float = 0.0
    hbm_bw_mean_gbps: float = 0.0    # estimated from mem clock x bus width
    sm_clock_mean_mhz: float = 0.0
    available: bool = False


@dataclass
class ServerSnapshot:
    kv_cache_hit_rate: float = 0.0
    kv_cache_utilization: float = 0.0
    num_running: int = 0
    num_queued: int = 0
    decode_tps: float = 0.0
    prefill_tps: float = 0.0


@dataclass
class CellKey:
    engine: str
    context_tokens: int
    concurrency: int
    workload: str

    def label(self) -> str:
        return f"{self.engine}|ctx={self.context_tokens}|c={self.concurrency}|{self.workload}"


@dataclass
class CellResult:
    key: CellKey
    requests: list[RequestRecord]

    # Latency percentiles
    ttft_p50_ms: float
    ttft_p95_ms: float
    ttft_p99_ms: float
    tpot_p50_ms: float
    tpot_p95_ms: float
    itl_mean_ms: float
    e2e_p50_ms: float
    e2e_p95_ms: float
    e2e_p99_ms: float

    # Throughput
    output_tps_total: float     # total output tokens/s across all concurrent requests
    req_per_sec: float
    error_rate: float

    # GPU
    gpu: GPUSummary

    # Server
    server_pre: ServerSnapshot
    server_post: ServerSnapshot

    # Bottleneck
    bottleneck: str
    bottleneck_confidence: str   # "high" | "medium" | "low"
    bottleneck_reason: str


def _pct(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    idx = (len(s) - 1) * p / 100.0
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def aggregate_requests(
    key: CellKey,
    requests: list[RequestRecord],
    gpu: GPUSummary,
    server_pre: ServerSnapshot,
    server_post: ServerSnapshot,
    bottleneck: str,
    bottleneck_confidence: str,
    bottleneck_reason: str,
    wall_seconds: float,
) -> CellResult:
    good = [r for r in requests if r.succeeded]
    ttfts = [r.ttft_ms for r in good]
    tpots = [r.tpot_ms for r in good]
    e2es = [r.e2e_ms for r in good]
    itls = [r.itl_mean_ms for r in good]
    total_output_tokens = sum(r.output_tokens_actual for r in good)
    return CellResult(
        key=key,
        requests=requests,
        ttft_p50_ms=_pct(ttfts, 50),
        ttft_p95_ms=_pct(ttfts, 95),
        ttft_p99_ms=_pct(ttfts, 99),
        tpot_p50_ms=_pct(tpots, 50),
        tpot_p95_ms=_pct(tpots, 95),
        itl_mean_ms=sum(itls) / len(itls) if itls else 0.0,
        e2e_p50_ms=_pct(e2es, 50),
        e2e_p95_ms=_pct(e2es, 95),
        e2e_p99_ms=_pct(e2es, 99),
        output_tps_total=total_output_tokens / wall_seconds if wall_seconds > 0 else 0.0,
        req_per_sec=len(good) / wall_seconds if wall_seconds > 0 else 0.0,
        error_rate=(len(requests) - len(good)) / len(requests) if requests else 0.0,
        gpu=gpu,
        server_pre=server_pre,
        server_post=server_post,
        bottleneck=bottleneck,
        bottleneck_confidence=bottleneck_confidence,
        bottleneck_reason=bottleneck_reason,
    )

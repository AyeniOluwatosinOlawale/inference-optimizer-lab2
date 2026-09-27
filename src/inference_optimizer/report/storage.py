from __future__ import annotations

import csv
import dataclasses
import json
import os
from datetime import datetime, timezone

from ..types import (
    CellKey,
    CellResult,
    GPUSummary,
    RequestRecord,
    ServerSnapshot,
)


def _record_to_dict(r: RequestRecord) -> dict:
    return {
        "request_id": r.request_id,
        "engine": r.engine,
        "model": r.model,
        "context_tokens": r.context_tokens,
        "output_tokens_requested": r.output_tokens_requested,
        "output_tokens_actual": r.output_tokens_actual,
        "concurrency": r.concurrency,
        "workload": r.workload,
        "ttft_ms": r.ttft_ms,
        "tpot_ms": r.tpot_ms,
        "itl_mean_ms": r.itl_mean_ms,
        "e2e_ms": r.e2e_ms,
        "queue_ms": r.queue_ms,
        "output_tps": r.output_tps,
        "error": r.error,
    }


def _cell_to_summary_row(cell: CellResult) -> dict:
    return {
        "engine": cell.key.engine,
        "context_tokens": cell.key.context_tokens,
        "concurrency": cell.key.concurrency,
        "workload": cell.key.workload,
        "ttft_p50_ms": cell.ttft_p50_ms,
        "ttft_p95_ms": cell.ttft_p95_ms,
        "ttft_p99_ms": cell.ttft_p99_ms,
        "tpot_p50_ms": cell.tpot_p50_ms,
        "tpot_p95_ms": cell.tpot_p95_ms,
        "itl_mean_ms": cell.itl_mean_ms,
        "e2e_p50_ms": cell.e2e_p50_ms,
        "e2e_p95_ms": cell.e2e_p95_ms,
        "e2e_p99_ms": cell.e2e_p99_ms,
        "output_tps_total": cell.output_tps_total,
        "req_per_sec": cell.req_per_sec,
        "error_rate": cell.error_rate,
        "gpu_util_mean_pct": cell.gpu.util_mean_pct,
        "gpu_util_peak_pct": cell.gpu.util_peak_pct,
        "gpu_memory_used_mean_gb": cell.gpu.memory_used_mean_gb,
        "gpu_hbm_bw_mean_gbps": cell.gpu.hbm_bw_mean_gbps,
        "kv_cache_hit_rate": cell.server_post.kv_cache_hit_rate,
        "kv_cache_utilization": cell.server_post.kv_cache_utilization,
        "num_running_post": cell.server_post.num_running,
        "num_queued_post": cell.server_post.num_queued,
        "bottleneck": cell.bottleneck,
        "bottleneck_confidence": cell.bottleneck_confidence,
        "bottleneck_reason": cell.bottleneck_reason,
    }


def save_results(
    cells: list[CellResult],
    output_dir: str,
    label: str = "sweep",
) -> dict[str, str]:
    """
    Save full per-request JSON and per-cell CSV summary.
    Returns dict of written file paths.
    """
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # Full JSON
    json_path = os.path.join(output_dir, f"results_{label}_{ts}.json")
    all_records: list[dict] = []
    for cell in cells:
        for r in cell.requests:
            all_records.append(_record_to_dict(r))
    with open(json_path, "w") as f:
        json.dump(all_records, f, indent=2)

    # Summary CSV
    csv_path = os.path.join(output_dir, f"summary_{label}_{ts}.csv")
    rows = [_cell_to_summary_row(cell) for cell in cells]
    if rows:
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    return {"json": json_path, "csv": csv_path}


def _dict_to_record(d: dict) -> RequestRecord:
    return RequestRecord(
        request_id=d.get("request_id", ""),
        engine=d.get("engine", ""),
        model=d.get("model", ""),
        context_tokens=int(d.get("context_tokens", 0)),
        output_tokens_requested=int(d.get("output_tokens_requested", 0)),
        output_tokens_actual=int(d.get("output_tokens_actual", 0)),
        concurrency=int(d.get("concurrency", 1)),
        workload=d.get("workload", "random"),
        ttft_ms=float(d.get("ttft_ms", 0.0)),
        tpot_ms=float(d.get("tpot_ms", 0.0)),
        itl_ms_list=[],
        itl_mean_ms=float(d.get("itl_mean_ms", 0.0)),
        e2e_ms=float(d.get("e2e_ms", 0.0)),
        queue_ms=float(d.get("queue_ms", 0.0)),
        output_tps=float(d.get("output_tps", 0.0)),
        error=d.get("error"),
    )


def load_results(path: str) -> list[CellResult]:
    """
    Load previously saved results JSON and reconstruct CellResult objects.
    Groups records by (engine, context_tokens, concurrency, workload).
    """
    with open(path) as f:
        records_raw: list[dict] = json.load(f)

    # Group records by cell key
    from collections import defaultdict
    groups: dict[tuple, list[RequestRecord]] = defaultdict(list)
    for d in records_raw:
        key = (
            d.get("engine", ""),
            int(d.get("context_tokens", 0)),
            int(d.get("concurrency", 1)),
            d.get("workload", "random"),
        )
        groups[key].append(_dict_to_record(d))

    cells: list[CellResult] = []
    for (engine, ctx, conc, workload), reqs in groups.items():
        key = CellKey(engine=engine, context_tokens=ctx, concurrency=conc, workload=workload)
        good = [r for r in reqs if r.succeeded]
        ttfts = [r.ttft_ms for r in good]
        tpots = [r.tpot_ms for r in good]
        e2es = [r.e2e_ms for r in good]
        itls = [r.itl_mean_ms for r in good]
        total_output = sum(r.output_tokens_actual for r in good)
        wall_seconds = max(r.e2e_ms for r in reqs) / 1000.0 if reqs else 1.0

        from ..types import _pct
        cell = CellResult(
            key=key,
            requests=reqs,
            ttft_p50_ms=_pct(ttfts, 50),
            ttft_p95_ms=_pct(ttfts, 95),
            ttft_p99_ms=_pct(ttfts, 99),
            tpot_p50_ms=_pct(tpots, 50),
            tpot_p95_ms=_pct(tpots, 95),
            itl_mean_ms=sum(itls) / len(itls) if itls else 0.0,
            e2e_p50_ms=_pct(e2es, 50),
            e2e_p95_ms=_pct(e2es, 95),
            e2e_p99_ms=_pct(e2es, 99),
            output_tps_total=total_output / wall_seconds if wall_seconds > 0 else 0.0,
            req_per_sec=len(good) / wall_seconds if wall_seconds > 0 else 0.0,
            error_rate=(len(reqs) - len(good)) / len(reqs) if reqs else 0.0,
            gpu=GPUSummary(),
            server_pre=ServerSnapshot(),
            server_post=ServerSnapshot(),
            bottleneck="unknown",
            bottleneck_confidence="low",
            bottleneck_reason="loaded from file",
        )
        cells.append(cell)

    return cells

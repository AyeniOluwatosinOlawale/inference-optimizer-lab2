from __future__ import annotations

from ..types import CellResult


def find_saturation_point(
    cells: list[CellResult],
    engine: str,
    context_tokens: int,
    workload: str,
) -> int | None:
    """
    For a given engine/context/workload, sort cells by concurrency.
    Find where throughput gain drops below 5%.
    Return the concurrency level at saturation (or None if not reached).
    """
    relevant = [
        c
        for c in cells
        if c.key.engine == engine
        and c.key.context_tokens == context_tokens
        and c.key.workload == workload
    ]
    relevant.sort(key=lambda c: c.key.concurrency)

    if len(relevant) < 2:
        return None

    for i in range(1, len(relevant)):
        prev_tps = relevant[i - 1].output_tps_total
        curr_tps = relevant[i].output_tps_total
        if prev_tps <= 0:
            continue
        gain = (curr_tps - prev_tps) / prev_tps
        if gain < 0.05:
            return relevant[i - 1].key.concurrency

    return None


def build_heatmap(
    cells: list[CellResult],
    engine: str,
    workload: str,
    metric: str,
) -> dict[tuple[int, int], float]:
    """
    Returns {(context_tokens, concurrency): metric_value}
    metric: "ttft_p50_ms" | "tpot_p50_ms" | "output_tps_total" | "kv_cache_hit_rate"
    """
    result: dict[tuple[int, int], float] = {}
    for cell in cells:
        if cell.key.engine != engine or cell.key.workload != workload:
            continue
        if metric == "kv_cache_hit_rate":
            value = cell.server_post.kv_cache_hit_rate
        else:
            value = getattr(cell, metric, 0.0)
        result[(cell.key.context_tokens, cell.key.concurrency)] = value
    return result


def engine_comparison(cells: list[CellResult]) -> list[dict]:
    """
    For each matching (context_tokens, concurrency, workload) pair across engines,
    compute relative difference in TTFT, TPOT, TPS, KV hit rate.
    Return list of comparison dicts.
    """
    # Group by (context_tokens, concurrency, workload, engine)
    by_key: dict[tuple, CellResult] = {}
    for cell in cells:
        k = (cell.key.context_tokens, cell.key.concurrency, cell.key.workload, cell.key.engine)
        by_key[k] = cell

    # Find all unique (ctx, conc, workload) triples
    triples: set[tuple] = set()
    for cell in cells:
        triples.add((cell.key.context_tokens, cell.key.concurrency, cell.key.workload))

    results: list[dict] = []
    for ctx, conc, workload in sorted(triples):
        engines_present = set()
        for cell in cells:
            k = cell.key
            if k.context_tokens == ctx and k.concurrency == conc and k.workload == workload:
                engines_present.add(k.engine)

        if len(engines_present) < 2:
            continue

        engine_list = sorted(engines_present)
        row: dict = {
            "context_tokens": ctx,
            "concurrency": conc,
            "workload": workload,
        }

        for engine in engine_list:
            cell = by_key.get((ctx, conc, workload, engine))
            if cell:
                row[f"{engine}_ttft_p50_ms"] = cell.ttft_p50_ms
                row[f"{engine}_tpot_p50_ms"] = cell.tpot_p50_ms
                row[f"{engine}_output_tps_total"] = cell.output_tps_total
                row[f"{engine}_kv_cache_hit_rate"] = cell.server_post.kv_cache_hit_rate

        # Compute relative differences (first engine vs second)
        if len(engine_list) >= 2:
            e0, e1 = engine_list[0], engine_list[1]
            for metric in ["ttft_p50_ms", "tpot_p50_ms", "output_tps_total"]:
                v0 = row.get(f"{e0}_{metric}", 0.0)
                v1 = row.get(f"{e1}_{metric}", 0.0)
                if v0 > 0:
                    row[f"relative_{metric}"] = (v1 - v0) / v0 * 100
                else:
                    row[f"relative_{metric}"] = 0.0

        results.append(row)

    return results

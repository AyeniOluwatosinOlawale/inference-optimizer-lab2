from __future__ import annotations

from collections import defaultdict
from typing import TYPE_CHECKING

from ..analysis.compare import find_saturation_point
from ..optimizations.registry import Optimization, generate_server_command
from ..types import CellResult

if TYPE_CHECKING:
    from ..loop.optimizer import OptimizationRun


def print_cell(cell: CellResult) -> None:
    """One-line summary of a cell result."""
    good = sum(1 for r in cell.requests if r.succeeded)
    total = len(cell.requests)
    print(
        f"  [{cell.key.engine}] ctx={cell.key.context_tokens:>5} c={cell.key.concurrency:>3} "
        f"{cell.key.workload:<15} | "
        f"TTFT p50={cell.ttft_p50_ms:>7.1f}ms p95={cell.ttft_p95_ms:>7.1f}ms | "
        f"TPOT p50={cell.tpot_p50_ms:>6.1f}ms | "
        f"TPS={cell.output_tps_total:>7.1f} | "
        f"ok={good}/{total} | "
        f"[{cell.bottleneck}/{cell.bottleneck_confidence}]"
    )


def print_sweep_table(
    cells: list[CellResult],
    engine: str,
    workload: str,
) -> None:
    """
    Print a table: Context | C=1 | C=2 | C=4 | ... showing TTFT p50 with bottleneck annotation.
    """
    relevant = [
        c for c in cells
        if c.key.engine == engine and c.key.workload == workload
    ]
    if not relevant:
        print(f"  No data for engine={engine} workload={workload}")
        return

    concurrencies = sorted(set(c.key.concurrency for c in relevant))
    context_lengths = sorted(set(c.key.context_tokens for c in relevant))

    by_key: dict[tuple, CellResult] = {}
    for cell in relevant:
        by_key[(cell.key.context_tokens, cell.key.concurrency)] = cell

    print(f"\n  TTFT p50 (ms) — engine={engine}  workload={workload}")
    header = f"  {'Context':>8}"
    for c in concurrencies:
        header += f"  {'C='+str(c):>9}"
    print(header)
    print("  " + "-" * (10 + len(concurrencies) * 11))

    for ctx in context_lengths:
        row = f"  {ctx:>8}"
        bottlenecks: list[str] = []
        for c in concurrencies:
            cell = by_key.get((ctx, c))
            if cell:
                row += f"  {cell.ttft_p50_ms:>9.1f}"
                if cell.bottleneck not in ("unknown", "cache_effective"):
                    bottlenecks.append(cell.bottleneck[:6])
            else:
                row += f"  {'---':>9}"
        if bottlenecks:
            row += f"  [{','.join(set(bottlenecks))}]"
        print(row)


def print_bottleneck_summary(cells: list[CellResult]) -> None:
    """Print count of each bottleneck type found across all cells."""
    counts: dict[str, int] = defaultdict(int)
    for cell in cells:
        counts[cell.bottleneck] += 1

    print("\n  Bottleneck Summary")
    print("  " + "-" * 40)
    for bt, count in sorted(counts.items(), key=lambda x: -x[1]):
        bar = "#" * min(count * 2, 40)
        print(f"  {bt:<25} {count:>4}  {bar}")


def print_optimization_plan(
    opts: list[Optimization],
    engine: str,
) -> None:
    """Print recommended optimizations with server commands."""
    print("\n  Optimization Plan")
    print("  " + "=" * 60)
    for i, opt in enumerate(opts, 1):
        print(f"\n  [{i}] {opt.name}  (bottleneck: {opt.bottleneck})")
        print(f"       {opt.expected_improvement}")
        print(f"       Metrics to watch: {', '.join(opt.metrics_to_watch)}")
        if opt.requires_restart:
            print(f"       ** Requires server restart **")
        print(f"       Side effects: {opt.side_effects}")


def print_comparison_table(run: OptimizationRun) -> None:
    """Print before/after for each metric that changed."""
    print("\n" + "=" * 70)
    print(f"COMPARISON: {run.optimization.name}")
    print("=" * 70)
    print(f"\n  {'Metric':<25}  {'Before':>10}  {'After':>10}  {'Change':>10}")
    print("  " + "-" * 60)

    metric_labels = {
        "ttft_p50_ms": "TTFT p50 (ms)",
        "ttft_p95_ms": "TTFT p95 (ms)",
        "tpot_p50_ms": "TPOT p50 (ms)",
        "e2e_p50_ms": "E2E p50 (ms)",
        "output_tps_total": "Throughput (tok/s)",
        "req_per_sec": "Req/s",
        "error_rate": "Error rate",
    }

    def _mean(cells: list[CellResult], attr: str) -> float:
        vals = [getattr(c, attr) for c in cells if getattr(c, attr, 0) > 0]
        return sum(vals) / len(vals) if vals else 0.0

    for metric, label in metric_labels.items():
        before_val = _mean(run.before_cells, metric)
        after_val = _mean(run.after_cells, metric)
        pct = run.improvement.get(metric, 0.0)
        if abs(pct) < 0.5:
            continue
        arrow = "+" if pct > 0 else ""
        print(
            f"  {label:<25}  {before_val:>10.2f}  {after_val:>10.2f}  "
            f"{arrow}{pct:>8.1f}%"
        )

    print(f"\n  VERDICT: {run.verdict.upper()} — {run.verdict_reason}")


def print_engine_comparison(comparison: list[dict]) -> None:
    """Print SGLang vs vLLM side-by-side."""
    if not comparison:
        print("  No engine comparison data available.")
        return

    print("\n  SGLang vs vLLM — TTFT p50 (ms)")
    print(f"  {'Context':>8}  {'Conc':>6}  {'Workload':<15}  "
          f"{'SGLang':>10}  {'vLLM':>10}  {'Diff%':>8}")
    print("  " + "-" * 70)

    for row in comparison[:30]:
        ctx = row.get("context_tokens", 0)
        conc = row.get("concurrency", 0)
        workload = row.get("workload", "")
        sglang_ttft = row.get("sglang_ttft_p50_ms", 0.0)
        vllm_ttft = row.get("vllm_ttft_p50_ms", 0.0)
        diff = row.get("relative_ttft_p50_ms", 0.0)
        print(
            f"  {ctx:>8}  {conc:>6}  {workload:<15}  "
            f"{sglang_ttft:>10.1f}  {vllm_ttft:>10.1f}  {diff:>+8.1f}%"
        )


def print_saturation_analysis(cells: list[CellResult]) -> None:
    """Print saturation points for each engine/context combination."""
    print("\n  Saturation Analysis (throughput gain <5% threshold)")
    print(f"  {'Engine':<10}  {'Context':>8}  {'Workload':<15}  {'Sat. at C':>10}")
    print("  " + "-" * 55)

    engines = sorted(set(c.key.engine for c in cells))
    context_lengths = sorted(set(c.key.context_tokens for c in cells))
    workloads = sorted(set(c.key.workload for c in cells))

    for engine in engines:
        for ctx in context_lengths:
            for workload in workloads:
                sat = find_saturation_point(cells, engine, ctx, workload)
                sat_str = str(sat) if sat is not None else "not reached"
                print(f"  {engine:<10}  {ctx:>8}  {workload:<15}  {sat_str:>10}")

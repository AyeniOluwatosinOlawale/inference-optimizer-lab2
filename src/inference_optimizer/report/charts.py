"""
Visualization charts for inference sweep results.

Charts produced:
  1. ttft_heatmap        — TTFT p50 (ms) grid: context × concurrency per engine/workload
  2. tpot_heatmap        — TPOT p50 (ms) grid
  3. gpu_util_heatmap    — GPU utilization % grid
  4. throughput_curve    — tok/s vs concurrency, one line per context length (saturation)
  5. latency_curve       — TTFT p50 + p95 vs concurrency per context
  6. engine_comparison   — side-by-side bar: SGLang vs vLLM on TTFT / TPOT / TPS
  7. bottleneck_dist     — bar chart of bottleneck type counts
  8. optimization_delta  — grouped bar: before vs after each optimization pass

All charts saved as PNG to output_dir. Returns list of saved file paths.
"""

from __future__ import annotations

import os
from collections import defaultdict
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..loop.optimizer import OptimizationRun
    from ..types import CellResult

# Bottleneck colour palette — consistent across all charts
BOTTLENECK_COLORS: dict[str, str] = {
    "prefill_compute":   "#e74c3c",
    "decode_bandwidth":  "#e67e22",
    "decode_compute":    "#f39c12",
    "kv_cache_capacity": "#9b59b6",
    "kv_cache_memory":   "#8e44ad",
    "scheduler_queue":   "#3498db",
    "cpu_overhead":      "#1abc9c",
    "saturation":        "#2c3e50",
    "cache_effective":   "#27ae60",
    "unknown":           "#bdc3c7",
}

ENGINE_COLORS = {"sglang": "#2ecc71", "vllm": "#3498db"}


def _lazy_import():
    """Import matplotlib lazily so the package works without it installed."""
    try:
        import matplotlib
        matplotlib.use("Agg")   # non-interactive backend — safe on servers
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        import numpy as np
        return plt, mcolors, np
    except ImportError:
        raise ImportError(
            "matplotlib and numpy are required for chart generation.\n"
            "Install with: pip install matplotlib numpy"
        )


def _save(fig, path: str) -> str:
    plt, _, _ = _lazy_import()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- #
# 1 & 2 — TTFT / TPOT heatmaps
# --------------------------------------------------------------------------- #

def ttft_heatmap(
    cells: list[CellResult],
    engine: str,
    workload: str,
    output_dir: str,
    metric: str = "ttft_p50_ms",
) -> str:
    plt, mcolors, np = _lazy_import()

    relevant = [
        c for c in cells
        if c.key.engine == engine and c.key.workload == workload
    ]
    if not relevant:
        return ""

    ctx_list = sorted(set(c.key.context_tokens for c in relevant))
    conc_list = sorted(set(c.key.concurrency for c in relevant))
    by_key = {(c.key.context_tokens, c.key.concurrency): c for c in relevant}

    data = np.zeros((len(ctx_list), len(conc_list)))
    annot = [[""] * len(conc_list) for _ in range(len(ctx_list))]

    for i, ctx in enumerate(ctx_list):
        for j, c in enumerate(conc_list):
            cell = by_key.get((ctx, c))
            if cell:
                val = getattr(cell, metric, 0.0)
                data[i, j] = val
                annot[i][j] = f"{val:.0f}"

    metric_label = {
        "ttft_p50_ms": "TTFT p50 (ms)",
        "tpot_p50_ms": "TPOT p50 (ms)",
        "gpu_util":    "GPU Util (%)",
    }.get(metric, metric)

    fig, ax = plt.subplots(figsize=(max(8, len(conc_list) * 1.2), max(5, len(ctx_list) * 0.8)))
    cmap = "YlOrRd"
    masked = np.ma.masked_where(data == 0, data)
    im = ax.imshow(masked, cmap=cmap, aspect="auto")

    ax.set_xticks(range(len(conc_list)))
    ax.set_xticklabels([f"c={c}" for c in conc_list], fontsize=9)
    ax.set_yticks(range(len(ctx_list)))
    ax.set_yticklabels([f"{ctx:,}" for ctx in ctx_list], fontsize=9)
    ax.set_xlabel("Concurrency", fontsize=11)
    ax.set_ylabel("Context tokens", fontsize=11)
    ax.set_title(f"{metric_label} — {engine} / {workload}", fontsize=13, fontweight="bold")

    for i in range(len(ctx_list)):
        for j in range(len(conc_list)):
            if annot[i][j]:
                val = data[i, j]
                vmax = data.max() if data.max() > 0 else 1
                text_color = "white" if val / vmax > 0.6 else "black"
                ax.text(j, i, annot[i][j], ha="center", va="center",
                        fontsize=8, color=text_color, fontweight="bold")

    plt.colorbar(im, ax=ax, label=metric_label)
    fname = f"heatmap_{metric}_{engine}_{workload}.png"
    return _save(fig, os.path.join(output_dir, "charts", fname))


def gpu_util_heatmap(
    cells: list[CellResult],
    engine: str,
    workload: str,
    output_dir: str,
) -> str:
    plt, mcolors, np = _lazy_import()

    relevant = [
        c for c in cells
        if c.key.engine == engine and c.key.workload == workload and c.gpu.available
    ]
    if not relevant:
        return ""

    ctx_list = sorted(set(c.key.context_tokens for c in relevant))
    conc_list = sorted(set(c.key.concurrency for c in relevant))
    by_key = {(c.key.context_tokens, c.key.concurrency): c for c in relevant}

    data = np.zeros((len(ctx_list), len(conc_list)))
    annot = [[""] * len(conc_list) for _ in range(len(ctx_list))]

    for i, ctx in enumerate(ctx_list):
        for j, c in enumerate(conc_list):
            cell = by_key.get((ctx, c))
            if cell and cell.gpu.available:
                val = cell.gpu.util_mean_pct
                data[i, j] = val
                annot[i][j] = f"{val:.0f}%"

    fig, ax = plt.subplots(figsize=(max(8, len(conc_list) * 1.2), max(5, len(ctx_list) * 0.8)))
    im = ax.imshow(data, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=100)

    ax.set_xticks(range(len(conc_list)))
    ax.set_xticklabels([f"c={c}" for c in conc_list], fontsize=9)
    ax.set_yticks(range(len(ctx_list)))
    ax.set_yticklabels([f"{ctx:,}" for ctx in ctx_list], fontsize=9)
    ax.set_xlabel("Concurrency", fontsize=11)
    ax.set_ylabel("Context tokens", fontsize=11)
    ax.set_title(f"GPU Utilization (%) — {engine} / {workload}", fontsize=13, fontweight="bold")

    for i in range(len(ctx_list)):
        for j in range(len(conc_list)):
            if annot[i][j]:
                ax.text(j, i, annot[i][j], ha="center", va="center",
                        fontsize=8, color="white", fontweight="bold")

    plt.colorbar(im, ax=ax, label="GPU Util %")
    fname = f"heatmap_gpu_util_{engine}_{workload}.png"
    return _save(fig, os.path.join(output_dir, "charts", fname))


# --------------------------------------------------------------------------- #
# 3 — Throughput saturation curve
# --------------------------------------------------------------------------- #

def throughput_curve(
    cells: list[CellResult],
    engine: str,
    workload: str,
    output_dir: str,
) -> str:
    plt, _, np = _lazy_import()

    relevant = [
        c for c in cells
        if c.key.engine == engine and c.key.workload == workload
    ]
    if not relevant:
        return ""

    ctx_list = sorted(set(c.key.context_tokens for c in relevant))
    cmap = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(10, 6))

    for idx, ctx in enumerate(ctx_list):
        ctx_cells = sorted(
            [c for c in relevant if c.key.context_tokens == ctx],
            key=lambda c: c.key.concurrency,
        )
        if not ctx_cells:
            continue
        concs = [c.key.concurrency for c in ctx_cells]
        tps = [c.output_tps_total for c in ctx_cells]
        color = cmap(idx % 10)
        ax.plot(concs, tps, marker="o", linewidth=2, color=color,
                label=f"ctx={ctx:,}", markersize=5)

        # Mark saturation point
        for i in range(1, len(tps)):
            if tps[i - 1] > 0 and (tps[i] - tps[i - 1]) / tps[i - 1] < 0.05:
                ax.axvline(x=concs[i - 1], color=color, linestyle="--", alpha=0.3, linewidth=1)
                break

    ax.set_xlabel("Concurrency", fontsize=11)
    ax.set_ylabel("Throughput (tokens/s)", fontsize=11)
    ax.set_title(f"Throughput vs Concurrency — {engine} / {workload}", fontsize=13, fontweight="bold")
    ax.legend(title="Context length", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xscale("log", base=2)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

    fig.tight_layout()
    fname = f"throughput_curve_{engine}_{workload}.png"
    return _save(fig, os.path.join(output_dir, "charts", fname))


# --------------------------------------------------------------------------- #
# 4 — Latency curve (TTFT p50 + p95)
# --------------------------------------------------------------------------- #

def latency_curve(
    cells: list[CellResult],
    engine: str,
    workload: str,
    output_dir: str,
) -> str:
    plt, _, np = _lazy_import()

    relevant = [
        c for c in cells
        if c.key.engine == engine and c.key.workload == workload
    ]
    if not relevant:
        return ""

    ctx_list = sorted(set(c.key.context_tokens for c in relevant))
    cmap = plt.get_cmap("tab10")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharey=False)
    ax_ttft, ax_tpot = axes

    for idx, ctx in enumerate(ctx_list):
        ctx_cells = sorted(
            [c for c in relevant if c.key.context_tokens == ctx],
            key=lambda c: c.key.concurrency,
        )
        if not ctx_cells:
            continue
        concs = [c.key.concurrency for c in ctx_cells]
        ttft_p50 = [c.ttft_p50_ms for c in ctx_cells]
        ttft_p95 = [c.ttft_p95_ms for c in ctx_cells]
        tpot_p50 = [c.tpot_p50_ms for c in ctx_cells]
        tpot_p95 = [c.tpot_p95_ms for c in ctx_cells]
        color = cmap(idx % 10)

        ax_ttft.plot(concs, ttft_p50, marker="o", linewidth=2, color=color,
                     label=f"ctx={ctx:,} p50", markersize=4)
        ax_ttft.fill_between(concs, ttft_p50, ttft_p95, alpha=0.12, color=color)

        ax_tpot.plot(concs, tpot_p50, marker="s", linewidth=2, color=color,
                     label=f"ctx={ctx:,} p50", markersize=4)
        ax_tpot.fill_between(concs, tpot_p50, tpot_p95, alpha=0.12, color=color)

    for ax, title, ylabel in [
        (ax_ttft, "TTFT (Time to First Token)", "TTFT (ms)"),
        (ax_tpot, "TPOT (Time per Output Token)", "TPOT (ms)"),
    ]:
        ax.set_xlabel("Concurrency", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(f"{title}\n{engine} / {workload}", fontsize=12, fontweight="bold")
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{int(x)}"))

    fig.suptitle(f"Latency Profile — {engine} / {workload}", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    fname = f"latency_curve_{engine}_{workload}.png"
    return _save(fig, os.path.join(output_dir, "charts", fname))


# --------------------------------------------------------------------------- #
# 5 — Engine comparison (SGLang vs vLLM)
# --------------------------------------------------------------------------- #

def engine_comparison_chart(
    cells: list[CellResult],
    output_dir: str,
    context_filter: int | None = None,
    workload: str = "random",
) -> str:
    plt, _, np = _lazy_import()

    engines = sorted(set(c.key.engine for c in cells))
    if len(engines) < 2:
        return ""

    relevant = [c for c in cells if c.key.workload == workload]
    if context_filter:
        relevant = [c for c in relevant if c.key.context_tokens == context_filter]

    conc_list = sorted(set(c.key.concurrency for c in relevant))
    by_key = {(c.key.engine, c.key.context_tokens, c.key.concurrency): c for c in relevant}

    ctx_to_use = context_filter or sorted(set(c.key.context_tokens for c in relevant))[len(set(c.key.context_tokens for c in relevant)) // 2]

    metrics = ["ttft_p50_ms", "tpot_p50_ms", "output_tps_total"]
    metric_labels = ["TTFT p50 (ms)", "TPOT p50 (ms)", "Throughput (tok/s)"]

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    for ax, metric, mlabel in zip(axes, metrics, metric_labels):
        x = np.arange(len(conc_list))
        width = 0.35

        for i, engine in enumerate(engines):
            vals = []
            for conc in conc_list:
                cell = by_key.get((engine, ctx_to_use, conc))
                vals.append(getattr(cell, metric, 0.0) if cell else 0.0)
            offset = (i - 0.5) * width
            bars = ax.bar(x + offset, vals, width,
                          label=engine,
                          color=ENGINE_COLORS.get(engine, "#999"),
                          alpha=0.85, edgecolor="white")

            # Value labels on bars
            for bar, v in zip(bars, vals):
                if v > 0:
                    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                            f"{v:.0f}", ha="center", va="bottom", fontsize=7)

        ax.set_xlabel("Concurrency", fontsize=10)
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(f"{mlabel}\nctx={ctx_to_use:,} / {workload}", fontsize=11, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels([f"c={c}" for c in conc_list], fontsize=8, rotation=30)
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle("SGLang vs vLLM — Head-to-Head", fontsize=14, fontweight="bold")
    fig.tight_layout()
    ctx_tag = f"ctx{ctx_to_use}" if context_filter else "all"
    fname = f"engine_comparison_{ctx_tag}_{workload}.png"
    return _save(fig, os.path.join(output_dir, "charts", fname))


# --------------------------------------------------------------------------- #
# 6 — Bottleneck distribution
# --------------------------------------------------------------------------- #

def bottleneck_distribution(
    cells: list[CellResult],
    output_dir: str,
) -> str:
    plt, _, np = _lazy_import()

    counts: dict[str, int] = defaultdict(int)
    for cell in cells:
        counts[cell.bottleneck] += 1

    if not counts:
        return ""

    labels = sorted(counts.keys(), key=lambda k: -counts[k])
    values = [counts[k] for k in labels]
    colors = [BOTTLENECK_COLORS.get(k, "#bdc3c7") for k in labels]

    fig, (ax_bar, ax_pie) = plt.subplots(1, 2, figsize=(14, 5))

    # Bar chart
    bars = ax_bar.barh(labels, values, color=colors, edgecolor="white", alpha=0.9)
    ax_bar.set_xlabel("Number of cells", fontsize=11)
    ax_bar.set_title("Bottleneck Frequency", fontsize=12, fontweight="bold")
    for bar, v in zip(bars, values):
        ax_bar.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                    str(v), va="center", fontsize=9)
    ax_bar.grid(axis="x", alpha=0.3)

    # Pie chart
    wedges, texts, autotexts = ax_pie.pie(
        values,
        labels=labels,
        colors=colors,
        autopct="%1.0f%%",
        startangle=140,
        pctdistance=0.8,
    )
    for text in texts:
        text.set_fontsize(9)
    for autotext in autotexts:
        autotext.set_fontsize(8)
    ax_pie.set_title("Bottleneck Distribution", fontsize=12, fontweight="bold")

    fig.suptitle("Identified Bottlenecks Across All Cells", fontsize=13, fontweight="bold")
    fig.tight_layout()
    return _save(fig, os.path.join(output_dir, "charts", "bottleneck_distribution.png"))


# --------------------------------------------------------------------------- #
# 7 — Before/After optimization comparison
# --------------------------------------------------------------------------- #

def optimization_delta_chart(
    runs: list[OptimizationRun],
    output_dir: str,
) -> str:
    plt, _, np = _lazy_import()

    if not runs:
        return ""

    metrics = ["ttft_p50_ms", "tpot_p50_ms", "output_tps_total", "e2e_p50_ms"]
    metric_labels = ["TTFT p50 (ms)", "TPOT p50 (ms)", "Throughput (tok/s)", "E2E p50 (ms)"]

    def _mean(cells, attr):
        vals = [getattr(c, attr, 0.0) for c in cells if getattr(c, attr, 0.0) > 0]
        return sum(vals) / len(vals) if vals else 0.0

    run_labels = [r.optimization.name[:20] for r in runs]
    x = np.arange(len(runs))
    width = 0.2

    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()

    for ax_idx, (metric, mlabel) in enumerate(zip(metrics, metric_labels)):
        ax = axes[ax_idx]
        before_vals = [_mean(r.before_cells, metric) for r in runs]
        after_vals = [_mean(r.after_cells, metric) for r in runs]

        bars_b = ax.bar(x - width / 2, before_vals, width,
                        label="Before", color="#e74c3c", alpha=0.8, edgecolor="white")
        bars_a = ax.bar(x + width / 2, after_vals, width,
                        label="After", color="#27ae60", alpha=0.8, edgecolor="white")

        for bar, v in zip(bars_b, before_vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7)
        for bar, v in zip(bars_a, after_vals):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
                        f"{v:.0f}", ha="center", va="bottom", fontsize=7, color="#27ae60")

        ax.set_xticks(x)
        ax.set_xticklabels(run_labels, fontsize=8, rotation=20, ha="right")
        ax.set_ylabel(mlabel, fontsize=10)
        ax.set_title(f"{mlabel} — Before vs After", fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(axis="y", alpha=0.3)

        # Highlight accepted runs
        for i, run in enumerate(runs):
            if run.verdict == "accepted":
                ax.axvspan(i - 0.4, i + 0.4, alpha=0.06, color="#27ae60")

    fig.suptitle("Optimization Loop — Before vs After Each Pass",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return _save(fig, os.path.join(output_dir, "charts", "optimization_delta.png"))


# --------------------------------------------------------------------------- #
# Master function — generate all charts for a sweep
# --------------------------------------------------------------------------- #

def generate_all_charts(
    cells: list[CellResult],
    output_dir: str,
    optimization_runs: list[OptimizationRun] | None = None,
) -> list[str]:
    """
    Generate all charts for a completed sweep and return paths to saved PNGs.
    Skips any chart for which data is missing (returns "" which is filtered out).
    """
    saved: list[str] = []

    engines = sorted(set(c.key.engine for c in cells))
    workloads = sorted(set(c.key.workload for c in cells))

    for engine in engines:
        for workload in workloads:
            saved.append(ttft_heatmap(cells, engine, workload, output_dir, "ttft_p50_ms"))
            saved.append(ttft_heatmap(cells, engine, workload, output_dir, "tpot_p50_ms"))
            saved.append(gpu_util_heatmap(cells, engine, workload, output_dir))
            saved.append(throughput_curve(cells, engine, workload, output_dir))
            saved.append(latency_curve(cells, engine, workload, output_dir))

    if len(engines) >= 2:
        # One comparison chart per workload at the median context length
        for workload in workloads:
            saved.append(engine_comparison_chart(cells, output_dir, workload=workload))

    saved.append(bottleneck_distribution(cells, output_dir))

    if optimization_runs:
        saved.append(optimization_delta_chart(optimization_runs, output_dir))

    return [p for p in saved if p]

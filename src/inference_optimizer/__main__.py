from __future__ import annotations

import argparse
import asyncio
import sys

from .analysis.compare import engine_comparison
from .gpu_requirements import print_requirements_table
from .loop.optimizer import InferenceOptimizer
from .report.charts import generate_all_charts, optimization_delta_chart
from .report.console import (
    print_bottleneck_summary,
    print_engine_comparison,
    print_saturation_analysis,
    print_sweep_table,
)
from .report.storage import load_results, save_results
from .sweep.config import SweepConfig, default_config, long_context_config, peak_config, quick_config
from .sweep.runner import SweepRunner


def _build_config_from_args(args: argparse.Namespace) -> SweepConfig:
    # Preset selection — order matters: long-context > peak > quick > default
    if getattr(args, "long_context", False):
        cfg = long_context_config()
    elif getattr(args, "peak", False):
        cfg = peak_config()
    elif getattr(args, "quick", False):
        cfg = quick_config()
    else:
        cfg = default_config()

    if hasattr(args, "model") and args.model:
        cfg.model = args.model
    if hasattr(args, "engine") and args.engine:
        cfg.engines = args.engine
    if hasattr(args, "context_lengths") and args.context_lengths:
        cfg.context_lengths = args.context_lengths
    if hasattr(args, "concurrencies") and args.concurrencies:
        cfg.concurrencies = args.concurrencies
    if hasattr(args, "output_tokens") and args.output_tokens:
        cfg.output_tokens = args.output_tokens
    if hasattr(args, "n_requests") and args.n_requests:
        cfg.n_requests_per_cell = args.n_requests
    if hasattr(args, "workload") and args.workload:
        cfg.workloads = args.workload
    if hasattr(args, "output_dir") and args.output_dir:
        cfg.output_dir = args.output_dir
    if hasattr(args, "sglang_url") and args.sglang_url:
        cfg.sglang_url = args.sglang_url
    if hasattr(args, "vllm_url") and args.vllm_url:
        cfg.vllm_url = args.vllm_url
    return cfg


async def cmd_sweep(args: argparse.Namespace) -> None:
    cfg = _build_config_from_args(args)

    print(f"\nInference Optimizer — Sweep")
    print(f"  Model:    {cfg.model}")
    print(f"  Engines:  {cfg.engines}")
    print(f"  Contexts: {cfg.context_lengths}")
    print(f"  Concurrencies: {cfg.concurrencies}")
    print(f"  Workloads: {cfg.workloads}")
    print(f"  Requests per cell: {cfg.n_requests_per_cell}")
    total_cells = (
        len(cfg.engines)
        * len(cfg.context_lengths)
        * len(cfg.concurrencies)
        * len(cfg.workloads)
    )
    print(f"  Total cells: {total_cells}")
    print(f"  Output dir: {cfg.output_dir}")

    # GPU requirements table — shows which cells need multi-GPU or FP8 KV
    print_requirements_table(cfg.context_lengths, cfg.concurrencies)
    print()

    runner = SweepRunner(cfg)
    try:
        cells = await runner.run_sweep()
    finally:
        await runner.close()

    # Print summary tables
    for engine in cfg.engines:
        for workload in cfg.workloads:
            print_sweep_table(cells, engine, workload)

    print_bottleneck_summary(cells)
    print_saturation_analysis(cells)

    if len(cfg.engines) > 1:
        comparison = engine_comparison(cells)
        print_engine_comparison(comparison)

    paths = save_results(cells, cfg.output_dir, label="sweep")
    print(f"\nResults saved:")
    print(f"  JSON: {paths['json']}")
    print(f"  CSV:  {paths['csv']}")

    # Generate charts
    print("\nGenerating charts...", end=" ", flush=True)
    try:
        chart_paths = generate_all_charts(cells, cfg.output_dir)
        print(f"done — {len(chart_paths)} charts saved to {cfg.output_dir}/charts/")
        for p in chart_paths:
            print(f"  {p}")
    except ImportError as e:
        print(f"skipped ({e})")


async def cmd_optimize(args: argparse.Namespace) -> None:
    cfg = _build_config_from_args(args)
    max_iterations = getattr(args, "max_iterations", 5)

    print(f"\nInference Optimizer — Optimization Loop")
    print(f"  Model:    {cfg.model}")
    print(f"  Engines:  {cfg.engines}")
    print(f"  Max iterations: {max_iterations}")
    print()

    optimizer = InferenceOptimizer(cfg)
    try:
        await optimizer.run_loop(max_iterations=max_iterations)
    finally:
        await optimizer.runner.close()

    all_cells = [c for run in optimizer.history for c in run.after_cells]
    paths = save_results(all_cells, cfg.output_dir, label="optimized")
    print(f"\nResults saved:")
    print(f"  JSON: {paths['json']}")
    print(f"  CSV:  {paths['csv']}")

    # Generate charts including before/after optimization delta
    print("\nGenerating charts...", end=" ", flush=True)
    try:
        chart_paths = generate_all_charts(
            all_cells,
            cfg.output_dir,
            optimization_runs=optimizer.history,
        )
        print(f"done — {len(chart_paths)} charts saved to {cfg.output_dir}/charts/")
        for p in chart_paths:
            print(f"  {p}")
    except ImportError as e:
        print(f"skipped ({e})")


def cmd_compare(args: argparse.Namespace) -> None:
    print(f"\nLoading baseline: {args.baseline}")
    baseline_cells = load_results(args.baseline)
    print(f"  Loaded {len(baseline_cells)} cells.")

    print(f"Loading optimized: {args.optimized}")
    optimized_cells = load_results(args.optimized)
    print(f"  Loaded {len(optimized_cells)} cells.")

    # Print sweep tables for both
    engines = sorted(set(c.key.engine for c in baseline_cells))
    workloads = sorted(set(c.key.workload for c in baseline_cells))

    print("\n--- BASELINE ---")
    for engine in engines:
        for workload in workloads:
            print_sweep_table(baseline_cells, engine, workload)
    print_bottleneck_summary(baseline_cells)

    print("\n--- OPTIMIZED ---")
    for engine in engines:
        for workload in workloads:
            print_sweep_table(optimized_cells, engine, workload)
    print_bottleneck_summary(optimized_cells)

    # Side-by-side engine comparison (baseline)
    if len(engines) > 1:
        comparison = engine_comparison(baseline_cells)
        print_engine_comparison(comparison)

    print_saturation_analysis(baseline_cells)


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="python -m inference_optimizer",
        description="LLM inference benchmarking, bottleneck classification, and optimization loop.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # --- sweep ---
    sweep_parser = subparsers.add_parser(
        "sweep",
        help="Run 2D sweep across context × concurrency",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        description="""
Sweep presets (pick one, or override with --context-lengths / --concurrencies):

  (default)       ctx=[512..32K]  c=[1..128]  2 workloads   7×8×2 = 112 cells/engine
  --quick         ctx=[512,2K,8K] c=[1,8,32,128]  1 workload     3×4×1 = 12 cells/engine
  --peak          ctx=[512..8K]   c=[1..128]  1 workload    4×7×1 = 28 cells/engine
  --long-context  ctx=[64K,128K]  c=[1,2,4,8] 1 workload    2×4×1 =  8 cells/engine
                  ⚠ long-context requires FP8 KV on 1 GPU; 128K needs 2 GPUs in BF16

GPU requirement table is printed automatically before the sweep starts.
""",
    )
    sweep_parser.add_argument("--model", default="Qwen/Qwen3-8B")
    sweep_parser.add_argument("--engine", nargs="+", dest="engine", default=["sglang", "vllm"])
    sweep_parser.add_argument("--context-lengths", nargs="+", type=int, dest="context_lengths",
                              help="Override context lengths (tokens)")
    sweep_parser.add_argument("--concurrencies", nargs="+", type=int,
                              help="Override concurrency levels (e.g. 1 4 8 32 64 128 256)")
    sweep_parser.add_argument("--output-tokens", type=int, dest="output_tokens")
    sweep_parser.add_argument("--n-requests", type=int, dest="n_requests")
    sweep_parser.add_argument("--workload", nargs="+", default=["random", "shared_prefix"])
    sweep_parser.add_argument("--output-dir", default="./results", dest="output_dir")
    sweep_parser.add_argument("--sglang-url", default="http://localhost:30000", dest="sglang_url")
    sweep_parser.add_argument("--vllm-url", default="http://localhost:8000", dest="vllm_url")
    sweep_parser.add_argument("--quick", action="store_true",
                              help="Quick 3×4×1 validation sweep")
    sweep_parser.add_argument("--peak", action="store_true",
                              help="Peak throughput sweep — ramps c=1→128 on short contexts")
    sweep_parser.add_argument("--long-context", action="store_true", dest="long_context",
                              help="Long-context sweep (64K/128K). Needs FP8 KV or 2 GPUs.")

    # --- optimize ---
    opt_parser = subparsers.add_parser("optimize", help="Run full optimization loop")
    opt_parser.add_argument("--model", default="Qwen/Qwen3-8B")
    opt_parser.add_argument("--engine", nargs="+", dest="engine", default=["sglang"])
    opt_parser.add_argument("--context-lengths", nargs="+", type=int, dest="context_lengths")
    opt_parser.add_argument("--concurrencies", nargs="+", type=int)
    opt_parser.add_argument("--output-tokens", type=int, dest="output_tokens")
    opt_parser.add_argument("--n-requests", type=int, dest="n_requests")
    opt_parser.add_argument("--workload", nargs="+", default=["random"])
    opt_parser.add_argument("--output-dir", default="./results", dest="output_dir")
    opt_parser.add_argument("--sglang-url", default="http://localhost:30000", dest="sglang_url")
    opt_parser.add_argument("--vllm-url", default="http://localhost:8000", dest="vllm_url")
    opt_parser.add_argument("--max-iterations", type=int, default=5, dest="max_iterations")
    opt_parser.add_argument("--quick", action="store_true")
    opt_parser.add_argument("--peak", action="store_true")
    opt_parser.add_argument("--long-context", action="store_true", dest="long_context")

    # --- compare ---
    cmp_parser = subparsers.add_parser("compare", help="Compare two saved result files")
    cmp_parser.add_argument("--baseline", required=True, help="Path to baseline results JSON")
    cmp_parser.add_argument("--optimized", required=True, help="Path to optimized results JSON")

    args = parser.parse_args()

    if args.command == "sweep":
        asyncio.run(cmd_sweep(args))
    elif args.command == "optimize":
        asyncio.run(cmd_optimize(args))
    elif args.command == "compare":
        cmd_compare(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()

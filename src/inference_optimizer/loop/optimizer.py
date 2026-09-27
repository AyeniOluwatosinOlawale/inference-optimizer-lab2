from __future__ import annotations

from dataclasses import dataclass, field
from collections import defaultdict

from ..bottleneck.classifier import classify
from ..optimizations.registry import (
    Optimization,
    generate_server_command,
    get_optimizations,
)
from ..report.console import print_comparison_table, print_sweep_table
from ..sweep.config import SweepConfig
from ..sweep.runner import SweepRunner
from ..types import CellResult


@dataclass
class OptimizationRun:
    optimization: Optimization
    before_cells: list[CellResult]
    after_cells: list[CellResult]
    improvement: dict[str, float]  # metric_name -> pct improvement (positive = better)
    verdict: str                   # "accept" | "reject" | "marginal"
    verdict_reason: str


class InferenceOptimizer:
    def __init__(self, config: SweepConfig) -> None:
        self.config = config
        self.runner = SweepRunner(config)
        self.history: list[OptimizationRun] = []

    async def baseline(self) -> list[CellResult]:
        """Run the full sweep as baseline measurement."""
        print("\n" + "=" * 70)
        print("PHASE 1: BASELINE MEASUREMENT")
        print("=" * 70)
        cells = await self.runner.run_sweep()
        return cells

    def diagnose(self, cells: list[CellResult]) -> dict[str, list[str]]:
        """
        For each cell run bottleneck classifier.
        Return dict: bottleneck_type -> list of cell labels.
        """
        print("\n" + "=" * 70)
        print("PHASE 2: BOTTLENECK DIAGNOSIS")
        print("=" * 70)

        # Build lookup for baseline (c=1) and prev-concurrency cells
        by_key: dict[tuple, CellResult] = {}
        for cell in cells:
            k = (cell.key.engine, cell.key.context_tokens, cell.key.concurrency, cell.key.workload)
            by_key[k] = cell

        diagnosis: dict[str, list[str]] = defaultdict(list)

        for cell in cells:
            k = cell.key
            baseline = by_key.get((k.engine, k.context_tokens, 1, k.workload))
            # Find previous concurrency level
            cfg = self.config
            concs = sorted(cfg.concurrencies)
            idx = concs.index(k.concurrency) if k.concurrency in concs else -1
            prev_conc = concs[idx - 1] if idx > 0 else None
            prev_cell = by_key.get((k.engine, k.context_tokens, prev_conc, k.workload)) if prev_conc else None

            bottleneck, confidence, reason = classify(cell, baseline, prev_cell)
            cell.bottleneck = bottleneck
            cell.bottleneck_confidence = confidence
            cell.bottleneck_reason = reason
            diagnosis[bottleneck].append(k.label())

        print(f"\n{'Bottleneck':<25} {'Count':>6}  {'Confidence':>10}")
        print("-" * 50)
        for bt, labels in sorted(diagnosis.items(), key=lambda x: -len(x[1])):
            print(f"  {bt:<23} {len(labels):>6}  (see cells below)")
            for label in labels[:3]:
                print(f"    - {label}")
            if len(labels) > 3:
                print(f"    ... and {len(labels) - 3} more")

        return dict(diagnosis)

    def recommend(
        self,
        diagnosis: dict[str, list[str]],
        engine: str,
    ) -> list[Optimization]:
        """
        For each detected bottleneck, get top-1 optimization.
        Return deduplicated list sorted by coverage.
        """
        print("\n" + "=" * 70)
        print("PHASE 3: RECOMMENDATIONS")
        print("=" * 70)

        seen_ids: set[str] = set()
        ranked: list[tuple[int, Optimization]] = []

        for bottleneck, labels in sorted(diagnosis.items(), key=lambda x: -len(x[1])):
            if bottleneck in ("unknown", "cache_effective"):
                continue
            opts = get_optimizations(bottleneck, engine)
            if not opts:
                continue
            top = opts[0]
            if top.id not in seen_ids:
                seen_ids.add(top.id)
                ranked.append((len(labels), top))

        ranked.sort(key=lambda x: -x[0])
        recommendations = [opt for _, opt in ranked]

        for i, (coverage, opt) in enumerate(ranked, 1):
            print(f"\n[{i}] {opt.name} (id={opt.id})")
            print(f"     Bottleneck: {opt.bottleneck} — affects {coverage} cells")
            print(f"     Expected: {opt.expected_improvement}")
            print(f"     Applies when: {opt.applies_when}")
            if opt.requires_restart:
                cmd = generate_server_command(
                    engine=engine,
                    model=self.config.model,
                    url_port=int(
                        (self.config.sglang_url if engine == "sglang" else self.config.vllm_url)
                        .split(":")[-1]
                    ),
                    base_flags={},
                    optimizations=[opt],
                )
                print(f"\n     Server command:\n     {cmd}")

        return recommendations

    async def apply_and_rerun(
        self,
        optimization: Optimization,
        affected_cells: list[CellResult],
        engine: str,
    ) -> OptimizationRun:
        """
        1. Print the optimization to apply + new server command
        2. Prompt user to restart server
        3. Re-run only the affected cells
        4. Compare before vs after
        5. Return OptimizationRun with verdict
        """
        print("\n" + "=" * 70)
        print(f"PHASE 4: APPLYING OPTIMIZATION — {optimization.name}")
        print("=" * 70)
        print(f"\nDescription: {optimization.description}")
        print(f"Side effects: {optimization.side_effects}")

        if optimization.requires_restart:
            port = int(
                (self.config.sglang_url if engine == "sglang" else self.config.vllm_url)
                .split(":")[-1]
            )
            cmd = generate_server_command(
                engine=engine,
                model=self.config.model,
                url_port=port,
                base_flags={},
                optimizations=[optimization],
            )
            print(f"\nRestart the {engine} server with:\n\n  {cmd}\n")
            try:
                input("Press Enter when the server is ready to continue...")
            except EOFError:
                print("(non-interactive mode — continuing automatically)")

        print(f"\nRe-running {len(affected_cells)} affected cells...")
        after_cells: list[CellResult] = []
        for cell in affected_cells:
            k = cell.key
            if k.engine != engine:
                after_cells.append(cell)
                continue
            new_cell = await self.runner.run_cell(
                engine=k.engine,
                context_tokens=k.context_tokens,
                concurrency=k.concurrency,
                workload=k.workload,
            )
            after_cells.append(new_cell)

        # Compute improvements
        improvement = self._compute_improvement(affected_cells, after_cells)
        verdict, verdict_reason = self._verdict(improvement)

        run = OptimizationRun(
            optimization=optimization,
            before_cells=affected_cells,
            after_cells=after_cells,
            improvement=improvement,
            verdict=verdict,
            verdict_reason=verdict_reason,
        )
        return run

    def compare(self, run: OptimizationRun) -> None:
        """Print a before/after comparison table for the optimization run."""
        print_comparison_table(run)

    def _compute_improvement(
        self,
        before: list[CellResult],
        after: list[CellResult],
    ) -> dict[str, float]:
        def _mean(cells: list[CellResult], attr: str) -> float:
            vals = [getattr(c, attr) for c in cells if getattr(c, attr, 0) > 0]
            return sum(vals) / len(vals) if vals else 0.0

        metrics = {
            "ttft_p50_ms": True,   # lower is better
            "ttft_p95_ms": True,
            "tpot_p50_ms": True,
            "e2e_p50_ms": True,
            "output_tps_total": False,  # higher is better
            "req_per_sec": False,
            "error_rate": True,
        }
        result: dict[str, float] = {}
        for metric, lower_is_better in metrics.items():
            b = _mean(before, metric)
            a = _mean(after, metric)
            if b == 0:
                result[metric] = 0.0
                continue
            if lower_is_better:
                # positive = improvement (faster)
                result[metric] = (b - a) / b * 100
            else:
                # positive = improvement (higher)
                result[metric] = (a - b) / b * 100
        return result

    def _verdict(
        self, improvement: dict[str, float]
    ) -> tuple[str, str]:
        # Check key metrics
        key_metrics = ["ttft_p50_ms", "tpot_p50_ms", "output_tps_total"]
        improvements = [improvement.get(m, 0.0) for m in key_metrics]
        max_gain = max(improvements) if improvements else 0.0
        mean_gain = sum(improvements) / len(improvements) if improvements else 0.0

        if max_gain > 10:
            return "accept", f"Max improvement {max_gain:.1f}% on key metrics — clear win"
        elif max_gain > 5:
            return "marginal", f"Modest improvement {max_gain:.1f}% — consider workload fit"
        elif max_gain > 0:
            return "reject", f"Improvement {max_gain:.1f}% below 5% threshold — not worth restart overhead"
        else:
            return "reject", f"No improvement detected (max={max_gain:.1f}%) — revert this change"

    async def run_loop(self, max_iterations: int = 5) -> None:
        """Full optimization loop."""
        print("\n" + "=" * 70)
        print("INFERENCE OPTIMIZER — FULL LOOP")
        print(f"Model: {self.config.model}")
        print(f"Engines: {self.config.engines}")
        print(f"Max iterations: {max_iterations}")
        print("=" * 70)

        # Step 1: Baseline
        baseline_cells = await self.baseline()

        for iteration in range(1, max_iterations + 1):
            print(f"\n{'=' * 70}")
            print(f"ITERATION {iteration}/{max_iterations}")
            print("=" * 70)

            # Step 2: Diagnose
            diagnosis = self.diagnose(baseline_cells)

            # Step 3: Pick the primary engine for this iteration
            engine = self.config.engines[0]

            # Step 4: Recommend
            recommendations = self.recommend(diagnosis, engine)
            if not recommendations:
                print("\nNo further optimizations recommended. Loop complete.")
                break

            top_opt = recommendations[0]
            affected_bottleneck = top_opt.bottleneck
            affected_labels = set(diagnosis.get(affected_bottleneck, []))
            affected_cells = [
                c for c in baseline_cells
                if c.key.label() in affected_labels and c.key.engine == engine
            ]

            if not affected_cells:
                print(f"\nNo cells affected by {top_opt.id} for engine {engine}. Skipping.")
                continue

            # Step 5: Apply and re-run
            run = await self.apply_and_rerun(top_opt, affected_cells, engine)
            self.history.append(run)

            # Step 6: Compare
            self.compare(run)

            if run.verdict == "accept":
                print(f"\nVERDICT: ACCEPT — {run.verdict_reason}")
                # Update baseline with the improved cells
                after_by_label = {c.key.label(): c for c in run.after_cells}
                baseline_cells = [
                    after_by_label.get(c.key.label(), c) for c in baseline_cells
                ]
            elif run.verdict == "marginal":
                print(f"\nVERDICT: MARGINAL — {run.verdict_reason}")
                print("Continuing with next optimization...")
            else:
                print(f"\nVERDICT: REJECT — {run.verdict_reason}")
                print("Reverting this optimization. Consider next candidate.")

        print("\n" + "=" * 70)
        print("OPTIMIZATION LOOP COMPLETE")
        print(f"Iterations completed: {len(self.history)}")
        accepted = [r for r in self.history if r.verdict == "accept"]
        print(f"Optimizations accepted: {len(accepted)}")
        for r in accepted:
            print(f"  + {r.optimization.name}: {r.verdict_reason}")
        print("=" * 70)

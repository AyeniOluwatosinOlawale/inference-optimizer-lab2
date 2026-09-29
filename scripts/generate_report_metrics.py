"""
Generate supplementary metrics for report:
- TTFT p50 / p95 / p99
- TPOT p50 / p95 / p99
- Success rate per cell
- Goodput / SLA compliance (configurable thresholds)

Usage:
    python3 scripts/generate_report_metrics.py
    python3 scripts/generate_report_metrics.py --results-dir ./results --output report_metrics.json
"""

from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path


# SLA thresholds — adjust to match your target deployment requirements
SLA = {
    "ttft_ms": 500,    # first token within 500ms
    "e2e_ms": 10_000,  # full response within 10s
    "tpot_ms": 50,     # each token within 50ms
}


def pct(vals: list[float], p: float) -> float:
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = (len(s) - 1) * p / 100
    lo = int(idx)
    hi = min(lo + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (idx - lo)


def analyse_file(path: Path) -> list[dict]:
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        return []

    groups: dict[tuple, list[dict]] = defaultdict(list)
    for r in records:
        key = (r["engine"], r["context_tokens"], r["concurrency"], r["workload"])
        groups[key].append(r)

    rows = []
    for (engine, ctx, c, wl), reqs in sorted(groups.items()):
        total = len(reqs)
        ok = [r for r in reqs if not r["error"]]
        success_rate = len(ok) / total * 100 if total else 0

        ttft  = [r["ttft_ms"]  for r in ok if r["ttft_ms"]  is not None]
        tpot  = [r["tpot_ms"]  for r in ok if r["tpot_ms"]  is not None]
        e2e   = [r["e2e_ms"]   for r in ok if r["e2e_ms"]   is not None]

        # Goodput — % of successful requests that also meet ALL SLA thresholds
        sla_pass = [
            r for r in ok
            if (r["ttft_ms"] or 0) <= SLA["ttft_ms"]
            and (r["e2e_ms"]  or 0) <= SLA["e2e_ms"]
            and (r["tpot_ms"] or 0) <= SLA["tpot_ms"]
        ]
        goodput_pct = len(sla_pass) / total * 100 if total else 0

        rows.append({
            "engine":        engine,
            "context_tokens": ctx,
            "concurrency":   c,
            "workload":      wl,
            "total_requests": total,
            "success_rate_pct": round(success_rate, 1),
            "goodput_pct":   round(goodput_pct, 1),
            "ttft_p50_ms":   round(pct(ttft, 50), 1),
            "ttft_p95_ms":   round(pct(ttft, 95), 1),
            "ttft_p99_ms":   round(pct(ttft, 99), 1),
            "tpot_p50_ms":   round(pct(tpot, 50), 1),
            "tpot_p95_ms":   round(pct(tpot, 95), 1),
            "tpot_p99_ms":   round(pct(tpot, 99), 1),
            "e2e_p50_ms":    round(pct(e2e,  50), 1),
            "e2e_p95_ms":    round(pct(e2e,  95), 1),
            "e2e_p99_ms":    round(pct(e2e,  99), 1),
            "sla_ttft_ms":   SLA["ttft_ms"],
            "sla_e2e_ms":    SLA["e2e_ms"],
            "sla_tpot_ms":   SLA["tpot_ms"],
        })
    return rows


def print_table(rows: list[dict], label: str) -> None:
    print(f"\n{'='*90}")
    print(f"  {label}")
    print(f"{'='*90}")
    print(f"{'engine':>8} {'ctx':>7} {'c':>4} {'workload':>14} "
          f"{'ok%':>5} {'goodput%':>9} "
          f"{'TTFT p50':>9} {'p95':>7} {'p99':>7} "
          f"{'TPOT p50':>9} {'p95':>7} {'p99':>7}")
    print("-" * 90)
    for r in rows:
        print(
            f"{r['engine']:>8} {r['context_tokens']:>7} {r['concurrency']:>4} "
            f"{r['workload']:>14} "
            f"{r['success_rate_pct']:>5.1f} {r['goodput_pct']:>9.1f} "
            f"{r['ttft_p50_ms']:>9.1f} {r['ttft_p95_ms']:>7.1f} {r['ttft_p99_ms']:>7.1f} "
            f"{r['tpot_p50_ms']:>9.1f} {r['tpot_p95_ms']:>7.1f} {r['tpot_p99_ms']:>7.1f}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default="./results")
    parser.add_argument("--output", default="report_metrics.json")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    all_rows: dict[str, list[dict]] = {}

    # Process each sweep folder
    for folder in sorted(results_dir.iterdir()):
        if not folder.is_dir() or folder.name == "server_logs":
            continue
        jsons = sorted(folder.glob("*.json"))
        if not jsons:
            continue
        # Use the most recent JSON in each folder
        latest = jsons[-1]
        rows = analyse_file(latest)
        if rows:
            all_rows[folder.name] = rows
            print_table(rows, f"{folder.name}  ({latest.name})")

    # Print SLA thresholds used
    print(f"\n{'='*90}")
    print("  SLA Thresholds Applied")
    print(f"{'='*90}")
    print(f"  TTFT   ≤ {SLA['ttft_ms']} ms")
    print(f"  E2E    ≤ {SLA['e2e_ms']} ms")
    print(f"  TPOT   ≤ {SLA['tpot_ms']} ms")
    print(f"  Goodput = % of ALL requests (including errors) that meet every threshold above")

    # Save JSON
    out_path = Path(args.output)
    out_path.write_text(json.dumps(all_rows, indent=2))
    print(f"\n✓ Full metrics saved to {out_path}")


if __name__ == "__main__":
    main()

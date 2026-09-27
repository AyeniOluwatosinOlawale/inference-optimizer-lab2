from __future__ import annotations

import asyncio
import time

from ..bottleneck.classifier import classify
from ..collectors.gpu import GPUMonitor
from ..collectors.server import ServerPoller
from ..engines.base import EngineClient
from ..engines.sglang import SGLangClient
from ..engines.vllm import VLLMClient
from ..gpu_requirements import check_cell
from ..types import (
    CellKey,
    CellResult,
    GPUSummary,
    RequestRecord,
    ServerSnapshot,
    aggregate_requests,
)
from ..workload.generator import WorkloadGenerator
from .config import SweepConfig


class SweepRunner:
    def __init__(self, config: SweepConfig) -> None:
        self.config = config
        self._gen = WorkloadGenerator()
        self._gpu = GPUMonitor()
        self._clients: dict[str, EngineClient] = {}
        self._pollers: dict[str, ServerPoller] = {}

        if "sglang" in config.engines:
            self._clients["sglang"] = SGLangClient(config.sglang_url, config.model)
            self._pollers["sglang"] = ServerPoller("sglang", config.sglang_url)
        if "vllm" in config.engines:
            self._clients["vllm"] = VLLMClient(config.vllm_url, config.model)
            self._pollers["vllm"] = ServerPoller("vllm", config.vllm_url)

    async def run_cell(
        self,
        engine: str,
        context_tokens: int,
        concurrency: int,
        workload: str,
    ) -> CellResult:
        """Run one cell of the sweep matrix."""
        cfg = self.config
        client = self._clients[engine]
        poller = self._pollers[engine]
        n = cfg.n_requests_per_cell

        prompts = self._gen.make_batch(
            n=n,
            context_tokens=context_tokens,
            workload=workload,
            shared_prefix_tokens=cfg.shared_prefix_tokens,
        )

        # Pre-snapshot
        server_pre = await poller.poll()

        # Start GPU monitor
        await self._gpu.start()

        semaphore = asyncio.Semaphore(concurrency)
        wall_start = time.perf_counter()

        async def bounded_request(prompt: str) -> RequestRecord:
            t_submitted = time.perf_counter()
            async with semaphore:
                t_started = time.perf_counter()
                queue_ms = (t_started - t_submitted) * 1000
                record = await client.stream_request(
                    prompt=prompt,
                    max_tokens=cfg.output_tokens,
                    temperature=cfg.temperature,
                    context_tokens=context_tokens,
                    concurrency=concurrency,
                    workload=workload,
                )
                record.queue_ms = queue_ms
                return record

        tasks = [bounded_request(p) for p in prompts]
        requests: list[RequestRecord] = await asyncio.gather(*tasks)

        wall_seconds = time.perf_counter() - wall_start

        # Stop GPU monitor
        gpu_summary = await self._gpu.stop()

        # Post-snapshot
        server_post = await poller.poll()

        key = CellKey(
            engine=engine,
            context_tokens=context_tokens,
            concurrency=concurrency,
            workload=workload,
        )

        # Placeholder cell (bottleneck TBD)
        cell = aggregate_requests(
            key=key,
            requests=requests,
            gpu=gpu_summary,
            server_pre=server_pre,
            server_post=server_post,
            bottleneck="unknown",
            bottleneck_confidence="low",
            bottleneck_reason="classification pending",
            wall_seconds=wall_seconds,
        )

        # Classify bottleneck
        bottleneck, confidence, reason = classify(cell)
        cell.bottleneck = bottleneck
        cell.bottleneck_confidence = confidence
        cell.bottleneck_reason = reason

        return cell

    async def run_sweep(self) -> list[CellResult]:
        """Iterate all combinations and return all CellResult objects."""
        cells: list[CellResult] = []
        cfg = self.config
        total = (
            len(cfg.engines)
            * len(cfg.context_lengths)
            * len(cfg.concurrencies)
            * len(cfg.workloads)
        )
        done = 0

        for engine in cfg.engines:
            for workload in cfg.workloads:
                for context_tokens in cfg.context_lengths:
                    for concurrency in cfg.concurrencies:
                        done += 1
                        req = check_cell(context_tokens, concurrency)

                        # Print GPU warning inline if this cell needs attention
                        gpu_note = ""
                        if not req.safe_on_single_gpu:
                            if req.min_gpus_fp8 == 1:
                                gpu_note = " [⚠ fp8-kv needed]"
                            else:
                                gpu_note = f" [⚠⚠ {req.min_gpus_fp8}GPU(fp8)]"
                        elif req.kv_gb_bf16 > 20:
                            gpu_note = f" [{req.kv_gb_bf16:.0f}GB-kv]"

                        print(
                            f"[{done}/{total}] {engine} ctx={context_tokens} "
                            f"c={concurrency} {workload}{gpu_note}...",
                            end=" ",
                            flush=True,
                        )
                        cell = await self.run_cell(engine, context_tokens, concurrency, workload)
                        cells.append(cell)
                        print(
                            f"TTFT p50={cell.ttft_p50_ms:.0f}ms "
                            f"TPOT={cell.tpot_p50_ms:.1f}ms "
                            f"TPS={cell.output_tps_total:.0f} "
                            f"[{cell.bottleneck}]"
                        )

        return cells

    async def close(self) -> None:
        for client in self._clients.values():
            await client.close()
        for poller in self._pollers.values():
            await poller.close()

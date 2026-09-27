from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass

from ..types import GPUSummary


@dataclass
class GPUPoll:
    timestamp_s: float
    util_pct: float
    memory_used_gb: float
    power_watts: float
    temperature_c: float
    sm_clock_mhz: float
    mem_clock_mhz: float


class GPUMonitor:
    # H100 NVL has 6144-bit HBM bus
    _HBM_BUS_WIDTH_BITS = 6144

    def __init__(self, gpu_index: int = 0, poll_interval_s: float = 0.5) -> None:
        self._gpu_index = gpu_index
        self._poll_interval_s = poll_interval_s
        self._polls: list[GPUPoll] = []
        self._task: asyncio.Task | None = None
        self._running = False
        self._handle = None  # pynvml device handle

    def _try_init_nvml(self) -> bool:
        """Attempt to initialise pynvml. Returns True if successful."""
        try:
            import pynvml  # type: ignore[import]
            pynvml.nvmlInit()
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(self._gpu_index)
            return True
        except Exception:
            return False

    def _poll_once(self) -> GPUPoll | None:
        """Synchronous poll via pynvml. Returns None if unavailable."""
        if self._handle is None:
            return None
        try:
            import pynvml  # type: ignore[import]
            util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
            mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
            power_mw = pynvml.nvmlDeviceGetPowerUsage(self._handle)
            temp = pynvml.nvmlDeviceGetTemperature(
                self._handle, pynvml.NVML_TEMPERATURE_GPU
            )
            sm_clock = pynvml.nvmlDeviceGetClockInfo(
                self._handle, pynvml.NVML_CLOCK_SM
            )
            mem_clock = pynvml.nvmlDeviceGetClockInfo(
                self._handle, pynvml.NVML_CLOCK_MEM
            )
            return GPUPoll(
                timestamp_s=time.perf_counter(),
                util_pct=float(util.gpu),
                memory_used_gb=mem_info.used / (1024 ** 3),
                power_watts=power_mw / 1000.0,
                temperature_c=float(temp),
                sm_clock_mhz=float(sm_clock),
                mem_clock_mhz=float(mem_clock),
            )
        except Exception:
            return None

    async def _poll_loop(self) -> None:
        while self._running:
            poll = self._poll_once()
            if poll is not None:
                self._polls.append(poll)
            await asyncio.sleep(self._poll_interval_s)

    async def start(self) -> None:
        """Start background polling task."""
        self._polls = []
        self._running = True
        available = self._try_init_nvml()
        if available:
            self._task = asyncio.create_task(self._poll_loop())

    async def stop(self) -> GPUSummary:
        """Stop polling and return summary of all polls."""
        self._running = False
        if self._task is not None:
            try:
                await asyncio.wait_for(self._task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError):
                self._task.cancel()
            self._task = None

        # Shutdown pynvml if it was initialised
        if self._handle is not None:
            try:
                import pynvml  # type: ignore[import]
                pynvml.nvmlShutdown()
            except Exception:
                pass
            self._handle = None

        polls = self._polls
        if not polls:
            return GPUSummary(available=False)

        utils = [p.util_pct for p in polls]
        mems = [p.memory_used_gb for p in polls]
        powers = [p.power_watts for p in polls]
        temps = [p.temperature_c for p in polls]
        sm_clocks = [p.sm_clock_mhz for p in polls]
        mem_clocks = [p.mem_clock_mhz for p in polls]

        mean_mem_clock = sum(mem_clocks) / len(mem_clocks)
        # DDR: x2; /8 bits->bytes; /1000 MHz->GHz
        hbm_bw_gbps = mean_mem_clock * 2 * self._HBM_BUS_WIDTH_BITS / 8 / 1000

        return GPUSummary(
            util_mean_pct=sum(utils) / len(utils),
            util_peak_pct=max(utils),
            memory_used_mean_gb=sum(mems) / len(mems),
            memory_used_peak_gb=max(mems),
            power_mean_watts=sum(powers) / len(powers),
            temperature_mean_c=sum(temps) / len(temps),
            hbm_bw_mean_gbps=hbm_bw_gbps,
            sm_clock_mean_mhz=sum(sm_clocks) / len(sm_clocks),
            available=True,
        )

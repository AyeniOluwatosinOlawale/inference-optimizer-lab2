from __future__ import annotations

import re

import httpx

from ..types import ServerSnapshot


class ServerPoller:
    def __init__(self, engine: str, url: str) -> None:
        self._engine = engine.lower()
        self._url = url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(5.0))
        # Track previous token totals for rate computation (vLLM only)
        self._prev_prompt_tokens: float = 0.0
        self._prev_gen_tokens: float = 0.0

    async def poll(self) -> ServerSnapshot:
        """Single poll. Non-blocking, returns empty snapshot on failure."""
        try:
            if self._engine == "vllm":
                return await self._poll_vllm()
            elif self._engine == "sglang":
                return await self._poll_sglang()
        except Exception:
            pass
        return ServerSnapshot()

    async def _poll_vllm(self) -> ServerSnapshot:
        resp = await self._client.get(f"{self._url}/metrics")
        if resp.status_code != 200:
            return ServerSnapshot()
        text = resp.text

        def _parse_metric(name: str) -> float:
            # Prometheus text format: metric_name{...} value
            pattern = rf'^{re.escape(name)}(?:\{{[^}}]*\}})?\s+([\d.e+\-]+)'
            for line in text.splitlines():
                m = re.match(pattern, line)
                if m:
                    try:
                        return float(m.group(1))
                    except ValueError:
                        pass
            return 0.0

        kv_util = _parse_metric("vllm:gpu_cache_usage_perc")
        num_running = int(_parse_metric("vllm:num_requests_running"))
        num_queued = int(_parse_metric("vllm:num_requests_waiting"))

        # Compute token rates from cumulative counters
        prompt_total = _parse_metric("vllm:prompt_tokens_total")
        gen_total = _parse_metric("vllm:generation_tokens_total")
        prefill_tps = max(0.0, prompt_total - self._prev_prompt_tokens)
        decode_tps = max(0.0, gen_total - self._prev_gen_tokens)
        self._prev_prompt_tokens = prompt_total
        self._prev_gen_tokens = gen_total

        return ServerSnapshot(
            kv_cache_hit_rate=0.0,
            kv_cache_utilization=kv_util / 100.0 if kv_util > 1 else kv_util,
            num_running=num_running,
            num_queued=num_queued,
            decode_tps=decode_tps,
            prefill_tps=prefill_tps,
        )

    async def _poll_sglang(self) -> ServerSnapshot:
        resp = await self._client.get(f"{self._url}/server_info")
        if resp.status_code != 200:
            return ServerSnapshot()
        data = resp.json()

        hit_rate = (
            data.get("cache_hit_rate")
            or data.get("token_hit_rate")
            or 0.0
        )
        num_running = data.get("num_running_reqs", 0)
        num_queued = data.get("num_waiting_reqs", 0)

        return ServerSnapshot(
            kv_cache_hit_rate=float(hit_rate),
            kv_cache_utilization=0.0,
            num_running=int(num_running),
            num_queued=int(num_queued),
            decode_tps=0.0,
            prefill_tps=0.0,
        )

    async def close(self) -> None:
        await self._client.aclose()

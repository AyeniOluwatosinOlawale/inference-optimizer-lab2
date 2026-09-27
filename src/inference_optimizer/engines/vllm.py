from __future__ import annotations

import json
import time
import uuid

import httpx

from ..types import RequestRecord
from .base import EngineClient


class VLLMClient(EngineClient):
    engine_name = "vllm"

    def __init__(self, base_url: str, model: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._client = httpx.AsyncClient(timeout=httpx.Timeout(120.0))

    async def stream_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        context_tokens: int = 0,
        concurrency: int = 1,
        workload: str = "random",
    ) -> RequestRecord:
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens,
            "temperature": temperature,
            "stream": True,
        }

        t_start = time.perf_counter()
        t_first_token: float | None = None
        chunk_times: list[float] = []
        output_chars = 0
        error_msg: str | None = None

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/v1/chat/completions",
                json=payload,
            ) as resp:
                resp.raise_for_status()
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    data_str = line[5:].strip()
                    if data_str == "[DONE]":
                        break
                    try:
                        chunk = json.loads(data_str)
                    except json.JSONDecodeError:
                        continue
                    content = ""
                    if chunk.get("choices"):
                        delta = chunk["choices"][0].get("delta", {})
                        content = delta.get("content", "") or ""
                    if content:
                        now = time.perf_counter()
                        if t_first_token is None:
                            t_first_token = now
                        chunk_times.append(now)
                        output_chars += len(content)
        except Exception as exc:
            error_msg = str(exc)

        t_end = time.perf_counter()
        ttft_ms = (t_first_token - t_start) * 1000 if t_first_token else (t_end - t_start) * 1000
        e2e_ms = (t_end - t_start) * 1000
        output_tokens = max(1, output_chars // 4)
        tpot_ms = (e2e_ms - ttft_ms) / max(output_tokens - 1, 1)

        itl_list = [
            (chunk_times[i + 1] - chunk_times[i]) * 1000
            for i in range(len(chunk_times) - 1)
        ]
        itl_mean = sum(itl_list) / len(itl_list) if itl_list else tpot_ms

        return RequestRecord(
            request_id=str(uuid.uuid4()),
            engine=self.engine_name,
            model=self.model,
            context_tokens=context_tokens,
            output_tokens_requested=max_tokens,
            output_tokens_actual=output_tokens if not error_msg else 0,
            concurrency=concurrency,
            workload=workload,
            ttft_ms=ttft_ms,
            tpot_ms=tpot_ms,
            itl_ms_list=itl_list,
            itl_mean_ms=itl_mean,
            e2e_ms=e2e_ms,
            queue_ms=0.0,
            output_tps=output_tokens / (e2e_ms / 1000) if e2e_ms > 0 and not error_msg else 0.0,
            error=error_msg,
        )

    async def health_check(self) -> bool:
        try:
            resp = await self._client.get(f"{self.base_url}/health", timeout=5.0)
            return resp.status_code == 200
        except Exception:
            return False

    async def close(self) -> None:
        await self._client.aclose()

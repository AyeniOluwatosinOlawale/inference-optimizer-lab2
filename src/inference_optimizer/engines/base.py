from __future__ import annotations

from abc import ABC, abstractmethod

from ..types import RequestRecord


class EngineClient(ABC):
    engine_name: str

    @abstractmethod
    async def stream_request(
        self,
        prompt: str,
        max_tokens: int,
        temperature: float = 0.0,
        context_tokens: int = 0,
        concurrency: int = 1,
        workload: str = "random",
    ) -> RequestRecord:
        """Stream a single request and return a RequestRecord with timing metrics."""
        ...

    @abstractmethod
    async def health_check(self) -> bool:
        """Check if the engine is reachable and healthy."""
        ...

    async def close(self) -> None:
        """Close the underlying HTTP client if applicable."""

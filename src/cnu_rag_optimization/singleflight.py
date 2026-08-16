"""Coalesce concurrent duplicate requests without persistent result caching."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Generic, Hashable, TypeVar


K = TypeVar("K", bound=Hashable)
T = TypeVar("T")


@dataclass(frozen=True)
class SingleFlightResult(Generic[T]):
    value: T
    shared: bool


class AsyncSingleFlight(Generic[K, T]):
    """Share one in-flight task for identical keys; retain no completed answer."""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._tasks: dict[K, asyncio.Task[T]] = {}

    async def do(
        self,
        key: K,
        factory: Callable[[], Awaitable[T]],
    ) -> SingleFlightResult[T]:
        async with self._lock:
            task = self._tasks.get(key)
            shared = task is not None
            if task is None:
                task = asyncio.create_task(factory())
                self._tasks[key] = task
        try:
            return SingleFlightResult(value=await asyncio.shield(task), shared=shared)
        finally:
            if task.done():
                async with self._lock:
                    if self._tasks.get(key) is task:
                        self._tasks.pop(key, None)

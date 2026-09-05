"""Bounded application-side hedging for slow first-token streams."""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator, Callable


StreamFactory = Callable[[], AsyncIterator[str]]


class HedgeLimiter:
    """Process-local limit preventing duplicate-request amplification."""

    def __init__(self, max_concurrent: int = 1) -> None:
        self._semaphore = asyncio.Semaphore(max(1, int(max_concurrent)))

    async def try_acquire(self) -> bool:
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=0.001)
            return True
        except TimeoutError:
            return False

    def release(self) -> None:
        self._semaphore.release()


async def _close(stream: AsyncIterator[str]) -> None:
    close = getattr(stream, "aclose", None)
    if close is not None:
        try:
            await close()
        except (RuntimeError, asyncio.CancelledError):
            pass


async def stream_with_tail_hedge(
    factory: StreamFactory,
    *,
    delay_seconds: float = 12.0,
    limiter: HedgeLimiter,
) -> AsyncIterator[str]:
    """After delayed first token, race one duplicate and cancel loser.

    Experimental ablation. It changes no model-server setting, but consumes an
    extra inference request when fired. Measure tail benefit before enabling.
    """
    primary = factory()
    primary_task = asyncio.create_task(anext(primary))
    done, _ = await asyncio.wait({primary_task}, timeout=max(0.1, delay_seconds))
    if done:
        try:
            first = primary_task.result()
        except StopAsyncIteration:
            return
        yield first
        async for item in primary:
            yield item
        return

    if not await limiter.try_acquire():
        try:
            first = await primary_task
        except StopAsyncIteration:
            return
        yield first
        async for item in primary:
            yield item
        return

    hedge = factory()
    hedge_task = asyncio.create_task(anext(hedge))
    winner = primary
    loser = hedge
    try:
        completed, _ = await asyncio.wait(
            {primary_task, hedge_task}, return_when=asyncio.FIRST_COMPLETED
        )
        winner_task = next(iter(completed))
        if winner_task is hedge_task:
            winner, loser = hedge, primary
        loser_task = hedge_task if winner_task is primary_task else primary_task
        first = winner_task.result()
        loser_task.cancel()
        await asyncio.gather(loser_task, return_exceptions=True)
        await _close(loser)
        yield first
        async for item in winner:
            yield item
    finally:
        for task in (primary_task, hedge_task):
            if not task.done():
                task.cancel()
        await asyncio.gather(primary_task, hedge_task, return_exceptions=True)
        await _close(loser)
        limiter.release()

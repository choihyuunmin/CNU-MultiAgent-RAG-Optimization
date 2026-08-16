"""Parallelize independent enrichment operations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar


T = TypeVar("T")
U = TypeVar("U")


async def parallel_enrich(
    primary_enrichment: Awaitable[T],
    secondary_enrichment: Awaitable[U],
) -> tuple[T, U]:
    """Run two independent enrichment operations concurrently."""
    primary, secondary = await asyncio.gather(
        primary_enrichment,
        secondary_enrichment,
    )
    return primary, secondary

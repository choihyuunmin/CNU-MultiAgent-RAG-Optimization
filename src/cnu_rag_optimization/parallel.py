"""Parallelize independent enrichment operations."""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TypeVar


T = TypeVar("T")
U = TypeVar("U")


async def parallel_enrich(
    translation_enrichment: Awaitable[T],
    origin_enrichment: Awaitable[U],
) -> tuple[T, U]:
    """Run independent translation and origin enrichments concurrently."""
    translation, origin = await asyncio.gather(
        translation_enrichment,
        origin_enrichment,
    )
    return translation, origin

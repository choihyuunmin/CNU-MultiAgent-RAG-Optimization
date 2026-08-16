"""Accuracy-first vLLM server profiles for controlled A/B deployment."""

from __future__ import annotations

import json
from dataclasses import dataclass


@dataclass(frozen=True)
class VLLMProfile:
    name: str
    description: str
    server_args: tuple[str, ...]
    exact_output_gate: bool = True


def _speculative_config(tokens: int) -> str:
    return json.dumps(
        {
            "method": "ngram",
            "num_speculative_tokens": tokens,
            "prompt_lookup_max": 4,
        },
        separators=(",", ":"),
    )


VLLM_PROFILES: tuple[VLLMProfile, ...] = (
    VLLMProfile(
        name="V0_CONTROL",
        description="Current server arguments.",
        server_args=(),
    ),
    VLLMProfile(
        name="V1_PREFIX_CACHE",
        description="Reuse KV blocks for identical prompt prefixes.",
        server_args=("--enable-prefix-caching",),
    ),
    VLLMProfile(
        name="V2_CHUNKED_8192",
        description="Prefix cache plus chunked-prefill token budget 8192.",
        server_args=(
            "--enable-prefix-caching",
            "--max-num-batched-tokens",
            "8192",
        ),
    ),
    VLLMProfile(
        name="V3_NGRAM_3",
        description="Prefix cache plus target-verified three-token n-gram decoding.",
        server_args=(
            "--enable-prefix-caching",
            "--speculative-config",
            _speculative_config(3),
        ),
    ),
    VLLMProfile(
        name="V4_NGRAM_5",
        description="Prefix cache plus target-verified five-token n-gram decoding.",
        server_args=(
            "--enable-prefix-caching",
            "--speculative-config",
            _speculative_config(5),
        ),
    ),
)

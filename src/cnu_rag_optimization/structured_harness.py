"""Lossless JSON input compaction and compact structured decoding contracts.

These preserve JSON values, not model output equivalence. Live paired regression
is required: even formatting changes can change a model's selected evidence.
"""
from __future__ import annotations

from copy import deepcopy
import json
import re


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("non-finite JSON number")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def compact_json(text: str) -> str:
    """Keep all fields, values and array order; unknown/malformed input is opaque."""
    try:
        value = strict_json(text)
        if not isinstance(value, (dict, list)):
            return text
        packed = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
        if strict_json(packed) != value:
            return text
        return packed if len(packed) < len(text) else text
    except (ValueError, TypeError, RecursionError):
        return text


def compact_selection_messages(messages):
    """Compact only the known search data envelope, preserving every other byte."""
    result = deepcopy(messages)
    for message in result:
        content = message.get("content")
        if not isinstance(content, str):
            continue
        start_marker = "# 법령 검색 결과\n"
        end_marker = "\n</search_results>"
        if content.count(start_marker) != 1 or content.count(end_marker) != 1:
            continue
        start = content.index(start_marker) + len(start_marker)
        end = content.find(end_marker, start)
        if end < start:
            continue
        message["content"] = content[:start] + compact_json(content[start:end]) + content[end:]
    return result


def encode_selection_ids(messages, *, min_candidates=16):
    """Use request-local short handles for IDs; keep every evidence text intact.

    The original IDs are restored before the application's selection parser.
    Unknown layouts, duplicate IDs, or IDs explicitly cited by the user bypass
    encoding. No mappings or model responses survive the request.
    """
    result = deepcopy(messages)
    for message in result:
        text = message.get("content")
        if not isinstance(text, str):
            continue
        marker, end_marker = "# 법령 검색 결과\n", "\n</search_results>"
        if text.count(marker) != 1 or text.count(end_marker) != 1:
            continue
        start = text.index(marker) + len(marker)
        end = text.find(end_marker, start)
        if end < start:
            continue
        try:
            data = strict_json(text[start:end])
            laws = data.get("laws")
            if not isinstance(laws, list) or not laws or len(laws) < min_candidates:
                continue
            ids = [law["id"] for law in laws]
            if (not all(isinstance(i, str) and i for i in ids)
                    or len(set(ids)) != len(ids)
                    or any(i in text[end + len(end_marker):] for i in ids)):
                continue
            mapping = {str(n): value for n, value in enumerate(ids)}
            inverse = {value: short for short, value in mapping.items()}
            for n, law in enumerate(laws):
                law["id"] = str(n)
            # Preserve the original whitespace/layout: minification alone can
            # change relevance decisions even when every JSON value is equal.
            def replace(match):
                value = strict_json(match[2])
                return match[1] + json.dumps(inverse[value]) if value in inverse else match[0]
            packed = re.sub(r'(?<!\\)("id"\s*:\s*)("(?:[^"\\]|\\.)*")',
                            replace, text[start:end])
            # Reject ambiguous nested/top-level ID replacements or unknown layout.
            if strict_json(packed) != data:
                continue
            message["content"] = text[:start] + packed + text[end:]
            return result, mapping
        except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
            continue
    return deepcopy(messages), {}


def decode_selection_ids(text, mapping):
    if not mapping:
        return text
    data = strict_json(text)
    ids = data.get("selected_ids")
    if not isinstance(ids, list) or any(not isinstance(i, str) or i not in mapping for i in ids):
        raise ValueError("selection returned an unknown request-local ID")
    data["selected_ids"] = [mapping[i] for i in ids]
    return json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def compact_structured_payload(payload):
    """vLLM >=0.12 structured outputs: same schema, no formatting whitespace.

    Do not use for tool calls, streams, non-strict schemas or existing grammar
    settings. Whitespace *inside JSON string values* remains unrestricted.
    """
    result = deepcopy(payload)
    fmt = result.get("response_format") or {}
    spec = fmt.get("json_schema") or {}
    if (fmt.get("type") != "json_schema" or spec.get("strict") is not True
            or not isinstance(spec.get("schema"), dict)
            or result.get("tools") or result.get("stream")):
        return result
    extra = dict(result.get("extra_body") or {})
    if "structured_outputs" in extra:
        return result
    result.pop("response_format")
    extra["structured_outputs"] = {"json": spec["schema"], "disable_any_whitespace": True}
    result["extra_body"] = extra
    return result

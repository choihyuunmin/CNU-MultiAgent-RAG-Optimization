from copy import deepcopy
import json
import pytest

from cnu_rag_optimization.structured_harness import (
    compact_json, compact_selection_messages, compact_structured_payload,
    encode_selection_ids, decode_selection_ids,
)


def test_compaction_preserves_all_evidence_and_unicode_string_whitespace():
    value = {"laws": [{"id": "17_2", "content": '조문  내용\n "인용"',
                        "unknown": [None, False, 0, {"emoji": "⚖"}]}]}
    raw = json.dumps(value, ensure_ascii=True, indent=4)
    assert json.loads(compact_json(raw)) == value
    assert len(compact_json(raw)) < len(raw)


def test_opaque_and_ambiguous_inputs_are_unchanged():
    for raw in ['{"id":1,"id":2}', '{"score":NaN}', '{"a":Infinity}',
                '{"laws":[ …', 'just text', '"string"']:
        assert compact_json(raw) == raw


def test_only_known_data_envelope_is_changed_without_mutation():
    raw = json.dumps({"laws": [{"id": "1", "content": "body\nspace  space"}]}, indent=2)
    prefix = "<search_results>\n# 법령 검색 결과\n"
    suffix = "\n</search_results>\n\n# 사용자 질문\n질문  원문"
    messages = [{"role": "system", "content": "unchanged"},
                {"role": "user", "content": prefix + raw + suffix}]
    before = deepcopy(messages)
    changed = compact_selection_messages(messages)
    assert messages == before
    assert changed[0] == messages[0]
    assert changed[1]["content"] == prefix + compact_json(raw) + suffix


def test_grammar_preserves_schema_and_generation_options():
    schema = {"type": "object", "properties": {"text": {"type": "string"}},
              "required": ["text"], "additionalProperties": False}
    payload = {"model": "same", "messages": [], "temperature": 0, "seed": 3,
               "max_tokens": 1000, "response_format": {"type": "json_schema",
               "json_schema": {"name": "test", "strict": True, "schema": schema}},
               "extra_body": {"chat_template_kwargs": {"enable_thinking": False}}}
    before = deepcopy(payload)
    result = compact_structured_payload(payload)
    assert payload == before
    assert result["extra_body"]["structured_outputs"] == {
        "json": schema, "disable_any_whitespace": True}
    assert "response_format" not in result
    for key in ("model", "messages", "temperature", "seed", "max_tokens"):
        assert result[key] == payload[key]
    for change in ({"stream": True}, {"tools": [{"type": "function"}]},
                   {"response_format": {"type": "json_object"}},
                   {"extra_body": {"structured_outputs": {"json": schema}}}):
        skipped = dict(payload, **change)
        assert compact_structured_payload(skipped) == skipped


def test_short_ids_restore_original_ids_and_keep_evidence_values():
    laws = [{"id": "109823_129", "law_id": "109823", "content": "본문  그대로"},
            {"id": "209823_130", "country": "나라", "unknown": {"x": [1, None]}}]
    def envelope(laws, question="주제"):
        return [{"role": "user", "content": "<search_results>\n# 법령 검색 결과\n"
                + json.dumps({"laws": laws}, ensure_ascii=False, indent=2)
                + "\n</search_results>\n# 사용자 질문\n" + question}]
    source = envelope(laws)
    encoded, mapping = encode_selection_ids(source, min_candidates=1)
    assert mapping == {"0": "109823_129", "1": "209823_130"}
    expected = source[0]["content"].replace('"id": "109823_129"', '"id": "0"').replace('"id": "209823_130"', '"id": "1"')
    assert encoded[0]["content"] == expected
    data = json.loads(encoded[0]["content"].split("# 법령 검색 결과\n")[1].split("\n</search_results>")[0])
    for law in data["laws"]:
        law["id"] = mapping[law["id"]]
    assert data["laws"] == laws
    restored = json.loads(decode_selection_ids('{"has_relevant_laws":true,"selected_ids":["1","0"]}', mapping))
    assert restored["selected_ids"] == ["209823_130", "109823_129"]
    with pytest.raises(ValueError, match="unknown"):
        decode_selection_ids('{"selected_ids":["invented"]}', mapping)
    assert encode_selection_ids(envelope([laws[0], laws[0]]), min_candidates=1)[1] == {}
    assert encode_selection_ids(envelope(laws, "109823_129 관련"), min_candidates=1)[1] == {}
    assert encode_selection_ids(source) == (source, {})
    assert source == envelope(laws)


def test_reversed_or_ambiguous_envelope_is_opaque():
    for text in ["\n</search_results># 법령 검색 결과\n{}",
                 "# 법령 검색 결과\n{}\n</search_results>\n</search_results>"]:
        messages = [{"role": "user", "content": text}]
        assert compact_selection_messages(messages) == messages
        assert encode_selection_ids(messages) == (messages, {})

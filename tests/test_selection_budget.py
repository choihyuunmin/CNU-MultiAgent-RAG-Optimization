import json

import pytest

from cnu_rag_optimization.selection_budget import SelectionBudget, budget_selection_input


def laws_text(n=3, content_len=1000):
    laws = [{"source": "article", "id": f"{i}_1", "title": f"law {i}", "subject": f"s{i}", "content": "x" * content_len,
             "law_id": str(i), "score": 0.5, "_score": 0.5, "country": "A"} for i in range(n)]
    return json.dumps({"laws": laws, "search_source": "article"}, ensure_ascii=False)


def test_keeps_every_document_and_id_and_drops_unreferenced_fields():
    text = laws_text()
    out, record = budget_selection_input(text, SelectionBudget(text_max_chars=100))
    data = json.loads(out)
    assert record["applied"] and record["documents_before"] == record["documents_after"] == 3
    assert [d["id"] for d in data["laws"]] == ["0_1", "1_1", "2_1"]
    assert set(data["laws"][0]) == {"id", "title", "subject", "content", "country"}
    assert record["dropped_fields"] == ["_score", "law_id", "score", "source"]
    assert data["search_source"] == "article"  # other top-level keys survive
    assert all(len(d["content"]) == 101 and d["content"].endswith("…") for d in data["laws"])
    assert record["shortened_fields"] == 3 and record["chars_after"] < record["chars_before"]
    assert record["ids_preserved"] is True


def test_short_text_and_missing_fields_are_left_alone():
    text = json.dumps({"laws": [{"id": "a", "title": "t", "content": "short"}, {"id": "b"}]})
    out, record = budget_selection_input(text, SelectionBudget(text_max_chars=10))
    assert json.loads(out)["laws"] == [{"id": "a", "title": "t", "content": "short"}, {"id": "b"}]
    assert record["shortened_fields"] == 0 and record["dropped_fields"] == []


@pytest.mark.parametrize("text", ["not json", json.dumps([1, 2]), json.dumps({"laws": "x"}), json.dumps({"other": []})])
def test_unrecognized_input_is_returned_unchanged(text):
    out, record = budget_selection_input(text, SelectionBudget())
    assert out == text and record["applied"] is False and record["reason"] in {"not_json", "no_document_list"}


def test_optional_document_cap_keeps_ranked_order():
    out, record = budget_selection_input(laws_text(5), SelectionBudget(max_documents=2))
    assert [d["id"] for d in json.loads(out)["laws"]] == ["0_1", "1_1"]
    assert record["documents_before"] == 5 and record["documents_after"] == 2


def test_id_is_always_kept_even_if_not_listed():
    out, _ = budget_selection_input(laws_text(1), SelectionBudget(keep_fields=("title",)))
    assert set(json.loads(out)["laws"][0]) == {"id", "title"}


@pytest.mark.parametrize("kwargs", [{"text_max_chars": 0}, {"max_documents": 0}, {"keep_fields": ()}])
def test_invalid_budget_rejected(kwargs):
    with pytest.raises(ValueError):
        SelectionBudget(**kwargs)

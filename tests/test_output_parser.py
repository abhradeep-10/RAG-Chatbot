import pytest

from ragbot.errors import LLMOutputError
from ragbot.llm.output_parser import parse_llm_answer


def test_valid_json():
    a = parse_llm_answer('{"answerable": true, "answer": "50 units.", "citations": ["C1", "C2"]}')
    assert a.answerable and a.answer == "50 units." and a.citations == ["C1", "C2"]


def test_code_fenced_json():
    a = parse_llm_answer('```json\n{"answerable": false, "answer": "Not stated.", "citations": []}\n```')
    assert not a.answerable


def test_json_wrapped_in_prose():
    a = parse_llm_answer('Sure! Here it is: {"answerable": true, "answer": "Yes.", "citations": ["C3"]} Hope that helps.')
    assert a.citations == ["C3"]


def test_citation_normalisation_and_inline_markers():
    a = parse_llm_answer('{"answerable": true, "answer": "It is 50 units [C2].", "citations": "[c1]"}')
    assert a.citations == ["C1", "C2"]
    b = parse_llm_answer('{"answerable": true, "answer": "x", "citations": [1, 2, 2]}')
    assert b.citations == ["C1", "C2"]


@pytest.mark.parametrize("raw", [
    "not json at all",
    "{broken json",
    "[1, 2, 3]",
    '{"answer": "missing answerable"}',
    '{"answerable": true, "answer": "   ", "citations": []}',
    '{"answerable": "maybe", "answer": "x"}',
])
def test_malformed_outputs_raise(raw):
    with pytest.raises(LLMOutputError):
        parse_llm_answer(raw)

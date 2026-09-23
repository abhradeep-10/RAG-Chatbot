"""Robust parsing + schema validation of the model's JSON answer."""
from __future__ import annotations

import json
import re

from pydantic import BaseModel, Field, ValidationError, field_validator

from ragbot.errors import LLMOutputError

_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$", re.I)
_INLINE_CITE_RE = re.compile(r"\[(C\d+)\]", re.I)
_USER_MSG = "The language model returned a malformed response."


class LLMAnswer(BaseModel):
    answerable: bool
    answer: str
    citations: list[str] = Field(default_factory=list)

    @field_validator("answer")
    @classmethod
    def _non_empty(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("answer is empty")
        return v

    @field_validator("citations", mode="before")
    @classmethod
    def _normalise(cls, v):
        if v is None:
            return []
        if isinstance(v, (str, int)):
            v = [v]
        if not isinstance(v, list):
            raise ValueError("citations must be a list")
        out = []
        for item in v:
            s = str(item).strip().strip("[]").strip().upper()
            if s.isdigit():
                s = f"C{s}"
            if s:
                out.append(s)
        return out


def extract_json_object(raw: str) -> dict:
    text = _FENCE_RE.sub("", raw.strip()).strip()
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start == -1 or end <= start:
            raise LLMOutputError("no JSON object in model output", _USER_MSG)
        try:
            data = json.loads(text[start:end + 1])
        except json.JSONDecodeError as exc:
            raise LLMOutputError(f"invalid JSON in model output: {exc}", _USER_MSG) from exc
    if not isinstance(data, dict):
        raise LLMOutputError("model output JSON is not an object", _USER_MSG)
    return data


def parse_llm_answer(raw: str) -> LLMAnswer:
    data = extract_json_object(raw)
    try:
        ans = LLMAnswer.model_validate(data)
    except ValidationError as exc:
        raise LLMOutputError(f"model output failed schema validation: {exc}", _USER_MSG) from exc
    inline = [m.upper() for m in _INLINE_CITE_RE.findall(ans.answer)]
    ans.citations = list(dict.fromkeys(ans.citations + inline))   # dedupe, keep order
    return ans

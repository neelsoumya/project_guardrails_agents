"""
guardrails/output/schema.py
============================
Pydantic-based schema enforcement for structured LLM outputs.

The most reliable output guardrail is to instruct the model to respond in JSON
and then immediately validate the response against a typed Pydantic schema.
This catches:
  - Hallucinated fields
  - Wrong value types
  - Missing required keys
  - Out-of-range numeric values (via Pydantic validators)

Public function:
  parse_and_validate(raw, schema) -> T   — parse + validate or raise ValueError

Usage pattern
-------------
1. Write a Pydantic model for your expected output.
2. Add a JSON-enforcing instruction to your system prompt (template below).
3. Call parse_and_validate(model_response_text, YourModel) before using the data.
"""

import json
import re
from typing import Type, TypeVar

from pydantic import BaseModel, ValidationError

T = TypeVar("T", bound=BaseModel)


def parse_and_validate(raw: str, schema: Type[T]) -> T:
    """
    Extract JSON from *raw* model output and validate it against *schema*.

    Handles common model response quirks:
      - Leading/trailing whitespace
      - Markdown code fences (```json ... ``` or ``` ... ```)
      - BOM characters

    Parameters
    ----------
    raw:    Raw text from the model's response.
    schema: A Pydantic BaseModel subclass defining the expected structure.

    Returns
    -------
    A validated instance of *schema*.

    Raises
    ------
    ValueError  if the output is not valid JSON or fails schema validation.
                The error message is human-readable and safe to log.

    Example
    -------
    >>> class Answer(BaseModel):
    ...     text: str
    ...     confidence: float
    >>> result = parse_and_validate('{"text": "Paris", "confidence": 0.99}', Answer)
    >>> result.text
    'Paris'
    """
    # --- Strip markdown fences ---
    # Models sometimes wrap JSON in ```json ... ``` even when told not to.
    cleaned = raw.strip().lstrip("\ufeff")  # strip BOM
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```\s*$", "", cleaned, flags=re.MULTILINE)
    cleaned = cleaned.strip()

    # --- Parse JSON ---
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model output is not valid JSON.\n"
            f"JSON error: {e}\n"
            f"Raw output (first 500 chars):\n{raw[:500]}"
        )

    # --- Validate against Pydantic schema ---
    try:
        return schema.model_validate(data)
    except ValidationError as e:
        raise ValueError(
            f"Model output failed schema validation for {schema.__name__}.\n"
            f"Validation errors:\n{e}"
        )


# ---------------------------------------------------------------------------
# Example schemas — copy and adapt for your application
# ---------------------------------------------------------------------------

class SimpleAnswer(BaseModel):
    """Minimal schema for a single-answer response."""
    answer: str
    confidence: float  # expected range: 0.0 to 1.0


class CitedAnswer(BaseModel):
    """Answer with supporting source citations."""
    answer: str
    confidence: float
    sources: list[str]
    caveat: str = ""   # optional; defaults to empty string if not returned


# System prompt template for reliable JSON output
# ------------------------------------------------
# Copy this into your system prompt, filling in the schema definition.
JSON_SYSTEM_PROMPT_TEMPLATE = """
You are a helpful assistant. You MUST respond ONLY with a valid JSON object.
Do not include any prose, explanation, or markdown formatting outside the JSON.
The JSON must conform exactly to this schema:
{schema_definition}

If you cannot produce a valid answer, still return JSON with a low confidence value
and explain in the "caveat" field.
""".strip()

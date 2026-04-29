"""
guardrails/output/grounding.py
================================
Hallucination / grounding check for RAG and document-grounded applications.

When a model's answer should be derived from retrieved context (e.g. a
knowledge base, uploaded documents, or search results), this module provides
a lightweight meta-check: a second, cheap model call that asks whether the
answer is actually supported by the context.

Design decisions
----------------
- Uses claude-haiku (fastest/cheapest) because this is a binary classification
  task, not a reasoning task.
- Returns a dict rather than raising, so callers can log the grounding result
  and decide their own policy (block, warn, pass through with disclaimer).
- Grounding checks should run AFTER the main model call, ideally
  asynchronously or in a background thread to avoid adding latency to the
  happy path.
"""

import json
from typing import TypedDict


class GroundingResult(TypedDict):
    grounded: bool
    reason: str


def check_grounded(answer: str, context: str) -> GroundingResult:
    """
    Ask a fast LLM judge whether *answer* is supported by *context*.

    Parameters
    ----------
    answer:  The model's response text to evaluate.
    context: The source text the answer should be grounded in
             (e.g. retrieved document chunks, search results).

    Returns
    -------
    A GroundingResult dict with:
      grounded: True if the answer is supported by context, False otherwise.
      reason:   One-sentence explanation of the grounding decision.

    Raises
    ------
    anthropic.APIError   if the API call fails.
    json.JSONDecodeError if the judge model returns malformed JSON (rare).

    Example
    -------
    >>> result = check_grounded(
    ...     answer="The Eiffel Tower is in Berlin.",
    ...     context="The Eiffel Tower is a wrought-iron lattice tower in Paris, France."
    ... )
    >>> result["grounded"]
    False
    """
    import anthropic  # local import keeps module loadable without the package

    client = anthropic.Anthropic()

    prompt = (
        f"CONTEXT:\n{context}\n\n"
        f"ANSWER:\n{answer}\n\n"
        "Is the ANSWER fully supported by the CONTEXT above, with no claims that go "
        "beyond or contradict the context? "
        'Reply ONLY with valid JSON: {"grounded": true|false, "reason": "<one sentence>"}.'
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # fast + cheap for meta-checks
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()
    return json.loads(raw)

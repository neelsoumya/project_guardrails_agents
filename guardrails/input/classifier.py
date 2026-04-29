"""
guardrails/input/classifier.py
================================
Input content classification.

Classifies user input into allowed / blocked categories BEFORE sending it
to the model, saving cost and latency for requests you would reject anyway.

Two implementations provided:
  classify_input(text)        — fast, zero-cost, keyword-based (use in production hot path)
  classify_with_llm(text)     — higher accuracy, uses Anthropic API (use for borderline cases)
"""

import json
from dataclasses import dataclass


@dataclass
class ClassificationResult:
    """
    Result of a content classification check.

    Attributes
    ----------
    allowed:    True if the content is within scope.
    category:   Semantic label (e.g. "safe", "self_harm", "medical_diagnosis").
    confidence: Subjective confidence level: "high" | "medium" | "low".
    """
    allowed: bool
    category: str
    confidence: str  # "high" | "medium" | "low"


# ---------------------------------------------------------------------------
# Blocked category definitions (rule-based classifier)
# ---------------------------------------------------------------------------
# Keys are category labels; values are keyword/phrase lists.
# A match on ANY keyword in a category triggers a block for that category.
# Add or remove categories to match your application's scope.

BLOCKED_CATEGORIES: dict[str, list[str]] = {
    "self_harm": [
        "how to hurt myself",
        "suicide method",
        "self harm",
        "ways to kill myself",
        "overdose instructions",
    ],
    "medical_diagnosis": [
        "diagnose me",
        "do i have cancer",
        "what illness do i have",
        "am i pregnant",
        "is my rash serious",
    ],
    "pii_extraction": [
        "find someone's address",
        "get their phone number",
        "look up their email",
        "find where they live",
    ],
    "illegal_activity": [
        "how to hack",
        "how to make a bomb",
        "how to synthesise drugs",
        "bypass security",
    ],
}


def classify_input(text: str) -> ClassificationResult:
    """
    Fast keyword-based content classifier.

    Scans *text* for substrings matching any blocked category keyword.
    Returns immediately on the first match.

    Confidence is "medium" for keyword matches (some false positives possible)
    and "high" for clean text (absence of keywords is a strong signal).

    This classifier runs in O(n * k) time where n = len(text) and k = total
    keywords. Suitable for the synchronous hot path in a web app.

    Example
    -------
    >>> r = classify_input("how to hack a website")
    >>> r.allowed, r.category
    (False, 'illegal_activity')
    >>> classify_input("What is the weather like?").allowed
    True
    """
    lower = text.lower()
    for category, keywords in BLOCKED_CATEGORIES.items():
        if any(kw in lower for kw in keywords):
            return ClassificationResult(
                allowed=False,
                category=category,
                confidence="medium",  # keyword match: some false-positive risk
            )
    return ClassificationResult(allowed=True, category="safe", confidence="high")


def classify_with_llm(text: str) -> ClassificationResult:
    """
    LLM-based content classifier using the Anthropic API.

    More accurate than the keyword classifier for ambiguous or paraphrased
    inputs. Uses a small, cheap model (Haiku) to keep latency and cost low.

    Best used:
      - Asynchronously, in a pre-screening step before the main model call.
      - For borderline inputs that the keyword classifier passed but that
        domain logic suggests might be risky.

    Raises
    ------
    anthropic.APIError   if the Anthropic API call fails.
    json.JSONDecodeError if the model returns malformed JSON (rare).

    Example
    -------
    >>> r = classify_with_llm("I think I might have appendicitis, what do you think?")
    >>> r.allowed, r.category
    (False, 'medical_diagnosis')
    """
    import anthropic  # imported here so the module loads without the package installed

    client = anthropic.Anthropic()

    system_prompt = (
        "You are a content safety classifier. "
        "You MUST reply ONLY with a valid JSON object — no prose, no fences. "
        "Schema: "
        '{"allowed": true|false, "category": "<label>", "reason": "<one sentence>"}. '
        "Category labels: safe | self_harm | medical_diagnosis | "
        "pii_extraction | illegal_activity | other_risk."
    )

    response = client.messages.create(
        model="claude-haiku-4-5-20251001",  # fast + cheap for meta-checks
        max_tokens=150,
        system=system_prompt,
        messages=[{"role": "user", "content": text}],
    )

    result = json.loads(response.content[0].text)
    return ClassificationResult(
        allowed=result["allowed"],
        category=result["category"],
        confidence="high",  # LLM-based: higher accuracy than keywords
    )

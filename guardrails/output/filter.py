"""
guardrails/output/filter.py
============================
Output content filtering.

Scans the model's response before it reaches the user for:
  1. System prompt leakage (model reveals internal instructions).
  2. PII in the response (model echoes or infers private data).

Returns a FilterResult dict rather than raising, so callers can choose
whether to block, redact, or log-and-pass based on their policy.
"""

from typing import TypedDict
from guardrails.input.pii import detect_pii, redact_pii


class FilterResult(TypedDict):
    safe: bool          # True if no issues were detected
    issues: list[str]   # List of issue labels detected (internal, do not expose to user)
    text: str           # Final text (redacted if redact=True and PII was found)


# ---------------------------------------------------------------------------
# System prompt leakage markers
# ---------------------------------------------------------------------------
# These strings in a model response strongly suggest the model is reflecting
# back its system prompt or internal instructions.
# Extend this list based on sensitive keywords in your own system prompt.

_LEAKAGE_MARKERS: list[str] = [
    "you are an ai assistant",
    "your system prompt",
    "instructions above",
    "as instructed by",
    "my system prompt says",
    "i was told to",
    "my instructions are",
    "the prompt i was given",
]


def filter_output(text: str, redact: bool = True) -> FilterResult:
    """
    Scan *text* (a model response) for content that should not reach the user.

    Parameters
    ----------
    text:   The raw model response string.
    redact: If True, PII found in the response is replaced with [REDACTED]
            before being returned. Set to False if you want to log the original
            and decide separately whether to redact.

    Returns
    -------
    A FilterResult with:
      safe:   True if no issues were detected.
      issues: List of short issue labels (for logging; never show to user).
      text:   The (possibly redacted) response text.

    Example
    -------
    >>> r = filter_output("Your email alice@example.com has been noted.")
    >>> r["safe"]
    False
    >>> r["issues"]
    ['pii_in_output: [email]']
    """
    issues: list[str] = []
    lower = text.lower()

    # --- Check for system prompt leakage ---
    for marker in _LEAKAGE_MARKERS:
        if marker in lower:
            issues.append(f"possible_system_prompt_leakage: '{marker}'")

    # --- Check for PII in output ---
    pii_found = detect_pii(text)
    if pii_found:
        issues.append(f"pii_in_output: {list(pii_found.keys())}")
        if redact:
            text = redact_pii(text)

    return FilterResult(
        safe=len(issues) == 0,
        issues=issues,
        text=text,
    )

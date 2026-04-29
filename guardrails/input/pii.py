"""
guardrails/input/pii.py
========================
PII (Personally Identifiable Information) detection and redaction.

Prevents user PII from being forwarded to external model APIs unnecessarily,
and removes PII from strings before storing or logging them.

Two public functions:
  detect_pii(text)  -> dict[str, list[str]]   — returns detected PII by type
  redact_pii(text)  -> str                    — replaces PII with [REDACTED]

For production systems with higher accuracy requirements, replace or augment
these regex patterns with:
  - Microsoft Presidio  (https://microsoft.github.io/presidio/)
  - spaCy NER models
  - AWS Comprehend / Azure Text Analytics PII detection
"""

import re


# ---------------------------------------------------------------------------
# PII pattern registry
# ---------------------------------------------------------------------------
# Each entry maps a human-readable label to a compiled regex pattern.
# Patterns are deliberately broad (favour recall over precision) because
# false positives (over-redaction) are safer than false negatives (PII leak).

_PII_PATTERNS: dict[str, str] = {
    # Email addresses
    "email": r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}",

    # UK mobile / landline (07xxx xxxxxx or +44 7xxx xxxxxx)
    "uk_phone": r"(\+44\s?|0)(\d\s?){9,10}",

    # US phone numbers in common formats
    "us_phone": r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",

    # Credit / debit card numbers (13-16 digits, optional separators)
    "credit_card": r"\b(?:\d[ -]?){13,16}\b",

    # UK NHS numbers (format: 000 000 0000)
    "nhs_number": r"\b\d{3}[\s-]\d{3}[\s-]\d{4}\b",

    # UK National Insurance numbers (format: AB 12 34 56 C)
    "uk_ni_number": r"\b[A-Z]{2}\s?\d{2}\s?\d{2}\s?\d{2}\s?[A-D]\b",

    # US Social Security Numbers (format: 000-00-0000)
    "us_ssn": r"\b\d{3}-\d{2}-\d{4}\b",

    # UK postcodes (e.g. CB2 1TN, SW1A 2AA)
    "uk_postcode": r"\b[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2}\b",
}

_compiled_pii: dict[str, re.Pattern] = {
    label: re.compile(pattern, re.IGNORECASE)
    for label, pattern in _PII_PATTERNS.items()
}


def detect_pii(text: str) -> dict[str, list[str]]:
    """
    Scan *text* for PII matching any registered pattern.

    Returns a dict of {pii_type: [matched_strings]}.
    An empty dict means no PII was detected.

    Note: the returned matched strings contain the actual PII values.
    Do NOT log or store the return value of this function as-is;
    log only the keys (types detected), not the values.

    Example
    -------
    >>> pii = detect_pii("Contact me at alice@example.com or 07700 900123")
    >>> list(pii.keys())
    ['email', 'uk_phone']
    """
    found: dict[str, list[str]] = {}
    for label, pattern in _compiled_pii.items():
        matches = pattern.findall(text)
        # findall returns strings or tuples (when groups present); normalise to strings
        normalised = [m if isinstance(m, str) else m[0] for m in matches]
        if normalised:
            found[label] = normalised
    return found


def redact_pii(text: str, replacement: str = "[REDACTED]") -> str:
    """
    Replace all PII detected by the registered patterns with *replacement*.

    The original *text* is never mutated; a new string is returned.

    Parameters
    ----------
    text:        Input string potentially containing PII.
    replacement: Token to substitute in place of each PII match.

    Example
    -------
    >>> redact_pii("Email me at alice@example.com")
    'Email me at [REDACTED]'
    """
    for pattern in _compiled_pii.values():
        text = pattern.sub(replacement, text)
    return text

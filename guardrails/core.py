"""
guardrails/core.py
==================
Shared data types and the canonical rejection-response factory.

All guardrail checks return (or raise) a GuardrailResult so that callers
can handle every type of block with a single isinstance / .allowed check.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class GuardrailResult:
    """
    Returned by every guardrail check.

    Attributes
    ----------
    allowed:
        True if the request/response should proceed; False if it should be blocked.
    rejection_code:
        Machine-readable reason for the block (e.g. "injection_attempt").
        None when allowed=True.
    rejection_message:
        Human-readable message that is SAFE to show to the end user.
        Never includes internal diagnostic detail.
    detail:
        Internal diagnostic string for logging and debugging.
        Must NEVER be exposed to the end user.
    """
    allowed: bool
    rejection_code: Optional[str] = None
    rejection_message: str = ""
    detail: Optional[str] = None


# Default user-facing messages keyed by rejection code.
# Override with the user_message parameter when you need context-specific wording.
_DEFAULT_MESSAGES: dict[str, str] = {
    "injection_attempt": (
        "Your input could not be processed. Please rephrase your request."
    ),
    "content_blocked": (
        "This type of request is outside the scope of this application."
    ),
    "rate_limited": (
        "You have reached the request limit. Please wait before trying again."
    ),
    "budget_exceeded": (
        "The session token budget has been reached."
    ),
    "schema_invalid": (
        "The model returned an unexpected response. Please try again."
    ),
    "output_unsafe": (
        "The response could not be delivered due to a content policy."
    ),
    "not_grounded": (
        "The response could not be verified against the provided sources."
    ),
}


def guardrail_rejection(
    reason: str,
    detail: str = "",
    user_message: str = "",
) -> GuardrailResult:
    """
    Build a blocked GuardrailResult.

    Parameters
    ----------
    reason:
        One of the keys in _DEFAULT_MESSAGES, or any custom string.
    detail:
        Internal diagnostic string (logged, never shown to user).
    user_message:
        Override the default user-facing message for this rejection code.
    """
    return GuardrailResult(
        allowed=False,
        rejection_code=reason,
        rejection_message=user_message or _DEFAULT_MESSAGES.get(reason, "Request blocked."),
        detail=detail,
    )

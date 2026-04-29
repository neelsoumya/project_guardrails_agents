# guardrails — AI input/output safety package
# Import the top-level helpers so callers can do:
#   from guardrails import guardrail_rejection, GuardrailResult
from .core import GuardrailResult, guardrail_rejection

__all__ = ["GuardrailResult", "guardrail_rejection"]

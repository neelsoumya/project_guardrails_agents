---
name: guardrails
description: Add input/output guardrails to any Python AI application, GenAI workflow, or agentic pipeline. Covers prompt injection defence, output validation, content filtering, schema enforcement, rate limiting, hallucination detection, and audit logging. Produces ready-to-run Python modules and wiring instructions.
version: 1.0
---

# Skill: AI Guardrails Implementation

## Purpose

Add production-grade guardrails to any Python codebase that calls an LLM, runs an agentic workflow, or exposes a GenAI feature to users. The skill covers the full guardrails stack: what to block before the model sees it, what to validate after the model responds, and how to log every decision for audit and debugging.

## When to trigger this skill

Use this skill when the user says things like:

- "add guardrails to my app"
- "make this safer / more robust"
- "prevent prompt injection"
- "validate LLM output"
- "stop hallucinations getting through"
- "add rate limiting / content filtering"
- "make this production-ready"
- "how do I audit what the model is doing"
- "schema enforcement on LLM output"

---

## Guardrails taxonomy

There are four layers. Implement only the layers the codebase needs; do not add layers that have no use case.

```
User input
    ↓
[LAYER 1: INPUT GUARDRAILS]      ← validate, sanitise, classify, block
    ↓
LLM / agent call
    ↓
[LAYER 2: OUTPUT GUARDRAILS]     ← parse, validate schema, fact-check, redact
    ↓
[LAYER 3: OPERATIONAL]           ← rate limiting, cost caps, retry logic
    ↓
[LAYER 4: AUDIT & OBSERVABILITY] ← structured logging, alerting
    ↓
Downstream system / user
```

---

## Layer 1 — Input guardrails

### 1a. Prompt injection defence

Prompt injection is when user-supplied text contains instructions that try to override the system prompt or hijack the model's behaviour (e.g. "Ignore previous instructions and...").

**Implementation pattern:**

```python
# guardrails/input/injection.py

import re
from typing import Optional

INJECTION_PATTERNS = [
    r"ignore\s+(previous|all|prior|above)\s+instructions",
    r"disregard\s+(your|the)\s+(system\s+)?prompt",
    r"you\s+are\s+now\s+(?:a\s+)?(?:dan|jailbreak|evil|unfiltered)",
    r"act\s+as\s+if\s+you\s+have\s+no\s+restrictions",
    r"pretend\s+you\s+are",
    r"forget\s+everything",
    r"new\s+instruction[s]?:",
    r"\[system\]",
    r"<\s*system\s*>",
]

_compiled = [re.compile(p, re.IGNORECASE) for p in INJECTION_PATTERNS]


def detect_injection(text: str) -> Optional[str]:
    """Returns the matched pattern string if injection is detected, else None."""
    for pattern in _compiled:
        m = pattern.search(text)
        if m:
            return m.group(0)
    return None


def sanitise_input(text: str, max_chars: int = 4000) -> str:
    """Light sanitisation: strip null bytes, truncate to token-safe length."""
    text = text.replace("\x00", "")
    text = text.strip()
    text = text[:max_chars]
    return text
```

**Wiring:**

```python
from guardrails.input.injection import detect_injection, sanitise_input

user_text = sanitise_input(raw_input)
if hit := detect_injection(user_text):
    return guardrail_rejection(reason="injection_attempt", detail=hit)
```

---

### 1b. Input content classification

Classify the input *before* it reaches the model so you do not pay for a call you will reject.

```python
# guardrails/input/classifier.py

from dataclasses import dataclass

@dataclass
class ClassificationResult:
    allowed: bool
    category: str   # e.g. "medical_diagnosis", "legal_advice", "safe"
    confidence: str # "high" | "medium" | "low"

BLOCKED_CATEGORIES = {
    "self_harm":          ["how to hurt myself", "suicide method", "self harm"],
    "medical_diagnosis":  ["diagnose me", "do i have cancer", "what illness"],
    "pii_extraction":     ["find someone's address", "get their phone number"],
}

def classify_input(text: str) -> ClassificationResult:
    lower = text.lower()
    for category, keywords in BLOCKED_CATEGORIES.items():
        if any(kw in lower for kw in keywords):
            return ClassificationResult(allowed=False, category=category, confidence="medium")
    return ClassificationResult(allowed=True, category="safe", confidence="high")
```

**LLM-based classifier (higher accuracy):**

```python
import anthropic, json

client = anthropic.Anthropic()

def classify_with_llm(text: str) -> ClassificationResult:
    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=100,
        system=(
            "You are a content classifier. Reply ONLY with a JSON object: "
            '{"allowed": true/false, "category": "<category>", "reason": "<one sentence>"}. '
            "Categories: safe | self_harm | medical_diagnosis | pii_extraction | other_risk."
        ),
        messages=[{"role": "user", "content": text}],
    )
    result = json.loads(response.content[0].text)
    return ClassificationResult(allowed=result["allowed"], category=result["category"], confidence="high")
```

---

### 1c. PII detection and redaction

```python
# guardrails/input/pii.py

import re

PII_PATTERNS = {
    "email":       r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z]{2,}",
    "uk_phone":    r"(\+44\s?|0)(\d\s?){9,10}",
    "us_phone":    r"\b(\+1[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b",
    "credit_card": r"\b(?:\d[ -]?){13,16}\b",
    "nhs_number":  r"\b\d{3}[\s-]\d{3}[\s-]\d{4}\b",
}

_compiled_pii = {k: re.compile(v) for k, v in PII_PATTERNS.items()}

def detect_pii(text: str) -> dict[str, list[str]]:
    """Returns {pii_type: [matched_strings]}. Empty dict = no PII found."""
    return {label: pattern.findall(text)
            for label, pattern in _compiled_pii.items()
            if pattern.findall(text)}

def redact_pii(text: str, replacement: str = "[REDACTED]") -> str:
    """Replace all detected PII with the replacement token."""
    for pattern in _compiled_pii.values():
        text = pattern.sub(replacement, text)
    return text
```

---

## Layer 2 — Output guardrails

### 2a. Schema enforcement (structured outputs)

```python
# guardrails/output/schema.py

from pydantic import BaseModel, ValidationError
from typing import Type, TypeVar
import json, re

T = TypeVar("T", bound=BaseModel)

def parse_and_validate(raw: str, schema: Type[T]) -> T:
    """
    Extract JSON from model output (tolerates markdown fences),
    parse it, and validate against the given Pydantic schema.
    Raises ValueError on failure.
    """
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip(), flags=re.MULTILINE)
    cleaned = re.sub(r"\s*```$", "", cleaned.strip(), flags=re.MULTILINE)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ValueError(f"Model output is not valid JSON: {e}")
    try:
        return schema.model_validate(data)
    except ValidationError as e:
        raise ValueError(f"Model output failed schema validation:\n{e}")


# Example schema:
class AnswerOutput(BaseModel):
    answer: str
    confidence: float   # 0.0 to 1.0
    sources: list[str]
```

**System prompt for reliable JSON output:**

```python
SYSTEM_PROMPT = """
You MUST respond ONLY with a valid JSON object matching this schema exactly:
{"answer": "<string>", "confidence": <0.0-1.0>, "sources": ["<string>", ...]}
No prose, no markdown fences.
""".strip()
```

---

### 2b. Hallucination / grounding check

```python
# guardrails/output/grounding.py

import anthropic, json

client = anthropic.Anthropic()

def check_grounded(answer: str, context: str) -> dict:
    """
    Returns {"grounded": bool, "reason": str}.
    Uses a cheap/fast model for this meta-check.
    """
    prompt = (
        f"CONTEXT:\n{context}\n\nANSWER:\n{answer}\n\n"
        "Is the answer fully supported by the context? "
        'Reply ONLY with JSON: {"grounded": true/false, "reason": "<one sentence>"}.'
    )
    response = client.messages.create(
        model="claude-haiku-4-5-20251001",
        max_tokens=150,
        messages=[{"role": "user", "content": prompt}],
    )
    return json.loads(response.content[0].text)
```

---

### 2c. Output content filtering

```python
# guardrails/output/filter.py

from guardrails.input.pii import detect_pii, redact_pii

SYSTEM_PROMPT_LEAKAGE_MARKERS = [
    "you are an ai assistant",
    "your system prompt",
    "instructions above",
    "as instructed by",
]

def filter_output(text: str, redact: bool = True) -> dict:
    """Returns {"safe": bool, "issues": [...], "text": <cleaned text>}."""
    issues = []
    lower = text.lower()
    for marker in SYSTEM_PROMPT_LEAKAGE_MARKERS:
        if marker in lower:
            issues.append(f"possible_system_prompt_leakage: '{marker}'")
    pii = detect_pii(text)
    if pii:
        issues.append(f"pii_in_output: {list(pii.keys())}")
        if redact:
            text = redact_pii(text)
    return {"safe": len(issues) == 0, "issues": issues, "text": text}
```

---

## Layer 3 — Operational guardrails

### 3a. Rate limiting (sliding window, thread-safe)

```python
# guardrails/operational/rate_limit.py

import time
from collections import defaultdict, deque
from threading import Lock

class RateLimiter:
    def __init__(self, max_calls: int, window_seconds: int):
        self.max_calls = max_calls
        self.window = window_seconds
        self._calls: dict[str, deque] = defaultdict(deque)
        self._lock = Lock()

    def is_allowed(self, user_id: str) -> bool:
        now = time.time()
        with self._lock:
            dq = self._calls[user_id]
            while dq and dq[0] < now - self.window:
                dq.popleft()
            if len(dq) >= self.max_calls:
                return False
            dq.append(now)
            return True

# limiter = RateLimiter(max_calls=10, window_seconds=60)
```

---

### 3b. Token budget tracker

```python
# guardrails/operational/cost.py

class TokenBudget:
    def __init__(self, max_input_tokens: int, max_output_tokens: int):
        self.max_input = max_input_tokens
        self.max_output = max_output_tokens
        self.used_input = 0
        self.used_output = 0

    def record(self, input_tokens: int, output_tokens: int) -> None:
        self.used_input += input_tokens
        self.used_output += output_tokens

    def check(self) -> bool:
        return self.used_input < self.max_input and self.used_output < self.max_output
```

---

### 3c. Retry with exponential backoff

```python
# guardrails/operational/retry.py

import time, logging
from typing import Callable, TypeVar

T = TypeVar("T")
log = logging.getLogger(__name__)

def with_retry(fn: Callable[[], T], max_attempts: int = 3,
               base_delay: float = 1.0, backoff: float = 2.0) -> T:
    delay = base_delay
    for attempt in range(1, max_attempts + 1):
        try:
            return fn()
        except Exception as e:
            if attempt == max_attempts:
                raise
            log.warning(f"Attempt {attempt} failed ({e}); retrying in {delay:.1f}s")
            time.sleep(delay)
            delay *= backoff
```

---

## Layer 4 — Audit and observability

```python
# guardrails/audit/logger.py

import logging, json, time, uuid
from dataclasses import dataclass, asdict, field
from typing import Any, Optional

@dataclass
class GuardrailEvent:
    event_id: str       = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float    = field(default_factory=time.time)
    user_id: str        = "anonymous"
    session_id: Optional[str] = None
    stage: str          = ""   # "input" | "output" | "operational"
    check: str          = ""   # e.g. "injection_detect", "schema_validate"
    passed: bool        = True
    detail: Any         = None
    input_preview: str  = ""   # first 200 chars only -- never store full input
    output_preview: str = ""

_audit_log = logging.getLogger("guardrails.audit")
_audit_log.setLevel(logging.INFO)

def emit(event: GuardrailEvent) -> None:
    _audit_log.info(json.dumps(asdict(event)))
```

**File handler setup in app entrypoint:**

```python
import logging, sys
from guardrails.audit.logger import _audit_log

handler = logging.FileHandler("guardrails_audit.jsonl")
handler.setFormatter(logging.Formatter("%(message)s"))
_audit_log.addHandler(handler)
_audit_log.addHandler(logging.StreamHandler(sys.stdout))  # dev only
```

---

## Canonical rejection response

```python
# guardrails/core.py

from dataclasses import dataclass
from typing import Optional

@dataclass
class GuardrailResult:
    allowed: bool
    rejection_code: Optional[str] = None
    rejection_message: str = ""
    detail: Optional[str] = None   # internal only -- never expose to user

def guardrail_rejection(reason: str, detail: str = "", user_message: str = "") -> GuardrailResult:
    DEFAULT_MESSAGES = {
        "injection_attempt": "Your input could not be processed. Please rephrase your request.",
        "content_blocked":   "This type of request is outside the scope of this application.",
        "rate_limited":      "You have reached the request limit. Please wait before trying again.",
        "budget_exceeded":   "The session token budget has been reached.",
        "schema_invalid":    "The model returned an unexpected response. Please try again.",
        "output_unsafe":     "The response could not be delivered due to a content policy.",
        "not_grounded":      "The response could not be verified against the provided sources.",
    }
    return GuardrailResult(
        allowed=False,
        rejection_code=reason,
        rejection_message=user_message or DEFAULT_MESSAGES.get(reason, "Request blocked."),
        detail=detail,
    )
```

---

## Full wiring example (Streamlit)

```python
# app_with_guardrails.py

import streamlit as st
import anthropic
from guardrails.core import guardrail_rejection
from guardrails.input.injection import detect_injection, sanitise_input
from guardrails.input.classifier import classify_input
from guardrails.input.pii import redact_pii
from guardrails.output.filter import filter_output
from guardrails.operational.rate_limit import RateLimiter
from guardrails.audit.logger import emit, GuardrailEvent

client = anthropic.Anthropic()
limiter = RateLimiter(max_calls=10, window_seconds=60)

st.title("Guardrailed AI App")
user_input = st.text_area("Your question:")

if st.button("Submit") and user_input:
    user_id = "demo_user"

    # Layer 3: rate limit
    if not limiter.is_allowed(user_id):
        emit(GuardrailEvent(user_id=user_id, stage="operational", check="rate_limit", passed=False))
        st.error(guardrail_rejection("rate_limited").rejection_message)
        st.stop()

    # Layer 1a: sanitise
    clean = sanitise_input(user_input)

    # Layer 1b: injection
    if hit := detect_injection(clean):
        emit(GuardrailEvent(user_id=user_id, stage="input", check="injection_detect",
                            passed=False, detail=hit, input_preview=clean[:200]))
        st.error(guardrail_rejection("injection_attempt").rejection_message)
        st.stop()

    # Layer 1c: classify
    clf = classify_input(clean)
    if not clf.allowed:
        emit(GuardrailEvent(user_id=user_id, stage="input", check="content_classify",
                            passed=False, detail=clf.category, input_preview=clean[:200]))
        st.error(guardrail_rejection("content_blocked").rejection_message)
        st.stop()

    # Layer 1d: PII redaction
    safe_input = redact_pii(clean)

    # Model call
    with st.spinner("Thinking..."):
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": safe_input}],
        )
    raw_output = response.content[0].text

    # Layer 2: output filter
    filtered = filter_output(raw_output)
    if not filtered["safe"]:
        emit(GuardrailEvent(user_id=user_id, stage="output", check="output_filter",
                            passed=False, detail=str(filtered["issues"]),
                            output_preview=raw_output[:200]))

    # Layer 4: audit success
    emit(GuardrailEvent(user_id=user_id, stage="output", check="all_passed", passed=True,
                        input_preview=clean[:200], output_preview=filtered["text"][:200]))

    st.write(filtered["text"])
```

---

## Project structure

```
guardrails/
├── core.py
├── input/
│   ├── __init__.py
│   ├── injection.py
│   ├── classifier.py
│   └── pii.py
├── output/
│   ├── __init__.py
│   ├── schema.py
│   ├── grounding.py
│   └── filter.py
├── operational/
│   ├── __init__.py
│   ├── rate_limit.py
│   ├── cost.py
│   └── retry.py
└── audit/
    ├── __init__.py
    └── logger.py
```

---

## Workflow

1. Read the existing codebase and identify all points where user input is received and all points where model output is consumed.
2. Determine which of the four layers are needed (not every app needs every layer).
3. Create the guardrails package structure above.
4. Wire input guardrails at the earliest possible point (before any model call).
5. Wire output guardrails immediately after the model responds (before any downstream use).
6. Add operational guardrails (rate limiting, cost caps) as singletons or middleware.
7. Configure the audit logger and verify a JSONL record is emitted for every check.
8. Add a pytest test for each guardrail module covering at least one passing and one failing case.
9. Do not add guardrails that have no concrete threat model -- over-engineering slows the app.

---

## Writing rules

- Guardrail code must never silently swallow errors. Choose fail-open or fail-closed explicitly and document the choice.
- All rejection messages shown to users must be non-diagnostic -- do not reveal which pattern matched, what the system prompt contains, or how the guardrail works internally.
- Audit logs may contain full internal detail but must never be exposed to users.
- PII must be redacted from audit log values -- log the fact of detection, not the PII itself.
- Rule-based checks (injection keywords, PII patterns, rate limits) must not add more than ~100ms to the happy path.
- Reserve LLM-based meta-checks (grounding, classification) for the async or background path.
- Prefer rule-based over LLM-based checks for high-frequency, low-ambiguity cases.

---

## Quality checklist

Before finishing, verify:

- Every model call site has at least a sanitise + injection check on input
- Every model output is filtered before being displayed or stored
- The rate limiter is initialised once and shared across requests (not re-created per call)
- Audit log is writing to a persistent file or external sink (not just stdout)
- Rejection messages are user-safe and non-diagnostic
- All guardrail modules have at least one passing and one failing pytest
- PII is not present in audit log values
- The guardrails package is importable without importing the main app (no circular imports)

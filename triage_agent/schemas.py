from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator

Decision = Literal["CLOSE", "MONITOR", "ESCALATE", "BLOCK"]
FraudType = Literal["account_takeover", "card_testing", "money_mule", "none", "other"]
Severity = Literal["low", "medium", "high", "critical"]

POSITIVE_DECISIONS = {"ESCALATE", "BLOCK"}


class Verdict(BaseModel):
    decision: Decision
    fraud_type: FraudType
    severity: Severity
    confidence: float = Field(ge=0, le=1)
    summary: str = Field(min_length=10, max_length=1200)
    key_evidence: list[str] = Field(min_length=1, max_length=12)
    recommended_actions: list[str] = Field(default_factory=list, max_length=10)

    @field_validator("decision", "severity", "fraud_type", mode="before")
    @classmethod
    def _normalise(cls, v):
        return v.strip() if isinstance(v, str) else v

    @field_validator("decision", mode="before")
    @classmethod
    def _upper(cls, v):
        return v.upper() if isinstance(v, str) else v


class Step(BaseModel):
    i: int
    tool: str
    args: dict
    result: dict | str
    latency_ms: float
    thought: str = ""


class TriageResult(BaseModel):
    alert_id: str
    mode: str
    model: str
    verdict: Verdict
    original_verdict: Verdict | None = None  # before guardrails, if changed
    guardrail_actions: list[str] = []
    workflow_status: str  # auto_closed | monitoring | analyst_queue | pending_human_approval
    steps: list[Step] = []
    injection_detected: bool = False
    fallback_used: bool = False
    usage: dict = {}
    latency_s: float = 0.0
    error: str | None = None

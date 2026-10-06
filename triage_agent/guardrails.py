"""Deterministic policy layer applied AFTER the model decides.

LLMs are good at reading evidence and writing explanations; they should not be
the last line of defence for irreversible actions. These rules are boring on
purpose and fully auditable.
"""
from __future__ import annotations

from .schemas import Verdict

AUTO_CLOSE_MAX_RISK = 0.90   # never auto-close if the ML model is this sure it's fraud
WORKFLOW = {"CLOSE": "auto_closed", "MONITOR": "monitoring", "ESCALATE": "analyst_queue",
            "BLOCK": "pending_human_approval"}
SEVERITY_ORDER = ["low", "medium", "high", "critical"]


def _at_least(sev: str, floor: str) -> str:
    return sev if SEVERITY_ORDER.index(sev) >= SEVERITY_ORDER.index(floor) else floor


def apply_policy(v: Verdict, risk_probability: float | None, injection_detected: bool,
                 tools_called: set[str]) -> tuple[Verdict, list[str], str]:
    actions: list[str] = []
    new = v.model_copy(deep=True)

    if injection_detected and new.decision in ("CLOSE", "MONITOR"):
        actions.append(f"PI-1: prompt-injection text found in event data -> {new.decision} upgraded to ESCALATE")
        new.decision, new.severity = "ESCALATE", _at_least(new.severity, "high")
        if new.fraud_type == "none":
            new.fraud_type = "other"

    if risk_probability is not None and risk_probability >= AUTO_CLOSE_MAX_RISK and new.decision == "CLOSE":
        actions.append(f"RISK-1: ML fraud probability {risk_probability:.2f} >= {AUTO_CLOSE_MAX_RISK} cannot be auto-closed -> ESCALATE")
        new.decision, new.severity = "ESCALATE", _at_least(new.severity, "high")

    if new.decision in ("CLOSE", "BLOCK") and "get_risk_score" not in tools_called:
        actions.append(f"EVID-1: {new.decision} without consulting the risk model -> routed to analyst (ESCALATE)")
        new.decision = "ESCALATE"

    if new.decision in ("ESCALATE", "BLOCK") and new.fraud_type == "none":
        new.fraud_type = "other"
        actions.append("CONS-1: positive decision with fraud_type=none -> fraud_type set to 'other'")

    if new.decision == "BLOCK":
        actions.append("HITL-1: BLOCK is never executed automatically; queued for human approval")

    return new, actions, WORKFLOW[new.decision]

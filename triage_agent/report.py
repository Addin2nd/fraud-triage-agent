from __future__ import annotations

from .schemas import TriageResult

STATUS_LABEL = {"auto_closed": "Auto-closed", "monitoring": "Watchlist (monitoring)",
                "analyst_queue": "Sent to analyst queue", "pending_human_approval": "BLOCK pending human approval"}


def render_markdown(res: TriageResult, alert: dict) -> str:
    v = res.verdict
    lines = [
        f"# Incident triage report — {res.alert_id}",
        "",
        f"| Field | Value |\n|---|---|\n| Rule | `{alert['rule']}` — {alert['description']} |"
        f"\n| Customer | `{alert['user_id']}` |\n| Event | `{alert['event_id']}` ({alert['event_type']}, "
        f"IDR {alert['amount']:,}) |\n| Raised at | {alert['created_at']} |"
        f"\n| **Decision** | **{v.decision}** ({v.severity}, confidence {v.confidence:.0%}) |"
        f"\n| Fraud type | {v.fraud_type} |\n| Workflow | {STATUS_LABEL[res.workflow_status]} |"
        f"\n| Agent | {res.model} · {len(res.steps)} steps · {res.latency_s:.1f}s |",
        "",
        "## Summary",
        v.summary,
        "",
        "## Key evidence",
        *[f"- {e}" for e in v.key_evidence],
        "",
        "## Recommended actions",
        *[f"- {a}" for a in v.recommended_actions],
    ]
    if res.guardrail_actions:
        lines += ["", "## Guardrail interventions", *[f"- {g}" for g in res.guardrail_actions]]
        if res.original_verdict:
            lines.append(f"- Agent's original decision: **{res.original_verdict.decision}**")
    if res.injection_detected:
        lines += ["", "> ⚠️ Prompt-injection text was found in attacker-controlled fields and ignored."]
    if res.error:
        lines += ["", f"> Fallback used: {res.error}"]
    lines += ["", "## Investigation trace", *[f"{s.i}. `{s.tool}({', '.join(f'{k}={v!r}' for k, v in s.args.items()) if s.tool != 'submit_verdict' else '…'})`"
                                               for s in res.steps]]
    return "\n".join(lines) + "\n"

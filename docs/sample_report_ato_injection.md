# Incident triage report — ALT-00083

| Field | Value |
|---|---|
| Rule | `NEW_DEVICE_HIGH_VALUE` — High-value outgoing payment from a device not previously seen on this account |
| Customer | `U00083` |
| Event | `E0006452` (payout, IDR 668,000) |
| Raised at | 2026-10-03T03:02:00 |
| **Decision** | **ESCALATE** (high, confidence 50%) |
| Fraud type | account_takeover |
| Workflow | Sent to analyst queue |
| Agent | playbook:v1 · 7 steps · 0.1s |

## Summary
Rule NEW_DEVICE_HIGH_VALUE fired on a payout. Combined evidence score 0.75 (model 0.89, similar cases 0.43) -> ESCALATE; pattern most consistent with account_takeover.

## Key evidence
- 1 failed logins in the last 24h
- Payout goes to a beneficiary never paid before
- Amount is 3.92x the customer's median transaction
- IP 103.246.0.122 is residential (MyRepublic, Semarang), abuse score 49, seen 0x on this account before the last 24h
- ML fraud probability 0.89 (high); 43% of 7 similar past cases were fraud
- Prompt-injection style text found in: triggering_event.note, events.[2].note

## Recommended actions
- Analyst review within SLA
- Hold the payout until reviewed
- Verify customer via call-back

> ⚠️ Prompt-injection text was found in attacker-controlled fields and ignored.

## Investigation trace
1. `get_alert_event()`
2. `get_user_profile(user_id='U00083')`
3. `get_recent_activity(user_id='U00083', hours=24)`
4. `check_ip_reputation(ip='103.246.0.122')`
5. `check_travel(user_id='U00083', hours=48)`
6. `get_risk_score(event_id='E0006452')`
7. `search_similar_cases(event_id='E0006452', k=7)`


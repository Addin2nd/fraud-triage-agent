# Incident triage report — ALT-00010

| Field | Value |
|---|---|
| Rule | `HIGH_VALUE_TXN` — Transaction amount far above the customer's usual spend |
| Customer | `U00010` |
| Event | `E0000821` (transaction, IDR 917,000) |
| Raised at | 2026-10-01T05:37:00 |
| **Decision** | **CLOSE** (low, confidence 99%) |
| Fraud type | none |
| Workflow | Auto-closed |
| Agent | playbook:v1 · 7 steps · 0.0s |

## Summary
Rule HIGH_VALUE_TXN fired on a transaction. Combined evidence score 0.00 (model 0.01, similar cases 0.00) -> CLOSE.

## Key evidence
- IP 45.70.123.73 is mobile (XL Axiata, Amsterdam), abuse score 3, seen 0x on this account before the last 24h
- ML fraud probability 0.01 (low); 0% of 7 similar past cases were fraud

## Recommended actions
- Close as false positive
- Feed outcome back to rule tuning

## Investigation trace
1. `get_alert_event()`
2. `get_user_profile(user_id='U00010')`
3. `get_recent_activity(user_id='U00010', hours=24)`
4. `check_ip_reputation(ip='45.70.123.73')`
5. `check_travel(user_id='U00010', hours=48)`
6. `get_risk_score(event_id='E0000821')`
7. `search_similar_cases(event_id='E0000821', k=7)`


## Evaluation — playbook:v1

Alerts: **200** (78 fraud). Positive = ESCALATE or BLOCK.

| Metric | Value |
|---|---|
| Precision | 96.2% |
| Recall (fraud caught) | 97.4% |
| F1 | 0.968 |
| False-positive rate | 2.5% |
| Recall before guardrails | 97.4% |
| Fraud-type accuracy (on fraud) | 96.2% |
| Alerts auto-resolved (no analyst needed) | 60.5% |
| Prompt-injection attacks resisted | 4/4 |
| Guardrail overrides | 0 |
| LLM fallbacks | 0 |
| Avg tool steps / alert | 7.0 |
| Avg latency / alert | 0.03s |
| Tokens (in / out) | 0 / 0 |

Confusion: TP 76 · FP 3 · TN 119 · FN 2

| Scenario (hidden label) | CLOSE | MONITOR | ESCALATE | BLOCK |
|---|---|---|---|---|
| ato_classic | 0 | 0 | 6 | 14 |
| ato_subtle | 2 | 0 | 9 | 8 |
| big_purchase | 20 | 0 | 0 | 0 |
| card_testing | 0 | 0 | 0 | 19 |
| merchant_settlement | 5 | 0 | 1 | 0 |
| money_mule | 0 | 0 | 20 | 0 |
| new_beneficiary_ok | 13 | 1 | 2 | 0 |
| new_phone | 24 | 0 | 0 | 0 |
| travel | 26 | 0 | 0 | 0 |
| velocity_noise | 14 | 0 | 0 | 0 |
| vpn_user | 16 | 0 | 0 | 0 |

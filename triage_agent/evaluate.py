"""Offline evaluation harness.

Runs an agent over the labelled alert queue and reports what a fraud-ops lead
actually cares about: how many frauds we catch, how much analyst time we save,
how often the agent is overruled by policy, and whether it can be manipulated.
"""
from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .agent import Triage
from .schemas import POSITIVE_DECISIONS


def evaluate(triage: Triage, labels_path: str | Path, limit: int | None = None, workers: int = 1,
             out_dir: str | Path = "results", tag: str | None = None) -> dict:
    labels = json.loads(Path(labels_path).read_text())
    ids = sorted(triage.store.alerts)[: limit or None]
    t0 = time.perf_counter()
    if workers > 1:
        with ThreadPoolExecutor(workers) as ex:
            results = list(ex.map(triage.run, ids))
    else:
        results = [triage.run(a) for a in ids]
    wall = time.perf_counter() - t0

    tp = fp = tn = fn = 0
    per_scenario: dict[str, Counter] = defaultdict(Counter)
    type_hits = type_total = 0
    rows = []
    for r in results:
        lab = labels[r.alert_id]
        pred_pos = r.verdict.decision in POSITIVE_DECISIONS
        if lab["is_fraud"]:
            tp += pred_pos
            fn += not pred_pos
            type_total += 1
            type_hits += r.verdict.fraud_type == lab["fraud_type"]
        else:
            fp += pred_pos
            tn += not pred_pos
        per_scenario[lab["scenario"]][r.verdict.decision] += 1
        rows.append({"alert_id": r.alert_id, "scenario": lab["scenario"], "truth": lab["expected_decision"],
                     "pred": r.verdict.decision, "pred_before_guardrails": (r.original_verdict or r.verdict).decision,
                     "fraud_type_true": lab["fraud_type"], "fraud_type_pred": r.verdict.fraud_type,
                     "injection": lab["prompt_injection"], "steps": len(r.steps), "fallback": r.fallback_used,
                     "guardrails": r.guardrail_actions, "latency_s": r.latency_s, **r.usage})

    n = len(results)
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    inj = [x for x in rows if x["injection"]]
    raw_pos_on_fraud = sum(x["pred_before_guardrails"] in POSITIVE_DECISIONS for x in rows if labels[x["alert_id"]]["is_fraud"])
    summary = {
        "agent": results[0].model if results else "", "n_alerts": n, "fraud_alerts": tp + fn,
        "precision": round(prec, 3), "recall": round(rec, 3),
        "f1": round(2 * prec * rec / (prec + rec), 3) if prec + rec else 0.0,
        "false_positive_rate": round(fp / (fp + tn), 3) if fp + tn else 0.0,
        "agent_recall_before_guardrails": round(raw_pos_on_fraud / (tp + fn), 3) if tp + fn else 0.0,
        "fraud_type_accuracy": round(type_hits / type_total, 3) if type_total else 0.0,
        "auto_resolved_share": round(sum(r.workflow_status in ("auto_closed", "monitoring") for r in results) / n, 3) if n else 0,
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "injection_cases": len(inj),
        "injection_resisted": sum(x["pred"] in POSITIVE_DECISIONS for x in inj),
        "guardrail_interventions": sum(any(not g.startswith("HITL") for g in x["guardrails"]) for x in rows),
        "fallbacks": sum(x["fallback"] for x in rows),
        "avg_steps": round(sum(x["steps"] for x in rows) / n, 2) if n else 0,
        "avg_latency_s": round(sum(x["latency_s"] for x in rows) / n, 3) if n else 0,
        "total_input_tokens": sum(x.get("input_tokens", 0) for x in rows),
        "total_output_tokens": sum(x.get("output_tokens", 0) for x in rows),
        "wall_time_s": round(wall, 1),
        "decisions_by_scenario": {k: dict(v) for k, v in sorted(per_scenario.items())},
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if not tag:
        tag = results[0].mode.replace(">", "").replace("-", "_") if results else "empty"
    (out / f"eval_{tag}.json").write_text(json.dumps({"summary": summary, "rows": rows}, indent=1))
    (out / f"eval_{tag}.md").write_text(to_markdown(summary))
    return summary


def to_markdown(s: dict) -> str:
    c = s["confusion"]
    lines = [f"## Evaluation — {s['agent']}", "",
             f"Alerts: **{s['n_alerts']}** ({s['fraud_alerts']} fraud). Positive = ESCALATE or BLOCK.", "",
             "| Metric | Value |", "|---|---|",
             f"| Precision | {s['precision']:.1%} |", f"| Recall (fraud caught) | {s['recall']:.1%} |",
             f"| F1 | {s['f1']:.3f} |", f"| False-positive rate | {s['false_positive_rate']:.1%} |",
             f"| Recall before guardrails | {s['agent_recall_before_guardrails']:.1%} |",
             f"| Fraud-type accuracy (on fraud) | {s['fraud_type_accuracy']:.1%} |",
             f"| Alerts auto-resolved (no analyst needed) | {s['auto_resolved_share']:.1%} |",
             f"| Prompt-injection attacks resisted | {s['injection_resisted']}/{s['injection_cases']} |",
             f"| Guardrail overrides | {s['guardrail_interventions']} |", f"| LLM fallbacks | {s['fallbacks']} |",
             f"| Avg tool steps / alert | {s['avg_steps']} |", f"| Avg latency / alert | {s['avg_latency_s']}s |",
             f"| Tokens (in / out) | {s['total_input_tokens']:,} / {s['total_output_tokens']:,} |", "",
             f"Confusion: TP {c['tp']} · FP {c['fp']} · TN {c['tn']} · FN {c['fn']}", "",
             "| Scenario (hidden label) | CLOSE | MONITOR | ESCALATE | BLOCK |", "|---|---|---|---|---|"]
    for sc, d in s["decisions_by_scenario"].items():
        lines.append(f"| {sc} | " + " | ".join(str(d.get(k, 0)) for k in ("CLOSE", "MONITOR", "ESCALATE", "BLOCK")) + " |")
    return "\n".join(lines) + "\n"

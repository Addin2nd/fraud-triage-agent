"""Analyst console for the Fraud Alert Triage Agent.

    streamlit run app/streamlit_app.py
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from triage_agent.cli import DATA_LIVE, MODELS, RESULTS, build_triage, load_dotenv  # noqa: E402
from triage_agent.llm import PRESETS  # noqa: E402
from triage_agent.report import render_markdown  # noqa: E402

load_dotenv()
st.set_page_config(page_title="Fraud Alert Triage Agent", page_icon="🛡️", layout="wide")

DECISION_COLOR = {"CLOSE": "green", "MONITOR": "blue", "ESCALATE": "orange", "BLOCK": "red"}

if not (DATA_LIVE / "alerts.json").exists() or not (MODELS / "risk_model.pkl").exists():
    st.error("No data/model found. Run `python -m triage_agent setup` first.")
    st.stop()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("🛡️ Triage Agent")
    providers = ["offline"] + list(PRESETS)
    default = os.getenv("LLM_PROVIDER", "offline")
    provider = st.selectbox("Brain", providers, index=providers.index(default) if default in providers else 0,
                            help="offline = deterministic SOP playbook (no API key). Others = LLM tool-calling agent.")
    model = None
    if provider != "offline":
        wire, _, key_env, default_model = PRESETS[provider]
        model = st.text_input("Model", value=os.getenv("LLM_MODEL", default_model))
        if key_env:
            key = st.text_input(key_env, type="password", value=os.getenv(key_env, ""))
            if key:
                os.environ[key_env] = key
    st.caption("Guardrails always on: scope limits, injection scanning, risk floor for auto-close, human approval for BLOCK.")


@st.cache_resource(show_spinner=False)
def get_triage(provider: str, model: str | None, key_fingerprint: str):
    return build_triage(provider, model)


def triage():
    key_env = PRESETS.get(provider, ("", "", "", ""))[2]
    fp = str(hash(os.getenv(key_env, ""))) if key_env else ""
    try:
        return get_triage(provider, model, fp)
    except Exception as e:  # missing key etc.
        st.error(str(e))
        st.stop()


if "results" not in st.session_state:
    st.session_state.results = {}

t = triage()
alerts = pd.DataFrame(t.store.alerts.values())

st.title("Fraud Alert Triage Agent")
st.caption("An AI agent investigates each alert with tools (profile, activity, IP intel, travel, ML risk, similar cases), "
           "decides CLOSE / MONITOR / ESCALATE / BLOCK and writes the incident report. Data is synthetic.")

tab_queue, tab_inv, tab_eval, tab_audit = st.tabs(["📥 Alert queue", "🔎 Investigate", "📊 Evaluation", "🧾 Audit log"])

# ------------------------------------------------------------------ queue
with tab_queue:
    c1, c2 = st.columns([3, 1])
    rules = c1.multiselect("Filter by rule", sorted(alerts["rule"].unique()))
    n_batch = c2.number_input("Batch size", 1, 50, 10)
    view = alerts[alerts["rule"].isin(rules)] if rules else alerts
    res = st.session_state.results
    view = view.assign(
        decision=[res[a].verdict.decision if a in res else "" for a in view["alert_id"]],
        workflow=[res[a].workflow_status if a in res else "open" for a in view["alert_id"]],
    )
    if st.button(f"▶️ Triage next {n_batch} open alerts", type="primary"):
        todo = [a for a in view["alert_id"] if a not in res][: int(n_batch)]
        bar = st.progress(0.0)
        for i, aid in enumerate(todo, 1):
            res[aid] = t.run(aid)
            bar.progress(i / len(todo), text=f"{aid}: {res[aid].verdict.decision}")
        st.rerun()
    done = view[view["decision"] != ""]
    if len(done):
        k = st.columns(4)
        for col, d in zip(k, ["CLOSE", "MONITOR", "ESCALATE", "BLOCK"]):
            col.metric(d, int((done["decision"] == d).sum()))
    st.dataframe(view[["alert_id", "created_at", "rule", "user_id", "event_type", "amount", "decision", "workflow"]],
                 width="stretch", hide_index=True, height=480)

# ------------------------------------------------------------------ investigate
with tab_inv:
    aid = st.selectbox("Alert", alerts["alert_id"].tolist())
    alert = t.store.alerts[aid]
    st.markdown(f"**{alert['rule']}** — {alert['description']}  \nCustomer `{alert['user_id']}` · "
                f"{alert['event_type']} · IDR {alert['amount']:,} · {alert['created_at']}")
    if st.button("🔎 Investigate this alert", type="primary"):
        with st.spinner("Agent is investigating ..."):
            st.session_state.results[aid] = t.run(aid)
    r = st.session_state.results.get(aid)
    if r:
        v = r.verdict
        m = st.columns(5)
        m[0].markdown(f"### :{DECISION_COLOR[v.decision]}[{v.decision}]")
        m[1].metric("Severity", v.severity)
        m[2].metric("Confidence", f"{v.confidence:.0%}")
        m[3].metric("Fraud type", v.fraud_type)
        m[4].metric("Steps · time", f"{len(r.steps)} · {r.latency_s:.1f}s")
        for g in r.guardrail_actions:
            (st.info if g.startswith("HITL") else st.warning)(f"Guardrail: {g}")
        if r.injection_detected:
            st.error("Prompt-injection text detected in attacker-controlled fields; it was treated as data.")
        if r.fallback_used:
            st.warning(f"LLM failed, playbook fallback used: {r.error}")
        st.subheader("Summary")
        st.write(v.summary)
        c1, c2 = st.columns(2)
        c1.subheader("Key evidence")
        c1.markdown("\n".join(f"- {e}" for e in v.key_evidence))
        c2.subheader("Recommended actions")
        c2.markdown("\n".join(f"- {a}" for a in v.recommended_actions))

        if r.workflow_status == "pending_human_approval":
            st.subheader("Human approval required")
            with st.form(f"hitl_{aid}"):
                analyst = st.text_input("Analyst name")
                comment = st.text_area("Comment")
                a1, a2 = st.columns(2)
                approve = a1.form_submit_button("✅ Approve BLOCK")
                reject = a2.form_submit_button("↩️ Reject (send to analyst queue)")
                if (approve or reject) and analyst:
                    t.record_human_decision(aid, approved=approve, analyst=analyst, comment=comment)
                    st.success("Decision recorded in the audit log.")
                elif approve or reject:
                    st.error("Analyst name is required.")

        st.subheader("Investigation trace")
        for s in r.steps:
            with st.expander(f"{s.i}. {s.tool}({', '.join(f'{k}={v!r}' for k, v in s.args.items()) if s.tool != 'submit_verdict' else '…'})"
                             f" · {s.latency_ms:.0f} ms"):
                if s.thought:
                    st.caption(f"Agent reasoning: {s.thought}")
                st.json(s.result if isinstance(s.result, dict) else {"result": s.result}, expanded=False)
        md = render_markdown(r, alert)
        st.download_button("⬇️ Download report (.md)", md, file_name=f"{aid}_report.md")

# ------------------------------------------------------------------ evaluation
with tab_eval:
    files = sorted(RESULTS.glob("eval_*.json"))
    if not files:
        st.info("No evaluation yet. Run `python -m triage_agent eval` (add `--provider anthropic` etc. for the LLM agent).")
    else:
        rows = []
        for f in files:
            s = json.loads(f.read_text())["summary"]
            rows.append({"run": f.stem.removeprefix("eval_"), "agent": s["agent"], "alerts": s["n_alerts"],
                         "precision": s["precision"], "recall": s["recall"], "f1": s["f1"],
                         "FPR": s["false_positive_rate"], "type acc.": s["fraud_type_accuracy"],
                         "auto-resolved": s["auto_resolved_share"],
                         "injection resisted": f"{s['injection_resisted']}/{s['injection_cases']}",
                         "avg steps": s["avg_steps"], "avg latency s": s["avg_latency_s"]})
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
        pick = st.selectbox("Run detail", [f.stem for f in files])
        data = json.loads((RESULTS / f"{pick}.json").read_text())
        st.markdown("**Decisions by hidden scenario**")
        st.dataframe(pd.DataFrame(data["summary"]["decisions_by_scenario"]).T.fillna(0).astype(int), width="stretch")
        errs = pd.DataFrame(data["rows"])
        errs = errs[(errs["truth"].isin(["CLOSE"])) != (~errs["pred"].isin(["ESCALATE", "BLOCK"]))]
        st.markdown(f"**Misclassified alerts ({len(errs)})**")
        st.dataframe(errs[["alert_id", "scenario", "truth", "pred", "fraud_type_true", "fraud_type_pred"]], hide_index=True)

# ------------------------------------------------------------------ audit
with tab_audit:
    log = RESULTS / "audit_log.jsonl"
    if log.exists():
        recs = [json.loads(line) for line in log.read_text().splitlines()[-300:]]
        st.dataframe(pd.DataFrame(recs[::-1]), hide_index=True, width="stretch")
    else:
        st.info("Audit log is empty.")

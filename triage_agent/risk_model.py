"""ML risk model + historical case base.

* `train()` fits a gradient-boosting classifier on a *separate* historical
  dataset (different seed from the live alert queue -> no leakage) and saves
  it together with a "case base": past investigated alerts with their outcome,
  used by the agent's `search_similar_cases` tool.
* `RiskModel.score()` returns a probability plus a simple, model-agnostic
  explanation: how much the score drops when each feature is reset to its
  typical (benign median) value.
"""
from __future__ import annotations

import json
import pickle
from pathlib import Path

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split

from .store import FEATURES, DataStore


def build_matrix(store: DataStore, labels: dict) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    X, y, meta = [], [], []
    for aid, alert in store.alerts.items():
        f = store.features(alert["event_id"])
        X.append([f[k] for k in FEATURES])
        lab = labels[aid]
        y.append(int(lab["is_fraud"]))
        meta.append({"case_id": aid.replace("ALT", "CASE"), "rule": alert["rule"], "fraud_type": lab["fraud_type"],
                     "outcome": "confirmed_fraud" if lab["is_fraud"] else "false_positive",
                     "decision": lab["expected_decision"]})
    return np.array(X, dtype=float), np.array(y), meta


def train(history_dir: str | Path, model_dir: str | Path) -> dict:
    store = DataStore(history_dir)
    labels = json.loads((Path(history_dir) / "labels.json").read_text())
    X, y, meta = build_matrix(store, labels)
    Xtr, Xva, ytr, yva = train_test_split(X, y, test_size=0.25, random_state=0, stratify=y)
    clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08, max_leaf_nodes=15,
                                         l2_regularization=1.0, random_state=0)
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xva)[:, 1]
    metrics = {"val_roc_auc": round(float(roc_auc_score(yva, p)), 4),
               "val_pr_auc": round(float(average_precision_score(yva, p)), 4),
               "n_train": int(len(ytr)), "n_val": int(len(yva)), "fraud_rate": round(float(y.mean()), 3)}
    clf.fit(X, y)  # refit on everything for deployment

    out = Path(model_dir)
    out.mkdir(parents=True, exist_ok=True)
    baseline = np.median(X[y == 0], axis=0)
    with open(out / "risk_model.pkl", "wb") as fh:
        pickle.dump({"model": clf, "features": FEATURES, "baseline": baseline, "metrics": metrics}, fh)
    mu, sd = X.mean(axis=0), X.std(axis=0) + 1e-9
    case_base = {"features": FEATURES, "mu": mu.tolist(), "sd": sd.tolist(), "X": X.tolist(), "meta": meta}
    (out / "case_base.json").write_text(json.dumps(case_base))
    (out / "model_card.json").write_text(json.dumps(metrics, indent=2))
    return metrics


class RiskModel:
    def __init__(self, model_dir: str | Path):
        d = Path(model_dir)
        with open(d / "risk_model.pkl", "rb") as fh:
            obj = pickle.load(fh)
        self.model, self.features, self.baseline = obj["model"], obj["features"], obj["baseline"]
        self.metrics = obj["metrics"]
        cb = json.loads((d / "case_base.json").read_text())
        self.cb_X = np.array(cb["X"])
        self.cb_mu, self.cb_sd = np.array(cb["mu"]), np.array(cb["sd"])
        self.cb_meta = cb["meta"]

    def score(self, feats: dict, top_k: int = 5) -> dict:
        x = np.array([[feats[k] for k in self.features]], dtype=float)
        p = float(self.model.predict_proba(x)[0, 1])
        # leave-one-feature-at-baseline attribution
        contrib = []
        for i, name in enumerate(self.features):
            if x[0, i] == self.baseline[i]:
                continue
            x2 = x.copy()
            x2[0, i] = self.baseline[i]
            delta = p - float(self.model.predict_proba(x2)[0, 1])
            if abs(delta) >= 0.01:
                contrib.append({"feature": name, "value": feats[name], "impact": round(delta, 3)})
        contrib.sort(key=lambda c: -abs(c["impact"]))
        band = "critical" if p >= 0.9 else "high" if p >= 0.7 else "medium" if p >= 0.4 else "low"
        return {"fraud_probability": round(p, 4), "risk_band": band, "top_drivers": contrib[:top_k],
                "model": "HistGradientBoostingClassifier", "model_val_pr_auc": self.metrics["val_pr_auc"]}

    def similar_cases(self, feats: dict, k: int = 5) -> dict:
        x = (np.array([feats[f] for f in self.features], dtype=float) - self.cb_mu) / self.cb_sd
        d = np.sqrt((((self.cb_X - self.cb_mu) / self.cb_sd - x) ** 2).sum(axis=1))
        idx = np.argsort(d)[:k]
        cases = [{**self.cb_meta[i], "distance": round(float(d[i]), 3)} for i in idx]
        fraud_share = sum(c["outcome"] == "confirmed_fraud" for c in cases) / k
        types: dict[str, int] = {}
        for c in cases:
            if c["fraud_type"] != "none":
                types[c["fraud_type"]] = types.get(c["fraud_type"], 0) + 1
        return {"k": k, "confirmed_fraud_share": round(fraud_share, 2), "fraud_type_counts": types, "cases": cases}

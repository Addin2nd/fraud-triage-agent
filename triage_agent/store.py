"""Read-only data access layer + feature engineering.

The agent only ever touches data through this class (via tools). Ground-truth
labels are deliberately NOT loaded here — see `evaluate.py`.
"""
from __future__ import annotations

import json
import math
from collections import Counter
from datetime import datetime, timedelta
from functools import lru_cache
from pathlib import Path

from .data import CITIES

FEATURES = [
    "is_new_device", "is_new_ip", "is_new_country", "distance_from_home_km", "max_travel_speed_kmh",
    "ip_abuse_score", "ip_is_anonymizer", "failed_logins_24h", "password_change_24h", "amount_ratio",
    "txn_count_1h", "distinct_merchants_1h", "small_txn_count_1h", "inbound_count_24h", "outflow_ratio_24h",
    "is_new_beneficiary", "account_age_days", "hour_deviation", "is_new_merchant",
]
ESTABLISHED_GAP = timedelta(hours=24)  # history older than this counts as "known behaviour"


def haversine_km(a: tuple[float, float], b: tuple[float, float]) -> float:
    lat1, lon1, lat2, lon2 = map(math.radians, (*a, *b))
    h = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * 6371 * math.asin(math.sqrt(h))


def _coords(city: str) -> tuple[float, float]:
    _, lat, lon = CITIES[city]
    return lat, lon


class DataStore:
    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        self.users = {u["user_id"]: u for u in json.loads((self.dir / "users.json").read_text())}
        self.ips = {i["ip"]: i for i in json.loads((self.dir / "ip_reputation.json").read_text())}
        self.alerts = {a["alert_id"]: a for a in json.loads((self.dir / "alerts.json").read_text())}
        events = json.loads((self.dir / "events.json").read_text())
        self.events = {e["event_id"]: e for e in events}
        self.by_user: dict[str, list[dict]] = {}
        for e in events:
            e["_t"] = datetime.fromisoformat(e["ts"])
            self.by_user.setdefault(e["user_id"], []).append(e)
        for lst in self.by_user.values():
            lst.sort(key=lambda e: e["_t"])

    # ------------------------------------------------------------ helpers
    @staticmethod
    def public(e: dict) -> dict:
        return {k: v for k, v in e.items() if not k.startswith("_") and v not in ("", None)}

    def _split(self, event_id: str) -> tuple[dict, list[dict], list[dict]]:
        """Return (event, established history, recent window incl. the event)."""
        ev = self.events[event_id]
        hist = self.by_user[ev["user_id"]]
        cutoff = ev["_t"] - ESTABLISHED_GAP
        established = [e for e in hist if e["_t"] < cutoff]
        recent = [e for e in hist if cutoff <= e["_t"] <= ev["_t"]]
        return ev, established, recent

    # ------------------------------------------------------------ profile
    def user_profile(self, user_id: str, as_of: datetime | None = None) -> dict:
        u = self.users[user_id]
        hist = self.by_user.get(user_id, [])
        if as_of is not None:
            hist = [e for e in hist if e["_t"] < as_of - ESTABLISHED_GAP]
        logins = [e for e in hist if e["event_type"] == "login"]
        txns = [e for e in hist if e["event_type"] == "transaction"]
        payouts = [e for e in hist if e["event_type"] == "payout"]
        ip_kinds = Counter(self.ips[e["ip"]]["kind"] for e in logins)
        hours = sorted(e["_t"].hour for e in logins)
        amounts = sorted(e["amount"] for e in txns)
        return {
            **u,
            "history_days_observed": (hist[-1]["_t"] - hist[0]["_t"]).days if len(hist) > 1 else 0,
            "known_devices": sorted({e["device_id"] for e in logins}),
            "known_countries": sorted({e["country"] for e in logins}),
            "known_cities": sorted({e["city"] for e in logins}),
            "known_beneficiaries": sorted({e["beneficiary_id"] for e in payouts}),
            "login_count_30d": len(logins),
            "login_ip_types": dict(ip_kinds),
            "typical_login_hour": hours[len(hours) // 2] if hours else None,
            "median_txn_amount": amounts[len(amounts) // 2] if amounts else None,
            "p90_txn_amount": amounts[int(len(amounts) * 0.9)] if amounts else None,
            "txn_count_30d": len(txns),
            "payout_count_30d": len(payouts),
        }

    def recent_activity(self, user_id: str, until: datetime, hours: int = 48, limit: int = 40) -> list[dict]:
        start = until - timedelta(hours=hours)
        evs = [self.public(e) for e in self.by_user.get(user_id, []) if start <= e["_t"] <= until]
        return evs[-limit:]

    # ------------------------------------------------------------ features
    @lru_cache(maxsize=4096)
    def features(self, event_id: str) -> dict:
        ev, est, recent = self._split(event_id)
        u = self.users[ev["user_id"]]
        ip = self.ips[ev["ip"]]
        est_logins = [e for e in est if e["event_type"] == "login"]
        known_dev = {e["device_id"] for e in est}
        known_ip = {e["ip"] for e in est}
        known_cty = {e["country"] for e in est_logins} or {u["home_country"]}
        known_ben = {e["beneficiary_id"] for e in est if e["event_type"] == "payout"}
        known_merch = {e["merchant"] for e in est if e["event_type"] == "transaction"}
        est_amounts = sorted(e["amount"] for e in est if e["event_type"] in ("transaction", "payout"))
        median_amt = est_amounts[len(est_amounts) // 2] if est_amounts else 100_000
        hours = sorted(e["_t"].hour for e in est_logins)
        typ_hour = hours[len(hours) // 2] if hours else 19
        hd = abs(ev["_t"].hour - typ_hour)

        # impossible travel: max speed between consecutive successful logins in last 48h
        win = [e for e in self.by_user[ev["user_id"]]
               if ev["_t"] - timedelta(hours=48) <= e["_t"] <= ev["_t"] and e["event_type"] in ("login", "transaction", "payout")]
        speed = 0.0
        for a, b in zip(win, win[1:]):
            d = haversine_km(_coords(a["city"]), _coords(b["city"]))
            dt_h = max((b["_t"] - a["_t"]).total_seconds() / 3600, 0.25)
            if d > 50:
                speed = max(speed, d / dt_h)

        hour_ago = ev["_t"] - timedelta(hours=1)
        last_hour_tx = [e for e in recent if e["event_type"] == "transaction" and e["_t"] >= hour_ago]
        inbound = [e for e in recent if e["event_type"] == "inbound_transfer"]
        inbound_sum = sum(e["amount"] for e in inbound)
        out_sum = sum(e["amount"] for e in recent if e["event_type"] == "payout")
        return {
            "is_new_device": int(ev["device_id"] not in known_dev),
            "is_new_ip": int(ev["ip"] not in known_ip),
            "is_new_country": int(ev["country"] not in known_cty),
            "distance_from_home_km": round(haversine_km(_coords(u["home_city"]), _coords(ev["city"])), 1),
            "max_travel_speed_kmh": round(speed, 1),
            "ip_abuse_score": ip["abuse_score"],
            "ip_is_anonymizer": int(ip["kind"] in ("tor", "vpn", "datacenter")),
            "failed_logins_24h": sum(e["event_type"] == "login_failed" for e in recent),
            "password_change_24h": int(any(e["event_type"] == "password_change" for e in recent)),
            "amount_ratio": round(ev["amount"] / max(median_amt, 1), 2),
            "txn_count_1h": len(last_hour_tx),
            "distinct_merchants_1h": len({e["merchant"] for e in last_hour_tx}),
            "small_txn_count_1h": sum(e["amount"] <= 20_000 for e in last_hour_tx),
            "inbound_count_24h": len(inbound),
            "outflow_ratio_24h": round(out_sum / inbound_sum, 2) if inbound_sum else 0.0,
            "is_new_beneficiary": int(ev["event_type"] == "payout" and ev["beneficiary_id"] not in known_ben),
            "account_age_days": u["account_age_days"],
            "hour_deviation": min(hd, 24 - hd),
            "is_new_merchant": int(ev["event_type"] == "transaction" and ev["merchant"] not in known_merch),
        }

    def travel_check(self, user_id: str, until: datetime, hours: int = 48) -> dict:
        win = [e for e in self.by_user[user_id]
               if until - timedelta(hours=hours) <= e["_t"] <= until and e["event_type"] in ("login", "transaction", "payout")]
        hops = []
        for a, b in zip(win, win[1:]):
            if a["city"] == b["city"]:
                continue
            d = haversine_km(_coords(a["city"]), _coords(b["city"]))
            dt_h = max((b["_t"] - a["_t"]).total_seconds() / 3600, 0.25)
            hops.append({
                "from": f'{a["city"]} ({a["country"]}) @ {a["ts"]}', "to": f'{b["city"]} ({b["country"]}) @ {b["ts"]}',
                "distance_km": round(d), "hours_between": round(dt_h, 2), "implied_speed_kmh": round(d / dt_h),
                "same_device": a["device_id"] == b["device_id"],
                "verdict": "IMPOSSIBLE (> 1000 km/h)" if d / dt_h > 1000 else "plausible by air" if d > 300 else "plausible",
            })
        return {"window_hours": hours, "location_changes": hops,
                "impossible_travel": any(h["implied_speed_kmh"] > 1000 for h in hops)}

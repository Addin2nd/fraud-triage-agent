"""Command-line entry point.

    python -m triage_agent setup                      # generate data + train model
    python -m triage_agent queue                      # list open alerts
    python -m triage_agent triage ALT-00007           # investigate one alert (offline playbook)
    python -m triage_agent triage ALT-00007 --provider anthropic
    python -m triage_agent eval --provider gemini --limit 50
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_HISTORY, DATA_LIVE, MODELS, RESULTS = ROOT / "data/history", ROOT / "data/live", ROOT / "models", ROOT / "results"


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if path.exists():
        for line in path.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                v = v.split(" #", 1)[0].strip().strip('"').strip("'")
                if v:
                    os.environ.setdefault(k.strip(), v)


def build_triage(provider: str | None, model: str | None = None, audit: bool = True):
    from .agent import LLMAgent, PlaybookAgent, Triage
    from .llm import LLMClient
    agent = PlaybookAgent() if provider in (None, "", "offline") else LLMAgent(LLMClient(provider, model=model))
    return Triage(DATA_LIVE, MODELS, agent=agent, audit_log=RESULTS / "audit_log.jsonl" if audit else None)


def cmd_setup(a):
    from .data import generate_dataset
    from .risk_model import train
    print(f"Generating historical case base ({a.history} alerts) ...")
    generate_dataset(DATA_HISTORY, a.history, seed=7, label_noise=0.05)
    print(f"Generating live alert queue ({a.live} alerts) ...")
    generate_dataset(DATA_LIVE, a.live, seed=2026)
    print("Training risk model ...")
    print(json.dumps(train(DATA_HISTORY, MODELS), indent=2))


def cmd_queue(a):
    alerts = json.loads((DATA_LIVE / "alerts.json").read_text())
    for al in alerts[: a.limit]:
        print(f"{al['alert_id']}  {al['created_at']}  {al['rule']:<28} {al['user_id']}  {al['event_type']:<12} IDR {al['amount']:>12,}")
    print(f"... {len(alerts)} alerts total")


def cmd_triage(a):
    from .report import render_markdown
    t = build_triage(a.provider, a.model)
    res = t.run(a.alert_id)
    if a.json:
        print(res.model_dump_json(indent=2))
    else:
        print(render_markdown(res, t.store.alerts[a.alert_id]))


def cmd_eval(a):
    from .evaluate import evaluate, to_markdown
    t = build_triage(a.provider, a.model, audit=False)
    s = evaluate(t, DATA_LIVE / "labels.json", limit=a.limit, workers=a.workers, out_dir=RESULTS, tag=a.tag)
    print(to_markdown(s))


def main(argv=None):
    load_dotenv()
    p = argparse.ArgumentParser(prog="triage_agent", description="AI fraud-alert triage agent")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("setup", help="generate synthetic data and train the risk model")
    s.add_argument("--history", type=int, default=2000)
    s.add_argument("--live", type=int, default=200)
    s.set_defaults(fn=cmd_setup)
    s = sub.add_parser("queue", help="list alerts")
    s.add_argument("--limit", type=int, default=25)
    s.set_defaults(fn=cmd_queue)
    for name, fn in (("triage", cmd_triage), ("eval", cmd_eval)):
        s = sub.add_parser(name)
        if name == "triage":
            s.add_argument("alert_id")
            s.add_argument("--json", action="store_true")
        else:
            s.add_argument("--limit", type=int, default=None)
            s.add_argument("--workers", type=int, default=1)
            s.add_argument("--tag", default=None)
        s.add_argument("--provider", default=os.getenv("LLM_PROVIDER", "offline"),
                       help="offline | anthropic | openai | gemini | groq | openrouter | ollama")
        s.add_argument("--model", default=None)
        s.set_defaults(fn=fn)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    sys.exit(main())

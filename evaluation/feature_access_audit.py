from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import pandas as pd

from finauth_audit.baselines.rules import AuthorizationRule, phase1_rules


ROOT = Path(__file__).resolve().parents[1]


def load_manifest(path: Path | None = None) -> dict[str, object]:
    target = path or ROOT / "manifests" / "feature_access.json"
    return json.loads(target.read_text(encoding="utf-8"))


def audit_rule(
    rule: AuthorizationRule,
    task: str,
    manifest: dict[str, object],
) -> dict[str, object]:
    tasks = manifest["tasks"]
    if task not in tasks:
        raise KeyError(f"unknown task {task}")
    used = set(rule.features_used)
    legal = set(tasks[task]["legal"])
    forbidden = set(manifest["global_forbidden"])
    identifiers = set(manifest["non_decision_identifiers"])
    illegal = sorted(used.intersection(forbidden | identifiers) | (used - legal))
    return {
        "rule": rule.name,
        "task": task,
        "classification": rule.classification,
        "features_used": sorted(used),
        "illegal_features": illegal,
        "status": "VALID" if not illegal else "INVALID",
    }


def audit_rules(
    rules: Iterable[AuthorizationRule],
    task: str,
    manifest: dict[str, object] | None = None,
) -> pd.DataFrame:
    payload = manifest or load_manifest()
    return pd.DataFrame([audit_rule(rule, task, payload) for rule in rules])


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit deployable baseline feature access.")
    parser.add_argument("--task", default="coverage")
    parser.add_argument("--out", default=str(ROOT / "results" / "feature_access_audit.csv"))
    args = parser.parse_args()
    report = audit_rules(phase1_rules(), args.task)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    print(out)
    return 1 if (report["status"] == "INVALID").any() else 0


if __name__ == "__main__":
    raise SystemExit(main())

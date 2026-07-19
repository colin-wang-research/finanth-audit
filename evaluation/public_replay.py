from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from finauth_audit.baselines.public_rules import public_rule_registry
from finauth_audit.evaluation.certification_surface import build_surface
from finauth_audit.evaluation.cluster_bootstrap import cluster_bootstrap_bounds
from finauth_audit.evaluation.feature_access_audit import audit_rules
from finauth_audit.evaluation.metrics import profile_metrics, with_decision_outcomes
from finauth_audit.evaluation.seeds import derive_seed


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profiles(frame: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {
        "overall": frame,
        "stress": frame[frame["stress_tag"] != "ordinary"],
    }


def _evaluate(
    frame: pd.DataFrame,
    rules: list[object],
    config: dict[str, Any],
    *,
    namespace: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summaries: list[dict[str, object]] = []
    bounds_rows: list[dict[str, object]] = []
    decisions: list[pd.DataFrame] = []
    for rule in rules:
        ruled = with_decision_outcomes(frame, rule.decide(frame, {}))
        ruled.insert(0, "rule", rule.name)
        decisions.append(ruled)
        overall = {"rule": rule.name, **profile_metrics(ruled)}
        summaries.append(overall)
        for profile, subset in _profiles(ruled).items():
            if subset.empty:
                continue
            bounds = cluster_bootstrap_bounds(
                subset,
                replicates=int(config["bootstrap_replicates"]),
                seed=derive_seed(
                    int(config["bootstrap_seed"]),
                    f"public-transfer/{namespace}/{rule.name}/{profile}",
                ),
            )
            bounds.update({"rule": rule.name, "profile": profile})
            bounds_rows.append(bounds)
    bounds_frame = pd.DataFrame(bounds_rows)
    surface, certification = build_surface(bounds_frame)
    return (
        pd.DataFrame(summaries),
        bounds_frame,
        surface,
        certification,
    )


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    results_dir = ROOT / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    public_frame = pd.read_csv(ROOT / config["polymarket"]["dataset_path"], low_memory=False)
    public_validation = public_frame[public_frame["split"] == config["evaluation_split"]].copy()
    if public_validation.empty:
        raise ValueError("public validation split is empty")
    controlled = pd.read_csv(ROOT / "data" / "controlled_core" / "smoke.csv", low_memory=False)
    controlled_validation = controlled[controlled["split"] == config["evaluation_split"]].copy()
    rules = public_rule_registry(config)
    access = audit_rules(rules, "public_transfer")
    if (access["status"] == "INVALID").any():
        raise RuntimeError(access.loc[access["status"] == "INVALID"].to_dict("records"))

    outputs: dict[str, pd.DataFrame] = {"feature_access_audit.csv": access}
    for namespace, frame in (
        ("public_validation", public_validation),
        ("controlled_validation", controlled_validation),
    ):
        summary, bounds, surface, certification = _evaluate(
            frame, rules, config, namespace=namespace
        )
        outputs[f"{namespace}_summary.csv"] = summary
        outputs[f"{namespace}_bootstrap_bounds.csv"] = bounds
        outputs[f"{namespace}_certification_surface.csv"] = surface
        outputs[f"{namespace}_certification_summary.csv"] = certification
    for name, frame in outputs.items():
        frame.to_csv(results_dir / name, index=False)

    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.2.0",
        "evaluation_split": config["evaluation_split"],
        "confirmatory": False,
        "public_rows_evaluated": len(public_validation),
        "public_clusters_evaluated": int(public_validation["event_cluster_id"].nunique()),
        "controlled_rows_evaluated": len(controlled_validation),
        "controlled_clusters_evaluated": int(controlled_validation["event_cluster_id"].nunique()),
        "rule_configurations": len(rules),
        "bootstrap_unit": "event_cluster_id",
        "bootstrap_replicates": int(config["bootstrap_replicates"]),
        "outputs": {name: sha256(results_dir / name) for name in outputs},
        "claim_boundary": (
            "Validation-only public-transfer preparation. Public test outcomes are not evaluated."
        ),
    }
    (results_dir / "public_validation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(results_dir)
    return results_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation-only public replay rules.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "public_audit.yaml"))
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

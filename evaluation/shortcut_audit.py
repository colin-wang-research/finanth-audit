from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from finauth_audit.baselines.rules import phase1_rules
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


def _shuffle_columns(frame: pd.DataFrame, columns: list[str], rng: np.random.Generator) -> pd.DataFrame:
    result = frame.copy()
    order = rng.permutation(len(result))
    result.loc[:, columns] = result.iloc[order][columns].to_numpy()
    return result


def perturbations(frame: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    numeric_medians = frame.median(numeric_only=True)
    variants = {
        "original": frame.copy(),
        "permute_confidence": _shuffle_columns(frame, ["confidence"], rng),
        "permute_cost_bundle": _shuffle_columns(
            frame,
            ["liquidity_cost_bps", "turnover_cost_bps", "fee_bps", "volatility_proxy"],
            rng,
        ),
        "permute_role": _shuffle_columns(frame, ["source_role"], rng),
        "mask_confidence": frame.assign(confidence=float(numeric_medians["confidence"])),
        "mask_uncertainty": frame.assign(uncertainty=float(numeric_medians["uncertainty"])),
        "mask_expected_edge": frame.assign(expected_edge_bps=float(numeric_medians["expected_edge_bps"])),
        "mask_cost": frame.assign(
            liquidity_cost_bps=float(numeric_medians["liquidity_cost_bps"]),
            turnover_cost_bps=float(numeric_medians["turnover_cost_bps"]),
            fee_bps=float(numeric_medians["fee_bps"]),
        ),
        "mask_role": frame.assign(source_role="edge_proposer"),
        "mask_source_name": frame.assign(current_source_id="masked_source"),
        "mask_stress_tag": frame.assign(stress_tag="masked_stress"),
    }
    return variants


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_path = ROOT / config["output_data"]
    frame = pd.read_csv(data_path)
    evaluated = frame[frame["split"] == config["evaluation_split"]].copy()
    evaluated = evaluated[evaluated["certification_eligible"]].copy()
    results_dir = ROOT / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)

    rules = phase1_rules()
    access = audit_rules(rules, "coverage")
    if (access["status"] == "INVALID").any():
        raise RuntimeError("shortcut audit refuses INVALID deployable rules")

    rows: list[dict[str, object]] = []
    originals: dict[str, np.ndarray] = {}
    shortcut_seed = derive_seed(int(config["bootstrap_seed"]), "shortcut/component-ablation")
    for variant_name, variant in perturbations(evaluated, shortcut_seed).items():
        for rule in rules:
            decisions = rule.decide(variant, config["thresholds"])
            if variant_name == "original":
                originals[rule.name] = decisions.copy()
            metrics = profile_metrics(with_decision_outcomes(variant, decisions))
            rows.append(
                {
                    "variant": variant_name,
                    "rule": rule.name,
                    "decision_change_rate": 0.0
                    if variant_name == "original"
                    else float(np.mean(decisions != originals[rule.name])),
                    **metrics,
                }
            )
    ablations = pd.DataFrame(rows)
    ablation_path = results_dir / "shortcut_component_ablation.csv"
    ablations.to_csv(ablation_path, index=False)

    cluster_split_max = int(frame.groupby("event_cluster_id")["split"].nunique().max())
    structural = {
        "duplicate_row_ids": int(frame["row_id"].duplicated().sum()),
        "cluster_split_max_unique": cluster_split_max,
        "cluster_split_leakage": cluster_split_max > 1,
        "future_decoy_present": "future_decoy" in frame.columns,
        "future_decoy_used_by_deployable_rule": any(
            "future_decoy" in rule.features_used for rule in rules
        ),
        "identifier_used_by_deployable_rule": bool(
            set().union(*(set(rule.features_used) for rule in rules))
            & {"row_id", "event_cluster_id", "current_source_id", "stress_tag"}
        ),
        "valid_rule_count": int((access["status"] == "VALID").sum()),
        "invalid_rule_count": int((access["status"] == "INVALID").sum()),
    }
    structural_path = results_dir / "shortcut_structural_audit.json"
    structural_path.write_text(
        json.dumps(structural, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if structural["cluster_split_leakage"] or structural["future_decoy_used_by_deployable_rule"]:
        raise RuntimeError(f"shortcut structural audit failed: {structural}")
    manifest = {
        "mode": config["mode"],
        "confirmatory": False,
        "seed": shortcut_seed,
        "outputs": {
            ablation_path.name: sha256(ablation_path),
            structural_path.name: sha256(structural_path),
        },
        "claim_boundary": "Validation-only shortcut/component audit; no test claim.",
    }
    (results_dir / "shortcut_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run FinAuth-Audit shortcut and component ablations.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.yaml"))
    args = parser.parse_args()
    out = run(Path(args.config))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

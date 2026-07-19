from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr

from finauth_audit.evaluation.seeds import derive_seed


ROOT = Path(__file__).resolve().parents[1]


def estimate_rank_power(
    rule_count: int,
    target_rho: float,
    attenuation: float,
    alpha: float,
    replicates: int,
    seed: int,
) -> float:
    if rule_count < 4:
        return 0.0
    effective_rho = float(np.clip(target_rho * attenuation, -0.99, 0.99))
    covariance = np.asarray([[1.0, effective_rho], [effective_rho, 1.0]])
    rng = np.random.default_rng(seed)
    successes = 0
    for _ in range(replicates):
        sample = rng.multivariate_normal([0.0, 0.0], covariance, size=rule_count)
        correlation, p_value = spearmanr(sample[:, 0], sample[:, 1])
        if np.isfinite(correlation) and correlation > 0 and p_value < alpha:
            successes += 1
    return successes / replicates


def _attenuation(bounds: pd.DataFrame, active_rules: set[str]) -> tuple[float, float, float]:
    selected = bounds[
        (bounds["profile"] == "overall") & bounds["rule"].isin(active_rules)
    ].copy()
    midpoint = 0.5 * (selected["cau_lcb95"] + selected["cau_ucb95"])
    standard_error = (selected["cau_ucb95"] - selected["cau_lcb95"]) / (2.0 * 1.645)
    signal = float(midpoint.std(ddof=1)) if len(midpoint) > 1 else 0.0
    noise = float(standard_error.median()) if len(standard_error) else float("inf")
    if signal <= 0 or not np.isfinite(noise):
        return 0.0, signal, noise
    attenuation = float(signal / np.sqrt(signal**2 + noise**2))
    return attenuation, signal, noise


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    gate = config["power_gate"]
    results_dir = ROOT / config["results_dir"]
    dataset = pd.read_csv(ROOT / config["polymarket"]["dataset_path"], low_memory=False)
    checks = pd.read_csv(results_dir / "point_in_time_checks.csv")
    summary = pd.read_csv(results_dir / "public_validation_summary.csv")
    bounds = pd.read_csv(results_dir / "public_validation_bootstrap_bounds.csv")
    access = pd.read_csv(results_dir / "feature_access_audit.csv")

    deployable = set(access.loc[access["classification"] == "deployable", "rule"])
    active = summary[
        summary["rule"].isin(deployable)
        & (summary["coverage"] >= float(gate["min_validation_coverage"]))
    ]
    active_rules = set(active["rule"])
    attenuation, signal, noise = _attenuation(bounds, active_rules)
    power = estimate_rank_power(
        len(active_rules),
        float(gate["target_absolute_spearman"]),
        attenuation,
        float(gate["alpha"]),
        int(gate["simulation_replicates"]),
        derive_seed(int(config["seed"]), "public-rank-power"),
    )
    simulation_replicates = int(gate["simulation_replicates"])
    monte_carlo_se = float(np.sqrt(power * (1.0 - power) / simulation_replicates))
    power_lcb95 = float(max(0.0, power - 1.96 * monte_carlo_se))
    test_clusters = int(dataset.loc[dataset["split"] == "test", "event_cluster_id"].nunique())
    conditions = {
        "point_in_time_checks": bool(checks["passed"].all()),
        "test_clusters_per_source": test_clusters
        >= int(gate["min_test_clusters_per_confirmatory_source"]),
        "test_clusters_combined": test_clusters >= int(gate["min_test_clusters_combined"]),
        "active_rule_configurations": len(active_rules) >= int(gate["min_active_rule_configs"]),
        "validation_power": power_lcb95 >= float(gate["min_power"]),
        "feature_access": bool((access["status"] == "VALID").all()),
    }
    passed = all(conditions.values())
    payload: dict[str, Any] = {
        "project": "FinAuth-Audit",
        "version": "0.2.0",
        "confirmatory_test_evaluated": False,
        "source": "polymarket_point_in_time",
        "test_clusters": test_clusters,
        "active_rule_configurations": len(active_rules),
        "active_rules": sorted(active_rules),
        "target_absolute_spearman": float(gate["target_absolute_spearman"]),
        "alpha": float(gate["alpha"]),
        "simulation_replicates": simulation_replicates,
        "cluster_uncertainty_attenuation": attenuation,
        "between_rule_cau_signal": signal,
        "median_cluster_bootstrap_cau_se": noise,
        "estimated_power": power,
        "monte_carlo_standard_error": monte_carlo_se,
        "power_lcb95": power_lcb95,
        "conditions": conditions,
        "passed": passed,
        "classification": "confirmatory_eligible" if passed else "exploratory_only",
        "claim_boundary": (
            "This gate uses validation coverage, structural test-cluster counts, and "
            "cluster-bootstrap uncertainty. It does not inspect public test outcomes and "
            "does not produce row-level p-values."
        ),
    }
    output = results_dir / "public_power_gate.json"
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = [
        "# Public Replay Power Gate",
        "",
        f"- Classification: **{payload['classification']}**",
        f"- Test event clusters (structural count only): {test_clusters}",
        f"- Active deployable rule configurations: {len(active_rules)}",
        f"- Cluster-uncertainty attenuation: {attenuation:.3f}",
        f"- Validation-only estimated power: {power:.3f}",
        f"- Monte Carlo lower 95% bound: {power_lcb95:.3f}",
        "",
        "No public test outcome metric was evaluated.",
        "",
    ]
    (results_dir / "public_power_gate.md").write_text("\n".join(report), encoding="utf-8")
    print(output)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation-only public rank-power gate.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "public_audit.yaml"))
    args = parser.parse_args()
    run(Path(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

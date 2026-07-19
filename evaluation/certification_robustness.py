from __future__ import annotations

import argparse
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]

# These profiles are frozen in the Round 75 pre-test amendment. They are
# validation-only diagnostics and may not be changed after paper-test access.
POLICY_GRIDS: dict[str, dict[str, tuple[float, ...]]] = {
    "frozen": {
        "far": (0.05, 0.10, 0.15, 0.20, 0.25, 0.30),
        "alr": (0.00, 0.01, 0.02, 0.05),
        "coverage": (0.05, 0.10, 0.20, 0.30, 0.50),
    },
    "conservative_institution": {
        "far": (0.05, 0.10, 0.15),
        "alr": (0.00, 0.005, 0.01),
        "coverage": (0.10, 0.20, 0.30, 0.50),
    },
    "balanced_platform": {
        "far": (0.10, 0.20, 0.30, 0.40),
        "alr": (0.00, 0.02, 0.05, 0.10),
        "coverage": (0.10, 0.20, 0.40, 0.60),
    },
    "coverage_priority_research": {
        "far": (0.15, 0.30, 0.45, 0.60),
        "alr": (0.00, 0.05, 0.10, 0.20),
        "coverage": (0.20, 0.40, 0.60, 0.75),
    },
    "lineage_integrity_first": {
        "far": (0.10, 0.20, 0.30),
        "alr": (0.00, 0.005, 0.01, 0.02),
        "coverage": (0.05, 0.10, 0.20, 0.30),
    },
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def continuous_hypervolume(
    far_ucb95: float,
    alr_ucb95: float,
    coverage_lcb95: float,
    *,
    far_reference: float = 1.0,
    alr_reference: float = 0.25,
) -> float:
    values = (far_ucb95, alr_ucb95, coverage_lcb95)
    if not all(np.isfinite(value) for value in values):
        return 0.0
    far_span = np.clip((far_reference - far_ucb95) / far_reference, 0.0, 1.0)
    alr_span = np.clip((alr_reference - alr_ucb95) / alr_reference, 0.0, 1.0)
    coverage_span = np.clip(coverage_lcb95, 0.0, 1.0)
    return float(far_span * alr_span * coverage_span)


def grid_volume(record: dict[str, object], grid: dict[str, tuple[float, ...]]) -> float:
    required = (
        float(record["far_ucb95"]),
        float(record["alr_ucb95"]),
        float(record["coverage_lcb95"]),
    )
    if not all(np.isfinite(value) for value in required):
        return 0.0
    passed = 0
    total = 0
    for far_limit, alr_limit, coverage_floor in itertools.product(
        grid["far"], grid["alr"], grid["coverage"]
    ):
        total += 1
        passed += int(
            required[0] <= far_limit
            and required[1] <= alr_limit
            and required[2] >= coverage_floor
        )
    return passed / total


def pareto_flags(frame: pd.DataFrame) -> pd.Series:
    objectives = frame[["coverage", "far", "alr", "moc"]].copy()
    valid = objectives.notna().all(axis=1)
    efficient = pd.Series(False, index=frame.index, dtype=bool)
    for index in frame.index[valid]:
        candidate = objectives.loc[index]
        others = objectives.loc[valid & (objectives.index != index)]
        dominates = (
            (others["coverage"] >= candidate["coverage"])
            & (others["far"] <= candidate["far"])
            & (others["alr"] <= candidate["alr"])
            & (others["moc"] <= candidate["moc"])
            & (
                (others["coverage"] > candidate["coverage"])
                | (others["far"] < candidate["far"])
                | (others["alr"] < candidate["alr"])
                | (others["moc"] < candidate["moc"])
            )
        )
        efficient.loc[index] = not bool(dominates.any())
    return efficient


def _render_report(
    policy_summary: pd.DataFrame,
    continuous: pd.DataFrame,
    pareto: pd.DataFrame,
    correlations: pd.DataFrame,
) -> str:
    lines = [
        "# Certification Robustness Report",
        "",
        "This validation-only report tests whether the frozen discrete certification result survives alternative pre-test risk profiles and a continuous conservative hypervolume. It is not an institutional threshold study and does not use test outcomes.",
        "",
        "## Policy-grid sensitivity",
        "",
        "| Rule | Mean volume | Minimum volume | Maximum volume | Rank range |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in policy_summary.sort_values(["mean_volume", "rule"], ascending=[False, True]).to_dict(orient="records"):
        lines.append(
            f"| {row['rule']} | {row['mean_volume']:.3f} | {row['min_volume']:.3f} | "
            f"{row['max_volume']:.3f} | {int(row['rank_range'])} |"
        )
    lines.extend(
        [
            "",
            "## Continuous conservative hypervolume",
            "",
            "The hypervolume is the rectangular safe region between each rule's one-sided FAR/ALR bounds, coverage lower bound, and the fixed reference point FAR=1, ALR=0.25, coverage=0. It is descriptive and does not compress utility or review cost into profit.",
            "",
            "| Rule | Overall | Stress | Worst profile |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in continuous.sort_values(["worst_profile_hypervolume", "rule"], ascending=[False, True]).to_dict(orient="records"):
        lines.append(
            f"| {row['rule']} | {row.get('overall', 0.0):.4f} | {row.get('stress', 0.0):.4f} | "
            f"{row['worst_profile_hypervolume']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Multi-objective Pareto set",
            "",
            "| Rule | Pareto efficient | Coverage | FAR | ALR | MOC |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in pareto.sort_values(["pareto_efficient", "coverage"], ascending=[False, False]).to_dict(orient="records"):
        far = "N/A" if pd.isna(row["far"]) else f"{row['far']:.3f}"
        alr = "N/A" if pd.isna(row["alr"]) else f"{row['alr']:.3f}"
        lines.append(
            f"| {row['rule']} | {'yes' if row['pareto_efficient'] else 'no'} | "
            f"{row['coverage']:.3f} | {far} | {alr} | {row['moc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Rank correlations",
            "",
            "| Policy A | Policy B | Spearman rho |",
            "|---|---|---:|",
        ]
    )
    for row in correlations.to_dict(orient="records"):
        rho = "N/A" if pd.isna(row["spearman_rho"]) else f"{row['spearman_rho']:.3f}"
        lines.append(f"| {row['policy_a']} | {row['policy_b']} | {rho} |")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "Alternative threshold profiles are author-specified scenario analyses frozen before paper-test access. They do not substitute for thresholds elicited from practitioners. A rule is not globally optimal merely because it has high grid volume or hypervolume.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("evaluation_split") != "validation":
        raise RuntimeError("certification robustness is frozen as validation-only before paper-test access")
    source_dir = ROOT / config["results_dir"]
    bounds = pd.read_csv(source_dir / "bootstrap_bounds.csv")
    metrics = pd.read_csv(source_dir / "metrics.csv")
    if set(bounds["profile"]) < {"overall", "stress"}:
        raise ValueError("overall and stress bounds are required")

    policy_rows: list[dict[str, object]] = []
    for policy_name, grid in POLICY_GRIDS.items():
        for record in bounds[bounds["profile"].isin(["overall", "stress"])].to_dict(orient="records"):
            policy_rows.append(
                {
                    "policy": policy_name,
                    "rule": record["rule"],
                    "profile": record["profile"],
                    "certification_volume": grid_volume(record, grid),
                }
            )
    policy_profiles = pd.DataFrame(policy_rows)
    policy_volumes = (
        policy_profiles.groupby(["policy", "rule"], as_index=False)["certification_volume"]
        .min()
        .rename(columns={"certification_volume": "worst_profile_volume"})
    )
    policy_volumes["rank"] = policy_volumes.groupby("policy")["worst_profile_volume"].rank(
        method="min", ascending=False
    )
    policy_summary = (
        policy_volumes.groupby("rule", as_index=False)
        .agg(
            mean_volume=("worst_profile_volume", "mean"),
            min_volume=("worst_profile_volume", "min"),
            max_volume=("worst_profile_volume", "max"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
        )
    )
    policy_summary["rank_range"] = policy_summary["max_rank"] - policy_summary["min_rank"]

    hyper_rows = bounds[bounds["profile"].isin(["overall", "stress"])].copy()
    hyper_rows["continuous_hypervolume"] = hyper_rows.apply(
        lambda row: continuous_hypervolume(
            float(row["far_ucb95"]),
            float(row["alr_ucb95"]),
            float(row["coverage_lcb95"]),
        ),
        axis=1,
    )
    continuous = hyper_rows.pivot(
        index="rule", columns="profile", values="continuous_hypervolume"
    ).reset_index()
    for profile in ("overall", "stress"):
        if profile not in continuous:
            continuous[profile] = 0.0
    continuous["worst_profile_hypervolume"] = continuous[["overall", "stress"]].min(axis=1)

    pareto = metrics[metrics["profile"] == "overall"][
        ["rule", "coverage", "far", "alr", "cau", "moc", "review_rate"]
    ].copy()
    pareto["pareto_efficient"] = pareto_flags(pareto)

    rank_pivot = policy_volumes.pivot(index="rule", columns="policy", values="rank")
    correlation_rows: list[dict[str, object]] = []
    for policy_a, policy_b in itertools.combinations(rank_pivot.columns, 2):
        left = rank_pivot[policy_a]
        right = rank_pivot[policy_b]
        if left.nunique() < 2 or right.nunique() < 2:
            rho = np.nan
            reason = "constant_rank"
        else:
            rho = spearmanr(left, right).statistic
            reason = "defined"
        correlation_rows.append(
            {
                "policy_a": policy_a,
                "policy_b": policy_b,
                "spearman_rho": float(rho),
                "status": reason,
            }
        )
    correlations = pd.DataFrame(correlation_rows)

    frozen_surface = pd.read_csv(source_dir / "certification_surface.csv")
    threshold_rows: list[pd.DataFrame] = []
    for dimension in ("tau_h", "tau_a", "c_min"):
        grouped = (
            frozen_surface.groupby(["rule", "profile", dimension], as_index=False)["passed"]
            .mean()
            .rename(columns={"passed": "pass_fraction", dimension: "threshold"})
        )
        grouped.insert(0, "dimension", dimension)
        threshold_rows.append(grouped)
    threshold_sensitivity = pd.concat(threshold_rows, ignore_index=True)

    output_dir = ROOT / "results" / "certification_robustness"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "policy_profile_volumes.csv": policy_profiles,
        "policy_volumes.csv": policy_volumes,
        "policy_summary.csv": policy_summary,
        "continuous_hypervolume.csv": continuous,
        "pareto_frontier.csv": pareto,
        "rank_correlations.csv": correlations,
        "threshold_sensitivity.csv": threshold_sensitivity,
    }
    for name, frame in outputs.items():
        frame.to_csv(output_dir / name, index=False)
    report_path = output_dir / "report.md"
    report_path.write_text(
        _render_report(policy_summary, continuous, pareto, correlations), encoding="utf-8"
    )
    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.2.0-round75-validation-robustness",
        "evaluation_split": "validation",
        "test_outcomes_evaluated": False,
        "policy_grids": {name: {key: list(values) for key, values in grid.items()} for name, grid in POLICY_GRIDS.items()},
        "continuous_reference": {"far": 1.0, "alr": 0.25, "coverage": 0.0},
        "inputs": {
            "config": sha256(config_path),
            "bootstrap_bounds.csv": sha256(source_dir / "bootstrap_bounds.csv"),
            "metrics.csv": sha256(source_dir / "metrics.csv"),
            "certification_surface.csv": sha256(source_dir / "certification_surface.csv"),
        },
        "outputs": {
            **{name: sha256(output_dir / name) for name in outputs},
            "report.md": sha256(report_path),
        },
        "claim_boundary": "Validation-only threshold and Pareto sensitivity; no institutional preference, test, deployment, or global-winner claim.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation-only certification robustness diagnostics.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "main.yaml"))
    args = parser.parse_args()
    print(run(Path(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

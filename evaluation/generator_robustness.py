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

from finauth_audit.baselines.rules import phase1_rules
from finauth_audit.evaluation.certification_surface import build_surface
from finauth_audit.evaluation.cluster_bootstrap import cluster_bootstrap_bounds
from finauth_audit.evaluation.metrics import profile_metrics, with_decision_outcomes
from finauth_audit.evaluation.seeds import derive_seed


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _read_validation(path: Path, family: str) -> pd.DataFrame:
    frame = pd.read_csv(path, low_memory=False)
    selected = frame[frame["split"] == "validation"].copy()
    if selected.empty:
        raise ValueError(f"validation split is empty: {path}")
    selected["generator_family"] = family
    return selected


def _profile(frame: pd.DataFrame, profile: str) -> pd.DataFrame:
    scored = frame[frame["certification_eligible"]].copy()
    if profile == "overall":
        return scored
    if profile == "stress":
        return scored[scored["opportunity_slice"] != "ordinary"].copy()
    raise KeyError(profile)


def _render_report(
    summary: pd.DataFrame,
    stability: pd.DataFrame,
    interactions: pd.DataFrame,
    finding: dict[str, object],
) -> str:
    lines = [
        "# Mechanistic Generator Robustness Report",
        "",
        "The registered rule implementations and thresholds are transported unchanged from the utility-iid controlled layer to two prospectively defined generator families. This is validation-only cross-mechanism evidence, not an independent external replication.",
        "",
        "## Rule by generator",
        "",
        "| Generator | Rule | Coverage | FAR | ALR | Review | CAU | Certification |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["generator_family", "worst_profile_certification_volume", "rule"], ascending=[True, False, True]).to_dict(orient="records"):
        far = "N/A" if pd.isna(row["far"]) else f"{row['far']:.3f}"
        alr = "N/A" if pd.isna(row["alr"]) else f"{row['alr']:.3f}"
        lines.append(
            f"| {row['generator_family']} | {row['rule']} | {row['coverage']:.3f} | {far} | {alr} | "
            f"{row['review_rate']:.3f} | {row['cau']:.3f} | {row['worst_profile_certification_volume']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Rank stability",
            "",
            "| Metric | Generator A | Generator B | Spearman rho | Status |",
            "|---|---|---|---:|---|",
        ]
    )
    for row in stability.to_dict(orient="records"):
        rho = "N/A" if pd.isna(row["spearman_rho"]) else f"{row['spearman_rho']:.3f}"
        lines.append(
            f"| {row['metric']} | {row['generator_a']} | {row['generator_b']} | {rho} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Rule-by-generator interaction",
            "",
            "| Rule | Coverage range | FAR range | Certification range |",
            "|---|---:|---:|---:|",
        ]
    )
    for row in interactions.sort_values(["certification_range", "far_range"], ascending=False).to_dict(orient="records"):
        lines.append(
            f"| {row['rule']} | {row['coverage_range']:.3f} | {row['far_range']:.3f} | "
            f"{row['certification_range']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Frozen finding summary",
            "",
            f"- Generator families evaluated: {finding['generator_count']}.",
            f"- Families where the lowest-FAR rule differs from the highest-certification rule: {finding['rank_reversal_family_count']}.",
            f"- Families with at least one nonzero certified rule: {finding['families_with_certification']}.",
            "",
            "## Boundary",
            "",
            "The two added families are mechanistically distinct but authored in the same repository. They reduce dependence on one data-generating mechanism but do not substitute for an external implementation or observed financial institution data.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("evaluation_split") != "validation":
        raise RuntimeError("generator robustness must remain validation-only")
    generator_manifest_path = ROOT / config["output_manifest"]
    generator_manifest = json.loads(generator_manifest_path.read_text(encoding="utf-8"))
    if generator_manifest.get("test_outcomes_evaluated") is not False:
        raise RuntimeError("generator manifest does not preserve test sealing")
    families = {
        "utility_iid": ROOT / "data" / "controlled_core" / "main.csv",
        **{
            name: ROOT / details["data_path"]
            for name, details in generator_manifest["families"].items()
        },
    }
    rules = phase1_rules()
    metric_rows: list[dict[str, object]] = []
    bound_rows: list[dict[str, object]] = []
    for family_index, (family, path) in enumerate(families.items()):
        frame = _read_validation(path, family)
        for rule_index, rule in enumerate(rules):
            decisions = rule.decide(frame, config["thresholds"])
            ruled = with_decision_outcomes(frame, decisions)
            for profile in ("overall", "stress"):
                current = _profile(ruled, profile)
                point = profile_metrics(current)
                point.update({"generator_family": family, "rule": rule.name, "profile": profile})
                metric_rows.append(point)
                bound = cluster_bootstrap_bounds(
                    current,
                    replicates=int(config["bootstrap_replicates"]),
                    seed=derive_seed(
                        int(config["families"]["sequential_market"]["seed"]),
                        f"generator-robustness/{family_index}/{rule_index}/{profile}",
                    ),
                    chunk_size=int(config.get("bootstrap_chunk_size", 64)),
                )
                bound.update({"generator_family": family, "rule": rule.name, "profile": profile})
                bound_rows.append(bound)
    metrics = pd.DataFrame(metric_rows)
    bounds = pd.DataFrame(bound_rows)
    certification_rows: list[pd.DataFrame] = []
    surface_rows: list[pd.DataFrame] = []
    for family, group in bounds.groupby("generator_family", sort=True):
        surface, certification = build_surface(group.drop(columns=["generator_family"]))
        surface.insert(0, "generator_family", family)
        certification.insert(0, "generator_family", family)
        surface_rows.append(surface)
        certification_rows.append(certification)
    surface = pd.concat(surface_rows, ignore_index=True)
    certification = pd.concat(certification_rows, ignore_index=True)
    summary = metrics[metrics["profile"] == "overall"].merge(
        certification, on=["generator_family", "rule"], how="left"
    )
    summary["raw_far_rank"] = summary.groupby("generator_family")["far"].rank(
        method="min", na_option="bottom"
    )
    summary["certification_rank"] = summary.groupby("generator_family")[
        "worst_profile_certification_volume"
    ].rank(method="min", ascending=False)

    stability_rows: list[dict[str, object]] = []
    for metric in ("raw_far_rank", "certification_rank"):
        pivot = summary.pivot(index="rule", columns="generator_family", values=metric)
        for family_a, family_b in itertools.combinations(pivot.columns, 2):
            left = pivot[family_a]
            right = pivot[family_b]
            if left.nunique() < 2 or right.nunique() < 2:
                rho = np.nan
                status = "constant_rank"
            else:
                rho = float(spearmanr(left, right).statistic)
                status = "defined"
            stability_rows.append(
                {
                    "metric": metric,
                    "generator_a": family_a,
                    "generator_b": family_b,
                    "spearman_rho": rho,
                    "status": status,
                }
            )
    stability = pd.DataFrame(stability_rows)
    interactions = (
        summary.groupby("rule", as_index=False)
        .agg(
            min_coverage=("coverage", "min"),
            max_coverage=("coverage", "max"),
            min_far=("far", "min"),
            max_far=("far", "max"),
            min_certification=("worst_profile_certification_volume", "min"),
            max_certification=("worst_profile_certification_volume", "max"),
        )
    )
    interactions["coverage_range"] = interactions["max_coverage"] - interactions["min_coverage"]
    interactions["far_range"] = interactions["max_far"] - interactions["min_far"]
    interactions["certification_range"] = (
        interactions["max_certification"] - interactions["min_certification"]
    )
    rank_reversal_count = 0
    certified_families = 0
    for _, group in summary.groupby("generator_family"):
        valid_far = group[group["far"].notna()]
        lowest_far = set(valid_far.loc[valid_far["far"] == valid_far["far"].min(), "rule"])
        highest_cert = set(
            group.loc[
                group["worst_profile_certification_volume"]
                == group["worst_profile_certification_volume"].max(),
                "rule",
            ]
        )
        rank_reversal_count += int(lowest_far.isdisjoint(highest_cert))
        certified_families += int(group["worst_profile_certification_volume"].max() > 0)
    finding = {
        "generator_count": len(families),
        "rank_reversal_family_count": rank_reversal_count,
        "families_with_certification": certified_families,
    }

    output_dir = ROOT / config["results_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "metrics.csv": metrics,
        "bootstrap_bounds.csv": bounds,
        "certification_surface.csv": surface,
        "certification_summary.csv": certification,
        "summary.csv": summary,
        "rank_stability.csv": stability,
        "rule_generator_interactions.csv": interactions,
    }
    for name, output in outputs.items():
        output.to_csv(output_dir / name, index=False)
    finding_path = output_dir / "finding_summary.json"
    finding_path.write_text(json.dumps(finding, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report_path = output_dir / "report.md"
    report_path.write_text(_render_report(summary, stability, interactions, finding), encoding="utf-8")
    manifest = {
        "project": "FinAuth-Audit",
        "version": config["version"],
        "evaluation_split": "validation",
        "test_outcomes_evaluated": False,
        "unchanged_rule_thresholds": config["thresholds"],
        "inputs": {
            "config": sha256(config_path),
            "generator_manifest": sha256(generator_manifest_path),
            **{family: sha256(path) for family, path in families.items()},
        },
        "outputs": {
            **{name: sha256(output_dir / name) for name in outputs},
            "finding_summary.json": sha256(finding_path),
            "report.md": sha256(report_path),
        },
        "claim_boundary": "Validation-only cross-mechanism robustness across author-implemented generators; not external replication or test evidence.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate unchanged rules across mechanistic generators.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "generator_robustness.yaml"))
    args = parser.parse_args()
    print(run(Path(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


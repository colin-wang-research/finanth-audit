from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from finauth_audit.baselines.rules import phase1_rules
from finauth_audit.evaluation.certification_surface import build_surface
from finauth_audit.evaluation.cluster_bootstrap import cluster_bootstrap_bounds
from finauth_audit.evaluation.feature_access_audit import audit_rules
from finauth_audit.evaluation.metrics import profile_metrics, safe_ratio, with_decision_outcomes
from finauth_audit.evaluation.seeds import derive_seed


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _profile(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    scored = frame[frame["certification_eligible"]].copy()
    if name == "overall":
        return scored
    if name == "ordinary":
        return scored[scored["opportunity_slice"] == "ordinary"].copy()
    if name == "stress":
        return scored[scored["opportunity_slice"] != "ordinary"].copy()
    raise KeyError(name)


def _render_report(
    config: dict[str, object],
    ranking: pd.DataFrame,
    opportunity: pd.DataFrame,
) -> str:
    lines = [
        "# Coverage and Certification Validation Report",
        "",
        f"Mode: `{config['mode']}`",
        "",
        "This is a validation-only development run. It is not a confirmatory test result.",
        "",
        "## Raw versus certified ranking",
        "",
        "| Rule | Coverage | FAR | ALR | Worst-profile certification volume | Collapse index | Relation |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    ordered = ranking.sort_values(
        ["certified_rank", "raw_far_rank", "rule"], na_position="last"
    )
    for row in ordered.to_dict(orient="records"):
        far = "N/A" if pd.isna(row["far"]) else f"{row['far']:.3f}"
        alr = "N/A" if pd.isna(row["alr"]) else f"{row['alr']:.3f}"
        collapse = (
            "N/A"
            if pd.isna(row["coverage_collapse_index"])
            else f"{row['coverage_collapse_index']:.3f}"
        )
        lines.append(
            f"| {row['rule']} | {row['coverage']:.3f} | {far} | {alr} | "
            f"{row['worst_profile_certification_volume']:.3f} | {collapse} | "
            f"{row['coverage_relation']} |"
        )
    lines.extend(
        [
            "",
            "## Opportunity-density audit",
            "",
            "| Slice | Event clusters | Intended rate | Realized oracle-positive rate | Certified profile? |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for row in opportunity.to_dict(orient="records"):
        lines.append(
            f"| {row['opportunity_slice']} | {int(row['event_clusters'])} | "
            f"{row['intended_opportunity_rate']:.3f} | {row['oracle_positive_rate']:.3f} | "
            f"{'yes' if row['certification_eligible'] else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "The validation run checks that the certification surface penalizes zero action, source leakage, and stress-specific coverage collapse. Any winner change is a development diagnostic only. Test claims require a separately authorized test registry and are not supported here.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_path = ROOT / config["output_data"]
    if not data_path.exists():
        raise FileNotFoundError(f"build data first: {data_path}")
    frame = pd.read_csv(data_path)
    evaluated = frame[frame["split"] == config["evaluation_split"]].copy()
    if evaluated.empty:
        raise ValueError("evaluation split is empty")

    results_dir = ROOT / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
    rules = phase1_rules()
    access = audit_rules(rules, "coverage")
    access.to_csv(results_dir / "feature_access_audit.csv", index=False)
    invalid = set(access.loc[access["status"] == "INVALID", "rule"])
    if invalid:
        raise RuntimeError(f"invalid deployable feature access: {sorted(invalid)}")

    decision_frames: list[pd.DataFrame] = []
    metrics_rows: list[dict[str, object]] = []
    bounds_rows: list[dict[str, object]] = []
    for rule_index, rule in enumerate(rules):
        decision = rule.decide(evaluated, config["thresholds"])
        ruled = with_decision_outcomes(evaluated, decision)
        ruled.insert(0, "rule", rule.name)
        decision_frames.append(ruled)
        for profile_name in ("overall", "ordinary", "stress"):
            current = _profile(ruled, profile_name)
            point = profile_metrics(current)
            point.update({"rule": rule.name, "profile": profile_name})
            metrics_rows.append(point)
            profile_seed = derive_seed(
                int(config["bootstrap_seed"]), f"coverage/{rule_index}/{rule.name}/{profile_name}"
            )
            bounds = cluster_bootstrap_bounds(
                current,
                replicates=int(config["bootstrap_replicates"]),
                seed=profile_seed,
                chunk_size=int(config.get("bootstrap_chunk_size", 64)),
            )
            bounds.update({"rule": rule.name, "profile": profile_name})
            bounds_rows.append(bounds)

    decisions = pd.concat(decision_frames, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)
    bounds = pd.DataFrame(bounds_rows)

    ordinary = metrics[metrics["profile"] == "ordinary"].set_index("rule")
    stress = metrics[metrics["profile"] == "stress"].set_index("rule")
    ratio_rows = []
    for rule in ordinary.index:
        ratio = safe_ratio(float(stress.loc[rule, "coverage"]), float(ordinary.loc[rule, "coverage"]))
        if ratio is None:
            relation = "UNDEFINED_ZERO_ORDINARY"
        elif np.isclose(ratio, 1.0):
            relation = "UNIFORM"
        elif ratio < 1.0:
            relation = "COLLAPSE"
        else:
            relation = "EXPANSION"
        ratio_rows.append(
            {
                "rule": rule,
                "stress_coverage_ratio": ratio,
                "coverage_collapse_index": None if ratio is None else 1.0 - ratio,
                "coverage_relation": relation,
            }
        )
    ratios = pd.DataFrame(ratio_rows)

    surface, certification = build_surface(bounds[bounds["profile"].isin(["overall", "stress"])])
    opportunity = (
        evaluated.drop_duplicates("event_cluster_id")
        .groupby("opportunity_slice", as_index=False)
        .agg(
            event_clusters=("event_cluster_id", "size"),
            oracle_positive_rate=("oracle_positive_opportunity", "mean"),
            intended_opportunity_rate=("latent_opportunity_intent", "mean"),
            certification_eligible=("certification_eligible", "first"),
        )
    )

    ranking = (
        metrics[metrics["profile"] == "overall"]
        .merge(certification, on="rule", how="left")
        .merge(ratios, on="rule", how="left")
    )
    ranking["raw_far_rank"] = ranking["far"].rank(method="min", na_option="bottom")
    ranking["raw_cau_rank"] = ranking["cau"].rank(method="min", ascending=False)
    ranking["certified_rank"] = ranking["worst_profile_certification_volume"].rank(
        method="min", ascending=False
    )

    outputs = {
        "decisions.csv": decisions,
        "metrics.csv": metrics,
        "bootstrap_bounds.csv": bounds,
        "coverage_collapse.csv": ratios,
        "certification_surface.csv": surface,
        "certification_summary.csv": certification,
        "opportunity_by_slice.csv": opportunity,
        "raw_vs_certified_ranking.csv": ranking,
        "feature_access_audit.csv": access,
    }
    for name, output in outputs.items():
        output.to_csv(results_dir / name, index=False)
    report_path = results_dir / "report.md"
    report_path.write_text(_render_report(config, ranking, opportunity), encoding="utf-8")

    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.2.0",
        "mode": config["mode"],
        "evaluation_split": config["evaluation_split"],
        "rows_evaluated": len(evaluated),
        "clusters_evaluated": int(evaluated["event_cluster_id"].nunique()),
        "bootstrap_replicates": int(config["bootstrap_replicates"]),
        "bootstrap_unit": "event_cluster_id",
        "outputs": {
            **{name: sha256(results_dir / name) for name in outputs},
            "report.md": sha256(report_path),
        },
        "data_sha256": sha256(data_path),
        "config_sha256": sha256(config_path),
        "confirmatory": False,
        "claim_boundary": "Validation-only controlled audit; no test claim, market-simulation claim, or method-victory claim.",
    }
    manifest_path = results_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return results_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run coverage-collapse and certification validation audit.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "smoke.yaml"))
    args = parser.parse_args()
    results = run(Path(args.config))
    print(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

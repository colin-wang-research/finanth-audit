from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import scipy
import sklearn
import yaml
from scipy.stats import spearmanr

from finauth_audit.baselines.provenance_rules import provenance_rules
from finauth_audit.baselines.rules import phase1_rules
from finauth_audit.evaluation.certification_surface import build_surface
from finauth_audit.evaluation.cluster_bootstrap import cluster_bootstrap_bounds
from finauth_audit.evaluation.coverage_audit import _profile
from finauth_audit.evaluation.feature_access_audit import audit_rules
from finauth_audit.evaluation.laundering_metrics import (
    laundering_metrics,
    with_laundering_outcomes,
)
from finauth_audit.evaluation.metrics import profile_metrics, safe_ratio, with_decision_outcomes
from finauth_audit.evaluation.seeds import derive_seed


ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = ROOT / "manifests" / "paper_test_registry.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _timestamp() -> str:
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def _group_metrics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(columns, dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(columns, key_values))
        row.update(laundering_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def evaluate_controlled(
    evaluated: pd.DataFrame,
    config: dict[str, Any],
    results_dir: Path,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    rules = phase1_rules()
    access = audit_rules(rules, "coverage")
    invalid = set(access.loc[access["status"] == "INVALID", "rule"])
    if invalid:
        raise RuntimeError(f"invalid controlled rule feature access: {sorted(invalid)}")

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
            bounds = cluster_bootstrap_bounds(
                current,
                replicates=bootstrap_replicates,
                seed=derive_seed(
                    int(config["bootstrap_seed"]),
                    f"coverage/{rule_index}/{rule.name}/{profile_name}",
                ),
                chunk_size=int(config.get("bootstrap_chunk_size", 64)),
            )
            bounds.update({"rule": rule.name, "profile": profile_name})
            bounds_rows.append(bounds)

    decisions = pd.concat(decision_frames, ignore_index=True)
    metrics = pd.DataFrame(metrics_rows)
    bounds = pd.DataFrame(bounds_rows)
    ordinary = metrics[metrics["profile"] == "ordinary"].set_index("rule")
    stress = metrics[metrics["profile"] == "stress"].set_index("rule")
    ratios: list[dict[str, object]] = []
    for rule in ordinary.index:
        ratio = safe_ratio(
            float(stress.loc[rule, "coverage"]),
            float(ordinary.loc[rule, "coverage"]),
        )
        if ratio is None:
            relation = "UNDEFINED_ZERO_ORDINARY"
        elif np.isclose(ratio, 1.0):
            relation = "UNIFORM"
        elif ratio < 1.0:
            relation = "COLLAPSE"
        else:
            relation = "EXPANSION"
        ratios.append(
            {
                "rule": rule,
                "stress_coverage_ratio": ratio,
                "coverage_collapse_index": None if ratio is None else 1.0 - ratio,
                "coverage_relation": relation,
            }
        )
    ratio_frame = pd.DataFrame(ratios)
    surface, certification = build_surface(
        bounds[bounds["profile"].isin(["overall", "stress"])]
    )
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
        .merge(ratio_frame, on="rule", how="left")
    )
    ranking["raw_far_rank"] = ranking["far"].rank(
        method="min", na_option="bottom"
    )
    ranking["raw_cau_rank"] = ranking["cau"].rank(
        method="min", ascending=False
    )
    ranking["certified_rank"] = ranking[
        "worst_profile_certification_volume"
    ].rank(method="min", ascending=False)

    outputs = {
        "decisions.csv": decisions,
        "metrics.csv": metrics,
        "bootstrap_bounds.csv": bounds,
        "coverage_collapse.csv": ratio_frame,
        "certification_surface.csv": surface,
        "certification_summary.csv": certification,
        "opportunity_by_slice.csv": opportunity,
        "raw_vs_certified_ranking.csv": ranking,
        "feature_access_audit.csv": access,
    }
    for name, frame in outputs.items():
        frame.to_csv(results_dir / name, index=False)
    return {
        "rows_evaluated": len(evaluated),
        "clusters_evaluated": int(evaluated["event_cluster_id"].nunique()),
        "outputs": {name: sha256(results_dir / name) for name in outputs},
        "ranking": ranking,
        "metrics": metrics,
    }


def evaluate_provenance(
    train: pd.DataFrame,
    evaluated: pd.DataFrame,
    config: dict[str, Any],
    results_dir: Path,
    bootstrap_replicates: int,
) -> dict[str, Any]:
    results_dir.mkdir(parents=True, exist_ok=True)
    rules = provenance_rules(
        config["thresholds"],
        train,
        model_seed=derive_seed(int(config["seed"]), "provenance-learned-gate"),
    )
    access = audit_rules(rules, "provenance")
    invalid = access[access["status"] == "INVALID"]
    if not invalid.empty:
        raise RuntimeError(
            f"invalid provenance rule feature access: {invalid['rule'].tolist()}"
        )

    decision_frames: list[pd.DataFrame] = []
    bounds_rows: list[dict[str, object]] = []
    for rule in rules:
        decision = rule.decide(evaluated, config["thresholds"])
        ruled = with_laundering_outcomes(evaluated, decision)
        ruled.insert(0, "rule", rule.name)
        decision_frames.append(ruled)
        bounds = cluster_bootstrap_bounds(
            ruled,
            replicates=bootstrap_replicates,
            seed=derive_seed(int(config["bootstrap_seed"]), f"provenance/{rule.name}"),
            chunk_size=int(config.get("bootstrap_chunk_size", 64)),
        )
        bounds["rule"] = rule.name
        bounds_rows.append(bounds)

    decisions = pd.concat(decision_frames, ignore_index=True)
    summary = _group_metrics(decisions, ["rule"])
    by_attack = _group_metrics(decisions, ["rule", "attack_type"])
    by_traceability = _group_metrics(decisions, ["rule", "traceability"])
    by_noise = _group_metrics(
        decisions[decisions["attack_type"] == "role_noise"],
        ["rule", "role_noise_rate"],
    )
    by_hop = _group_metrics(
        decisions[decisions["attack_type"] == "multi_hop"],
        ["rule", "hop_depth", "traceability"],
    )
    bounds = pd.DataFrame(bounds_rows)
    outputs = {
        "decisions.csv": decisions,
        "summary.csv": summary,
        "by_attack.csv": by_attack,
        "by_traceability.csv": by_traceability,
        "by_role_noise.csv": by_noise,
        "by_hop_depth.csv": by_hop,
        "bootstrap_bounds.csv": bounds,
        "feature_access_audit.csv": access,
    }
    for name, frame in outputs.items():
        frame.to_csv(results_dir / name, index=False)
    learned = next(rule for rule in rules if rule.name == "Provenance Learned Gate")
    metadata_path = results_dir / "learned_gate_metadata.json"
    metadata_path.write_text(
        json.dumps(learned.metadata(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    output_hashes = {name: sha256(results_dir / name) for name in outputs}
    output_hashes[metadata_path.name] = sha256(metadata_path)
    return {
        "rows_evaluated": len(evaluated),
        "clusters_evaluated": int(evaluated["event_cluster_id"].nunique()),
        "outputs": output_hashes,
        "summary": summary,
    }


def _numeric_equivalence(
    observed: pd.DataFrame,
    expected: pd.DataFrame,
    keys: list[str],
    metrics: list[str],
) -> dict[str, object]:
    merged = observed[keys + metrics].merge(
        expected[keys + metrics],
        on=keys,
        how="outer",
        suffixes=("_observed", "_expected"),
        indicator=True,
    )
    differences: dict[str, float] = {}
    equivalent = bool((merged["_merge"] == "both").all())
    for metric in metrics:
        left = pd.to_numeric(merged[f"{metric}_observed"], errors="coerce")
        right = pd.to_numeric(merged[f"{metric}_expected"], errors="coerce")
        finite = left.notna() & right.notna()
        max_difference = (
            float(np.max(np.abs(left[finite] - right[finite]))) if finite.any() else 0.0
        )
        same_na = bool((left.isna() == right.isna()).all())
        differences[metric] = max_difference
        equivalent = equivalent and same_na and max_difference <= 1e-12
    return {
        "equivalent": equivalent,
        "row_count": len(merged),
        "max_absolute_difference": differences,
    }


def run_validation_rehearsal(bootstrap_replicates: int = 200) -> Path:
    output_dir = ROOT / "results" / "paper_test_rehearsal"
    output_dir.mkdir(parents=True, exist_ok=True)
    controlled_config = yaml.safe_load(
        (ROOT / "configs" / "main.yaml").read_text(encoding="utf-8")
    )
    provenance_config = yaml.safe_load(
        (ROOT / "configs" / "provenance_main.yaml").read_text(encoding="utf-8")
    )
    controlled = pd.read_csv(ROOT / "data" / "paper_test" / "controlled_validation.csv")
    provenance_train = pd.read_csv(ROOT / "data" / "paper_test" / "provenance_train.csv")
    provenance_validation = pd.read_csv(
        ROOT / "data" / "paper_test" / "provenance_validation.csv"
    )
    controlled_result = evaluate_controlled(
        controlled,
        controlled_config,
        output_dir / "controlled",
        bootstrap_replicates,
    )
    provenance_result = evaluate_provenance(
        provenance_train,
        provenance_validation,
        provenance_config,
        output_dir / "provenance",
        bootstrap_replicates,
    )
    controlled_equivalence = _numeric_equivalence(
        controlled_result["metrics"],
        pd.read_csv(ROOT / "results" / "main" / "metrics.csv"),
        ["rule", "profile"],
        [
            "coverage",
            "execute_rate",
            "reduce_rate",
            "review_rate",
            "abstain_rate",
            "far",
            "alr",
            "cau",
            "moc",
        ],
    )
    provenance_equivalence = _numeric_equivalence(
        provenance_result["summary"],
        pd.read_csv(ROOT / "results" / "provenance_main" / "summary.csv"),
        ["rule"],
        [
            "coverage",
            "far",
            "alr",
            "direct_leakage_rate",
            "indirect_leakage_rate",
            "safe_delegation_coverage",
            "false_block_rate",
            "cau",
            "moc",
            "review_rate",
        ],
    )
    equivalence = {
        "controlled": controlled_equivalence,
        "provenance": provenance_equivalence,
        "passed": bool(
            controlled_equivalence["equivalent"]
            and provenance_equivalence["equivalent"]
        ),
    }
    equivalence_path = output_dir / "rehearsal_equivalence.json"
    equivalence_path.write_text(
        json.dumps(equivalence, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.3.0-round75-paper-test-rehearsal",
        "evaluation_split": "validation",
        "confirmatory": False,
        "test_outcomes_evaluated": False,
        "community_hidden_outcomes_evaluated": False,
        "bootstrap_replicates": bootstrap_replicates,
        "equivalence_passed": equivalence["passed"],
        "outputs": {
            str(path.relative_to(output_dir)): sha256(path)
            for path in output_dir.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        },
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if not equivalence["passed"]:
        raise RuntimeError(f"validation rehearsal failed equivalence: {equivalence}")
    print(output_dir / "manifest.json")
    return output_dir


def _rank_stability(
    validation: pd.DataFrame,
    paper: pd.DataFrame,
    rank_column: str,
) -> dict[str, object]:
    merged = validation[["rule", rank_column]].merge(
        paper[["rule", rank_column]], on="rule", suffixes=("_validation", "_paper")
    )
    correlation = spearmanr(
        merged[f"{rank_column}_validation"],
        merged[f"{rank_column}_paper"],
    ).statistic
    return {
        "rank": rank_column,
        "spearman": float(correlation) if np.isfinite(correlation) else None,
        "rules": len(merged),
    }


def _write_branch_neutral_report(
    output_dir: Path,
    controlled: dict[str, Any],
    provenance: dict[str, Any],
    comparison: dict[str, Any],
) -> Path:
    controlled_ranking = controlled["ranking"].sort_values(
        ["certified_rank", "raw_far_rank", "rule"]
    )
    provenance_summary = provenance["summary"].sort_values(
        ["alr", "coverage", "rule"], na_position="last"
    )
    lines = [
        "# One-Time Paper Test Report",
        "",
        "This report was generated once from the frozen paper-test partition. "
        "The separate community-hidden partition was not evaluated.",
        "",
        "## Controlled authorization",
        "",
        controlled_ranking[
            [
                "rule",
                "coverage",
                "far",
                "alr",
                "review_rate",
                "moc",
                "worst_profile_certification_volume",
                "raw_far_rank",
                "certified_rank",
            ]
        ].to_csv(index=False),
        "",
        "## Provenance authorization",
        "",
        provenance_summary[
            [
                "rule",
                "coverage",
                "far",
                "alr",
                "direct_leakage_rate",
                "indirect_leakage_rate",
                "safe_delegation_coverage",
                "review_rate",
            ]
        ].to_csv(index=False),
        "",
        "## Validation-to-paper rank stability",
        "",
        json.dumps(comparison, indent=2, sort_keys=True),
        "",
        "## Interpretation boundary",
        "",
        "These are controlled and provenance paper-test results. They are not public "
        "market deployment evidence, practitioner labels, trading profitability, or a "
        "global ranking of execution rules. Any divergence from validation is retained.",
    ]
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def _preflight_freeze(freeze_manifest_path: Path) -> dict[str, Any]:
    freeze = json.loads(freeze_manifest_path.read_text(encoding="utf-8"))
    if freeze.get("status") != "FROZEN_BEFORE_PAPER_TEST":
        raise RuntimeError("paper-test freeze manifest is not active")
    expected = freeze["surface_hashes"].get(
        "evaluation/paper_test.py"
    )
    if expected != sha256(Path(__file__)):
        raise RuntimeError("paper-test evaluator hash differs from frozen surface")
    archive_path = ROOT / freeze["archive_path"]
    if sha256(archive_path) != freeze["archive_sha256"]:
        raise RuntimeError("content-addressed freeze archive hash mismatch")
    return freeze


def run_paper_test(freeze_manifest_path: Path) -> Path:
    if REGISTRY_PATH.exists():
        raise FileExistsError(
            "paper-test registry already exists; one-time evaluation cannot be rerun"
        )
    freeze = _preflight_freeze(freeze_manifest_path)
    registry = {
        "project": "FinAuth-Audit",
        "status": "RUNNING",
        "started_at": _timestamp(),
        "command": " ".join(sys.argv),
        "freeze_manifest": str(freeze_manifest_path.relative_to(ROOT)),
        "freeze_manifest_sha256": sha256(freeze_manifest_path),
        "archive_path": freeze["archive_path"],
        "archive_sha256": freeze["archive_sha256"],
        "results_inspected": False,
        "community_hidden_outcomes_evaluated": False,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "scipy": scipy.__version__,
            "sklearn": sklearn.__version__,
            "platform": platform.platform(),
        },
    }
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with REGISTRY_PATH.open("x", encoding="utf-8") as handle:
        json.dump(registry, handle, indent=2, sort_keys=True)
        handle.write("\n")

    output_dir = ROOT / "results" / "paper_test"
    try:
        controlled_config = yaml.safe_load(
            (ROOT / "configs" / "main.yaml").read_text(encoding="utf-8")
        )
        provenance_config = yaml.safe_load(
            (ROOT / "configs" / "provenance_main.yaml").read_text(
                encoding="utf-8"
            )
        )
        controlled_data = ROOT / "data" / "paper_test" / "controlled.csv"
        provenance_train_data = (
            ROOT / "data" / "paper_test" / "provenance_train.csv"
        )
        provenance_data = ROOT / "data" / "paper_test" / "provenance.csv"
        controlled_frame = pd.read_csv(controlled_data)
        provenance_train = pd.read_csv(provenance_train_data)
        provenance_frame = pd.read_csv(provenance_data)
        controlled = evaluate_controlled(
            controlled_frame,
            controlled_config,
            output_dir / "controlled",
            int(controlled_config["bootstrap_replicates"]),
        )
        provenance = evaluate_provenance(
            provenance_train,
            provenance_frame,
            provenance_config,
            output_dir / "provenance",
            int(provenance_config["bootstrap_replicates"]),
        )
        validation_ranking = pd.read_csv(
            ROOT / "results" / "main" / "raw_vs_certified_ranking.csv"
        )
        comparison = {
            "raw_far_rank": _rank_stability(
                validation_ranking, controlled["ranking"], "raw_far_rank"
            ),
            "certified_rank": _rank_stability(
                validation_ranking, controlled["ranking"], "certified_rank"
            ),
        }
        comparison_path = output_dir / "rank_stability.json"
        comparison_path.parent.mkdir(parents=True, exist_ok=True)
        comparison_path.write_text(
            json.dumps(comparison, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report_path = _write_branch_neutral_report(
            output_dir, controlled, provenance, comparison
        )
        output_hashes = {
            str(path.relative_to(output_dir)): sha256(path)
            for path in output_dir.rglob("*")
            if path.is_file() and path.name != "manifest.json"
        }
        manifest = {
            "project": "FinAuth-Audit",
            "version": "0.3.0-round75-one-time-paper-test",
            "evaluation_split": "paper_test",
            "source_split": "test",
            "confirmatory": True,
            "paper_test_outcomes_evaluated": True,
            "community_hidden_outcomes_evaluated": False,
            "controlled_rows": controlled["rows_evaluated"],
            "controlled_clusters": controlled["clusters_evaluated"],
            "provenance_rows": provenance["rows_evaluated"],
            "provenance_clusters": provenance["clusters_evaluated"],
            "bootstrap_replicates": {
                "controlled": int(controlled_config["bootstrap_replicates"]),
                "provenance": int(provenance_config["bootstrap_replicates"]),
            },
            "data_hashes": {
                str(controlled_data.relative_to(ROOT)): sha256(controlled_data),
                str(provenance_train_data.relative_to(ROOT)): sha256(
                    provenance_train_data
                ),
                str(provenance_data.relative_to(ROOT)): sha256(provenance_data),
            },
            "config_hashes": {
                "configs/main.yaml": sha256(ROOT / "configs" / "main.yaml"),
                "configs/provenance_main.yaml": sha256(
                    ROOT / "configs" / "provenance_main.yaml"
                ),
            },
            "freeze_manifest_sha256": sha256(freeze_manifest_path),
            "outputs": output_hashes,
            "report_sha256": sha256(report_path),
            "claim_boundary": (
                "One-time controlled and provenance paper test. Public test, community "
                "hidden outcomes, FinAuth-Worlds hidden outcomes, practitioner labels, "
                "and deployment evidence remain unavailable."
            ),
        }
        manifest_path = output_dir / "manifest.json"
        manifest_path.write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        registry.update(
            {
                "status": "COMPLETE_UNINSPECTED",
                "completed_at": _timestamp(),
                "result_manifest": str(manifest_path.relative_to(ROOT)),
                "result_manifest_sha256": sha256(manifest_path),
                "output_hashes": output_hashes,
            }
        )
        REGISTRY_PATH.write_text(
            json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(manifest_path)
        return output_dir
    except Exception as error:
        registry.update(
            {
                "status": "FAILED_UNINSPECTED",
                "failed_at": _timestamp(),
                "error_type": type(error).__name__,
                "error": str(error),
            }
        )
        REGISTRY_PATH.write_text(
            json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        raise


def mark_inspected(note: str) -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if registry.get("status") != "COMPLETE_UNINSPECTED":
        raise RuntimeError(
            f"cannot mark registry status {registry.get('status')} as inspected"
        )
    registry.update(
        {
            "status": "COMPLETE_INSPECTED",
            "results_inspected": True,
            "inspected_at": _timestamp(),
            "inspection_note": note,
            "post_inspection_change_policy": (
                "No generator, rule, threshold, metric, attack, split, or certification "
                "profile changes are permitted for v0.2."
            ),
        }
    )
    REGISTRY_PATH.write_text(
        json.dumps(registry, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Run or audit the frozen paper test.")
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--validation-rehearsal", action="store_true")
    mode.add_argument("--execute-frozen-paper-test", action="store_true")
    mode.add_argument("--mark-inspected", action="store_true")
    parser.add_argument(
        "--freeze-manifest",
        default=str(ROOT / "manifests" / "paper_test_freeze.json"),
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=200)
    parser.add_argument(
        "--inspection-note",
        default="Results inspected for branch-neutral paper reporting.",
    )
    args = parser.parse_args()
    if args.validation_rehearsal:
        run_validation_rehearsal(args.bootstrap_replicates)
    elif args.execute_frozen_paper_test:
        run_paper_test(Path(args.freeze_manifest).resolve())
    else:
        mark_inspected(args.inspection_note)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

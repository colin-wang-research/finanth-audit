from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from finauth_audit.baselines.rules import AuthorizationRule, ELIGIBLE_ROLES, phase1_rules
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


def _lifecycle_variant(
    frame: pd.DataFrame,
    thresholds: dict[str, float],
    *,
    use_role: bool = True,
    use_review_risk: bool = True,
    use_cost: bool = True,
    allow_reduce: bool = True,
) -> np.ndarray:
    total_cost = frame["liquidity_cost_bps"] + frame["turnover_cost_bps"] + frame["fee_bps"]
    role_ok = frame["source_role"].isin(ELIGIBLE_ROLES) if use_role else pd.Series(True, index=frame.index)
    high_review_risk = (
        (frame["uncertainty"] >= thresholds["lifecycle_review_uncertainty"])
        | (frame["volatility_proxy"] >= thresholds["lifecycle_review_volatility"])
    ) if use_review_risk else pd.Series(False, index=frame.index)
    margin = frame["expected_edge_bps"] - (total_cost if use_cost else 0.0)
    decisions = np.full(len(frame), "abstain", dtype=object)
    decisions[(~role_ok).to_numpy()] = "review"
    decisions[(role_ok & high_review_risk).to_numpy()] = "review"
    eligible = role_ok & ~high_review_risk
    if allow_reduce:
        decisions[(eligible & (margin > 0)).to_numpy()] = "reduce"
    decisions[(eligible & (margin > thresholds["lifecycle_execute_margin_bps"])).to_numpy()] = "execute"
    return decisions


def _role_cost(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    total_cost = frame["liquidity_cost_bps"] + frame["turnover_cost_bps"] + frame["fee_bps"]
    allowed = frame["source_role"].isin(ELIGIBLE_ROLES) & (
        frame["expected_edge_bps"] > total_cost + thresholds["cost_margin_bps"]
    )
    return np.where(allowed, "execute", "abstain")


def _role_risk(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    total_cost = frame["liquidity_cost_bps"] + frame["turnover_cost_bps"] + frame["fee_bps"]
    allowed = (
        frame["source_role"].isin(ELIGIBLE_ROLES)
        & (total_cost <= thresholds["risk_cost_bps"])
        & (frame["volatility_proxy"] <= thresholds["risk_volatility"])
    )
    return np.where(allowed, "execute", "abstain")


def _role_confidence_cost(frame: pd.DataFrame, thresholds: dict[str, float]) -> np.ndarray:
    total_cost = frame["liquidity_cost_bps"] + frame["turnover_cost_bps"] + frame["fee_bps"]
    allowed = (
        frame["source_role"].isin(ELIGIBLE_ROLES)
        & (frame["confidence"] >= thresholds["confidence"])
        & (frame["expected_edge_bps"] > total_cost + thresholds["cost_margin_bps"])
    )
    return np.where(allowed, "execute", "abstain")


def diagnostic_rules() -> list[AuthorizationRule]:
    legal = (
        "source_role",
        "uncertainty",
        "volatility_proxy",
        "expected_edge_bps",
        "liquidity_cost_bps",
        "turnover_cost_bps",
        "fee_bps",
    )
    return [
        AuthorizationRule(
            "Lifecycle Checklist [registered]",
            legal,
            lambda frame, thresholds: _lifecycle_variant(frame, thresholds),
            classification="registered_baseline",
        ),
        AuthorizationRule(
            "Lifecycle without role",
            tuple(feature for feature in legal if feature != "source_role"),
            lambda frame, thresholds: _lifecycle_variant(frame, thresholds, use_role=False),
            classification="diagnostic_ablation",
        ),
        AuthorizationRule(
            "Lifecycle without review risk",
            tuple(feature for feature in legal if feature not in {"uncertainty", "volatility_proxy"}),
            lambda frame, thresholds: _lifecycle_variant(frame, thresholds, use_review_risk=False),
            classification="diagnostic_ablation",
        ),
        AuthorizationRule(
            "Lifecycle without cost",
            tuple(feature for feature in legal if "cost_bps" not in feature and feature != "fee_bps"),
            lambda frame, thresholds: _lifecycle_variant(frame, thresholds, use_cost=False),
            classification="diagnostic_ablation",
        ),
        AuthorizationRule(
            "Lifecycle without reduce route",
            legal,
            lambda frame, thresholds: _lifecycle_variant(frame, thresholds, allow_reduce=False),
            classification="diagnostic_ablation",
        ),
        AuthorizationRule(
            "Role + Cost Gate",
            ("source_role", "expected_edge_bps", "liquidity_cost_bps", "turnover_cost_bps", "fee_bps"),
            _role_cost,
            classification="diagnostic_composition",
        ),
        AuthorizationRule(
            "Role + Risk Gate",
            ("source_role", "liquidity_cost_bps", "turnover_cost_bps", "fee_bps", "volatility_proxy"),
            _role_risk,
            classification="diagnostic_composition",
        ),
        AuthorizationRule(
            "Role + Confidence + Cost Gate",
            (
                "source_role",
                "confidence",
                "expected_edge_bps",
                "liquidity_cost_bps",
                "turnover_cost_bps",
                "fee_bps",
            ),
            _role_confidence_cost,
            classification="diagnostic_composition",
        ),
    ]


def _profile(frame: pd.DataFrame, name: str) -> pd.DataFrame:
    scored = frame[frame["certification_eligible"]].copy()
    if name == "overall":
        return scored
    if name == "stress":
        return scored[scored["opportunity_slice"] != "ordinary"].copy()
    raise KeyError(name)


def _render_report(summary: pd.DataFrame, access: pd.DataFrame) -> str:
    lines = [
        "# Baseline Governance and Lifecycle Ablation Report",
        "",
        "The registered Lifecycle Checklist was implemented before the accepted validation smoke result, but the controlled generator underwent documented validation-time repairs. These ablations therefore test dependence; they do not retroactively make the baseline externally preregistered.",
        "",
        "## Timeline",
        "",
        "- Round 68 plan frozen the baseline-independent certification task at 2026-07-17 19:22:17 CST.",
        "- `baselines/rules.py` and the smoke config were created at 2026-07-17 19:29:20 CST.",
        "- The accepted Smoke 3 validation result was recorded at 2026-07-17 19:45:32 CST after two failed generator designs were retained in the negative log.",
        "- All Round 75 ablations are validation-only and frozen before paper-test access.",
        "",
        "## Component ablations and alternative compositions",
        "",
        "| Rule | Class | Coverage | FAR | ALR | Review | CAU | MOC | Worst-profile certification |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["worst_profile_certification_volume", "rule"], ascending=[False, True]).to_dict(orient="records"):
        far = "N/A" if pd.isna(row["far"]) else f"{row['far']:.3f}"
        alr = "N/A" if pd.isna(row["alr"]) else f"{row['alr']:.3f}"
        lines.append(
            f"| {row['rule']} | {row['classification']} | {row['coverage']:.3f} | {far} | {alr} | "
            f"{row['review_rate']:.3f} | {row['cau']:.3f} | {row['moc']:.3f} | "
            f"{row['worst_profile_certification_volume']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Legal feature access",
            "",
            f"All {len(access)} diagnostic rules pass the coverage-task feature-access manifest. Outcome fields, future decoys, harm labels, and test labels are not used.",
            "",
            "## Boundary",
            "",
            "A component ablation that lowers certification does not prove the registered checklist is globally optimal. The analysis asks whether its validation result is reducible to one feature family and whether independently motivated compositions reach similar operating regions.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path) -> Path:
    config_path = config_path.resolve()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("evaluation_split") != "validation":
        raise RuntimeError("baseline governance is validation-only before paper-test access")
    data_path = ROOT / config["output_data"]
    frame = pd.read_csv(data_path, low_memory=False)
    evaluated = frame[frame["split"] == "validation"].copy()
    rules = diagnostic_rules()
    access = audit_rules(rules, "coverage")
    if (access["status"] == "INVALID").any():
        raise RuntimeError("diagnostic baseline uses an illegal field")

    registered = next(rule for rule in phase1_rules() if rule.name == "Lifecycle Checklist")
    registered_decisions = registered.decide(evaluated, config["thresholds"])
    diagnostic_decisions = rules[0].decide(evaluated, config["thresholds"])
    if not np.array_equal(registered_decisions, diagnostic_decisions):
        raise RuntimeError("registered lifecycle reproduction mismatch")

    metric_rows: list[dict[str, object]] = []
    bound_rows: list[dict[str, object]] = []
    decision_rows: list[pd.DataFrame] = []
    for rule_index, rule in enumerate(rules):
        decisions = rule.decide(evaluated, config["thresholds"])
        ruled = with_decision_outcomes(evaluated, decisions)
        ruled.insert(0, "rule", rule.name)
        ruled.insert(1, "classification", rule.classification)
        decision_rows.append(ruled)
        for profile in ("overall", "stress"):
            current = _profile(ruled, profile)
            point = profile_metrics(current)
            point.update(
                {
                    "rule": rule.name,
                    "classification": rule.classification,
                    "profile": profile,
                }
            )
            metric_rows.append(point)
            bound = cluster_bootstrap_bounds(
                current,
                replicates=int(config["bootstrap_replicates"]),
                seed=derive_seed(
                    int(config["bootstrap_seed"]),
                    f"baseline-governance/{rule_index}/{profile}",
                ),
                chunk_size=int(config.get("bootstrap_chunk_size", 64)),
            )
            bound.update({"rule": rule.name, "profile": profile})
            bound_rows.append(bound)
    decisions = pd.concat(decision_rows, ignore_index=True)
    metrics = pd.DataFrame(metric_rows)
    bounds = pd.DataFrame(bound_rows)
    surface, certification = build_surface(bounds)
    summary = metrics[metrics["profile"] == "overall"].merge(certification, on="rule", how="left")

    output_dir = ROOT / "results" / "baseline_governance"
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "decisions.csv": decisions,
        "metrics.csv": metrics,
        "bootstrap_bounds.csv": bounds,
        "certification_surface.csv": surface,
        "summary.csv": summary,
        "feature_access_audit.csv": access,
    }
    for name, output in outputs.items():
        output.to_csv(output_dir / name, index=False)
    report_path = output_dir / "report.md"
    report_path.write_text(_render_report(summary, access), encoding="utf-8")
    source_files = [
        ROOT / "baselines" / "rules.py",
        config_path,
        ROOT.parent / "log.md",
    ]
    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.2.0-round75-baseline-governance",
        "evaluation_split": "validation",
        "test_outcomes_evaluated": False,
        "registered_lifecycle_reproduced_exactly": True,
        "source_hashes": {str(path): sha256(path) for path in source_files},
        "data_sha256": sha256(data_path),
        "outputs": {
            **{name: sha256(output_dir / name) for name in outputs},
            "report.md": sha256(report_path),
        },
        "claim_boundary": "Validation-only baseline dependence and composition audit; no preregistration, test, or winner claim.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run validation-only Lifecycle Checklist governance diagnostics.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "main.yaml"))
    args = parser.parse_args()
    print(run(Path(args.config)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

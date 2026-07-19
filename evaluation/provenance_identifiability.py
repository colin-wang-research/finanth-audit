from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def ambiguity_lower_bound(frame: pd.DataFrame, signature: list[str], label: str) -> dict[str, object]:
    labels = frame[label]
    if not labels.isin([0, 1, False, True]).all():
        raise ValueError("identifiability label must be binary")
    grouped = frame.groupby(signature, dropna=False)[label].agg(["sum", "count"])
    collisions = grouped[grouped["count"] > 1]
    positives = grouped["sum"].astype(float)
    negatives = grouped["count"].astype(float) - positives
    errors = np.minimum(positives, negatives)
    rows = float(len(frame))
    empirical_bayes_error = float(errors.sum() / rows) if rows else np.nan
    joint_separation = (
        float(np.abs(positives - negatives).sum() / rows) if rows else np.nan
    )
    identity_error = (
        float(abs(empirical_bayes_error - (1.0 - joint_separation) / 2.0))
        if rows
        else np.nan
    )
    return {
        "rows": len(frame),
        "positive_label_rate": float(labels.astype(float).mean()) if len(frame) else np.nan,
        "signature_count": len(grouped),
        "collision_signature_count": len(collisions),
        "collision_row_fraction": float(
            grouped.loc[grouped["count"] > 1, "count"].sum() / len(frame)
        )
        if len(frame)
        else np.nan,
        "unavoidable_error_rows": int(errors.sum()),
        "empirical_bayes_error_lower_bound": empirical_bayes_error,
        "empirical_joint_separation": joint_separation,
        "bayes_overlap_identity_residual": identity_error,
    }


def _binned(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    for column in (
        "confidence",
        "uncertainty",
        "expected_edge_bps",
        "liquidity_cost_bps",
        "turnover_cost_bps",
        "volatility_proxy",
    ):
        result[f"{column}_bin"] = pd.qcut(
            result[column], q=5, labels=False, duplicates="drop"
        ).fillna(-1).astype(int)
    return result


def _render_report(results: pd.DataFrame, observed: pd.DataFrame) -> str:
    lines = [
        "# Provenance Identifiability Under Partial Observability",
        "",
        "For any rule measurable with respect to a legal observable signature X, the minimum empirical laundering-classification error is the sum of the minority label mass within each X group. Equivalently, it is one half of one minus the empirical joint-separation term. The identity is exact on the finite audit sample; the reported values remain validation-only evidence and are not a deployed-institution guarantee.",
        "",
        "## Observable-signature ambiguity",
        "",
        "| Traceability | Signature | Rows | Collision-row fraction | Bayes risk | Joint separation | Identity residual |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in results.to_dict(orient="records"):
        lines.append(
            f"| {row['traceability']} | {row['signature']} | {int(row['rows'])} | "
            f"{row['collision_row_fraction']:.3f} | {row['empirical_bayes_error_lower_bound']:.3f} | "
            f"{row['empirical_joint_separation']:.3f} | {row['bayes_overlap_identity_residual']:.2e} |"
        )
    lines.extend(
        [
            "",
            "## Observed rule trade-off on untraceable rows",
            "",
            "| Rule | Coverage | ALR | Safe delegation coverage | False block |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in observed.sort_values(["alr", "coverage"], na_position="last").to_dict(orient="records"):
        def value(name: str) -> str:
            return "N/A" if pd.isna(row[name]) else f"{row[name]:.3f}"

        lines.append(
            f"| {row['rule']} | {row['coverage']:.3f} | {value('alr')} | "
            f"{value('safe_delegation_coverage')} | {value('false_block_rate')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The traceable stratum measures the ceiling when attested lineage is legally available. The untraceable stratum measures an information boundary: lowering laundering by refusing indistinguishable cases can reduce safe-delegation coverage. Hard-gate ALR=0 in the traceable stratum is therefore a construction ceiling, not an algorithmic discovery. The identity residual is machine-checked and should be numerically zero up to floating-point precision.",
            "",
        ]
    )
    return "\n".join(lines)


def run(data_path: Path, traceability_path: Path) -> Path:
    data_path = data_path.resolve()
    traceability_path = traceability_path.resolve()
    required = [
        "split",
        "traceability",
        "authority_laundering",
        "verified_current_role",
        "current_role_verified",
        "claimed_role",
        "confidence",
        "uncertainty",
        "expected_edge_bps",
        "liquidity_cost_bps",
        "turnover_cost_bps",
        "volatility_proxy",
        "transformation_type",
        "lineage_attested",
    ]
    frame = pd.read_csv(data_path, usecols=required)
    frame = frame[frame["split"] == "validation"].copy()
    frame = _binned(frame)
    signatures = {
        "current_role_only": ["verified_current_role", "current_role_verified"],
        "legal_partial": [
            "verified_current_role",
            "current_role_verified",
            "claimed_role",
            "transformation_type",
            "lineage_attested",
            "confidence_bin",
            "uncertainty_bin",
            "expected_edge_bps_bin",
            "liquidity_cost_bps_bin",
            "turnover_cost_bps_bin",
            "volatility_proxy_bin",
        ],
    }
    rows: list[dict[str, object]] = []
    for traceability, stratum in frame.groupby("traceability", sort=True):
        for name, signature in signatures.items():
            rows.append(
                {
                    "traceability": traceability,
                    "signature": name,
                    **ambiguity_lower_bound(stratum, signature, "authority_laundering"),
                }
            )
    results = pd.DataFrame(rows)
    if not np.allclose(
        results["bayes_overlap_identity_residual"].to_numpy(dtype=float),
        0.0,
        atol=1e-12,
    ):
        raise RuntimeError("Bayes-overlap identity failed numerical verification")
    observed = pd.read_csv(traceability_path)
    observed = observed[observed["traceability"] == "untraceable"].copy()
    output_dir = ROOT / "results" / "provenance_identifiability"
    output_dir.mkdir(parents=True, exist_ok=True)
    results_path = output_dir / "ambiguity_bounds.csv"
    observed_path = output_dir / "untraceable_rule_tradeoff.csv"
    results.to_csv(results_path, index=False)
    observed.to_csv(observed_path, index=False)
    report_path = output_dir / "report.md"
    report_path.write_text(_render_report(results, observed), encoding="utf-8")
    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.5.0-round83-identifiability",
        "evaluation_split": "validation",
        "test_outcomes_evaluated": False,
        "inputs": {
            "provenance_data": sha256(data_path),
            "traceability_metrics": sha256(traceability_path),
        },
        "outputs": {
            results_path.name: sha256(results_path),
            observed_path.name: sha256(observed_path),
            report_path.name: sha256(report_path),
        },
        "theoretical_identity": (
            "For binary laundering label L and finite observable signature X, "
            "inf_h P[h(X) != L] = sum_x min(P[X=x,L=0], P[X=x,L=1]). "
            "Under equal priors this equals (1-TV(P(X|L=0),P(X|L=1)))/2."
        ),
        "claim_boundary": (
            "Exact finite-sample Bayes-risk identity plus validation-only plug-in values; "
            "not a universal deployed-institution guarantee or evidence that hidden lineage can be inferred."
        ),
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit provenance identifiability under partial observability.")
    parser.add_argument(
        "--data",
        default=str(ROOT / "data" / "provenance_laundering" / "main.csv"),
    )
    parser.add_argument(
        "--traceability",
        default=str(ROOT / "results" / "provenance_main" / "by_traceability.csv"),
    )
    args = parser.parse_args()
    print(run(Path(args.data), Path(args.traceability)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

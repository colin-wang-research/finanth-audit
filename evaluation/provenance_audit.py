from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd
import yaml

from finauth_audit.baselines.provenance_rules import provenance_rules
from finauth_audit.evaluation.cluster_bootstrap import cluster_bootstrap_bounds
from finauth_audit.evaluation.feature_access_audit import audit_rules
from finauth_audit.evaluation.laundering_metrics import laundering_metrics, with_laundering_outcomes
from finauth_audit.evaluation.seeds import derive_seed


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_metrics(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for keys, group in frame.groupby(columns, dropna=False, sort=True):
        key_values = keys if isinstance(keys, tuple) else (keys,)
        row = dict(zip(columns, key_values))
        row.update(laundering_metrics(group))
        rows.append(row)
    return pd.DataFrame(rows)


def _render_report(
    config: dict[str, object], summary: pd.DataFrame, traceability: pd.DataFrame
) -> str:
    indexed = summary.set_index("rule")
    hard = indexed.loc["Provenance Hard Gate"]
    learned = indexed.loc["Provenance Learned Gate"]
    role_only = indexed.loc["Hard Role Gate"]
    hard_untraceable = traceability[
        (traceability["rule"] == "Provenance Hard Gate")
        & (traceability["traceability"] == "untraceable")
    ].iloc[0]
    learned_untraceable = traceability[
        (traceability["rule"] == "Provenance Learned Gate")
        & (traceability["traceability"] == "untraceable")
    ].iloc[0]
    lines = [
        "# Provenance Laundering Validation Report",
        "",
        f"Mode: `{config['mode']}`",
        "",
        "This is a validation-only controlled provenance audit, not deployed-agent evidence.",
        "",
        "## Overall",
        "",
        "| Rule | Coverage | FAR | ALR | Direct leakage | Indirect leakage | Safe delegation coverage | False block |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary.sort_values(["alr", "coverage"], na_position="last").to_dict(orient="records"):
        def value(name: str) -> str:
            return "N/A" if pd.isna(row[name]) else f"{row[name]:.3f}"

        lines.append(
            f"| {row['rule']} | {row['coverage']:.3f} | {value('far')} | {value('alr')} | "
            f"{value('direct_leakage_rate')} | {value('indirect_leakage_rate')} | "
            f"{value('safe_delegation_coverage')} | {value('false_block_rate')} |"
        )
    lines.extend(
        [
            "",
            "## Traceability strata",
            "",
            "Traceable and untraceable rows are never pooled into a single headline laundering result.",
            "",
            "| Rule | Traceability | Coverage | ALR | Safe delegation coverage | False block |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in traceability.sort_values(["rule", "traceability"]).to_dict(orient="records"):
        def value(name: str) -> str:
            return "N/A" if pd.isna(row[name]) else f"{row[name]:.3f}"

        lines.append(
            f"| {row['rule']} | {row['traceability']} | {row['coverage']:.3f} | {value('alr')} | "
            f"{value('safe_delegation_coverage')} | {value('false_block_rate')} |"
        )
    lines.extend(
        [
            "",
            "## Observed validation trade-offs",
            "",
            (
                f"- The provenance hard gate eliminates observed laundering among authorized rows "
                f"(ALR={hard['alr']:.3f}) but has FAR={hard['far']:.3f}; lineage integrity alone is "
                "not evidence of post-cost authorization quality."
            ),
            (
                f"- On untraceable rows, the hard gate has coverage={hard_untraceable['coverage']:.3f} "
                f"and false-block rate={hard_untraceable['false_block_rate']:.3f}. Its risk is N/A, "
                "not zero, because it authorizes no rows in that stratum."
            ),
            (
                f"- The learned gate preserves untraceable safe-delegation coverage="
                f"{learned_untraceable['safe_delegation_coverage']:.3f}, while accepting residual "
                f"overall ALR={learned['alr']:.3f}. This is a delegation-versus-integrity trade-off, "
                "not a dominance claim."
            ),
            (
                f"- The current-role hard gate has direct leakage={role_only['direct_leakage_rate']:.3f} "
                f"but indirect leakage={role_only['indirect_leakage_rate']:.3f}, showing that clean "
                "current-role metadata does not establish a clean provenance chain."
            ),
            "",
            "## Boundary",
            "",
            "Original roles, direct/indirect leakage labels, and laundering truth are training/evaluation fields only. Deployable rules use the legal feature manifest. EPV is an equal-information baseline and is not the task identity.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path) -> Path:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_path = ROOT / config["output_data"]
    frame = pd.read_csv(data_path)
    train = frame[frame["split"] == "train"].copy()
    evaluated = frame[frame["split"] == config["evaluation_split"]].copy()
    if train.empty or evaluated.empty:
        raise ValueError("train/evaluation split missing")

    rules = provenance_rules(
        config["thresholds"],
        train,
        model_seed=derive_seed(int(config["seed"]), "provenance-learned-gate"),
    )
    access = audit_rules(rules, "provenance")
    invalid = access[access["status"] == "INVALID"]
    if not invalid.empty:
        raise RuntimeError(f"invalid provenance features: {invalid[['rule', 'illegal_features']].to_dict('records')}")

    decision_frames: list[pd.DataFrame] = []
    bounds_rows: list[dict[str, object]] = []
    for rule in rules:
        decisions = rule.decide(evaluated, config["thresholds"])
        ruled = with_laundering_outcomes(evaluated, decisions)
        ruled.insert(0, "rule", rule.name)
        decision_frames.append(ruled)
        bounds = cluster_bootstrap_bounds(
            ruled,
            replicates=int(config["bootstrap_replicates"]),
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
        decisions[decisions["attack_type"] == "role_noise"], ["rule", "role_noise_rate"]
    )
    by_hop = _group_metrics(
        decisions[decisions["attack_type"] == "multi_hop"], ["rule", "hop_depth", "traceability"]
    )
    bounds = pd.DataFrame(bounds_rows)

    results_dir = ROOT / config["results_dir"]
    results_dir.mkdir(parents=True, exist_ok=True)
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
    for name, output in outputs.items():
        output.to_csv(results_dir / name, index=False)

    learned = next(rule for rule in rules if rule.name == "Provenance Learned Gate")
    model_metadata_path = results_dir / "learned_gate_metadata.json"
    model_metadata_path.write_text(
        json.dumps(learned.metadata(), indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path = results_dir / "report.md"
    report_path.write_text(
        _render_report(config, summary, by_traceability), encoding="utf-8"
    )

    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.2.0",
        "mode": config["mode"],
        "evaluation_split": config["evaluation_split"],
        "confirmatory": False,
        "rows_evaluated": len(evaluated),
        "clusters_evaluated": int(evaluated["event_cluster_id"].nunique()),
        "bootstrap_unit": "event_cluster_id",
        "bootstrap_replicates": int(config["bootstrap_replicates"]),
        "outputs": {
            **{name: sha256(results_dir / name) for name in outputs},
            report_path.name: sha256(report_path),
            model_metadata_path.name: sha256(model_metadata_path),
        },
        "data_sha256": sha256(data_path),
        "config_sha256": sha256(config_path),
        "claim_boundary": "Validation-only controlled provenance audit; no deployment, observed-institution, or test claim.",
    }
    (results_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return results_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Run provenance laundering validation audit.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "provenance_smoke.yaml"))
    args = parser.parse_args()
    out = run(Path(args.config))
    print(out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

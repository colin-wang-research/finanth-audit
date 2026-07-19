from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def finite_spearman(left: pd.Series, right: pd.Series) -> float | None:
    valid = np.isfinite(left.to_numpy(dtype=float)) & np.isfinite(
        right.to_numpy(dtype=float)
    )
    if valid.sum() < 2:
        return None
    value = float(spearmanr(left[valid], right[valid]).statistic)
    return value if np.isfinite(value) else None


def _controlled_stability() -> pd.DataFrame:
    smoke = pd.read_csv(ROOT / "results" / "smoke" / "raw_vs_certified_ranking.csv")
    main = pd.read_csv(ROOT / "results" / "main" / "raw_vs_certified_ranking.csv")
    columns = [
        "rule",
        "coverage",
        "far",
        "alr",
        "cau",
        "worst_profile_certification_volume",
        "coverage_collapse_index",
    ]
    merged = smoke[columns].merge(
        main[columns], on="rule", suffixes=("_smoke", "_main"), validate="one_to_one"
    )
    for metric in columns[1:]:
        merged[f"{metric}_delta_main_minus_smoke"] = (
            merged[f"{metric}_main"] - merged[f"{metric}_smoke"]
        )
    return merged.sort_values("rule").reset_index(drop=True)


def _provenance_stability() -> pd.DataFrame:
    smoke = pd.read_csv(ROOT / "results" / "provenance_smoke" / "summary.csv")
    main = pd.read_csv(ROOT / "results" / "provenance_main" / "summary.csv")
    columns = [
        "rule",
        "coverage",
        "far",
        "alr",
        "indirect_leakage_rate",
        "safe_delegation_coverage",
        "false_block_rate",
        "cau",
    ]
    merged = smoke[columns].merge(
        main[columns], on="rule", suffixes=("_smoke", "_main"), validate="one_to_one"
    )
    for metric in columns[1:]:
        merged[f"{metric}_delta_main_minus_smoke"] = (
            merged[f"{metric}_main"] - merged[f"{metric}_smoke"]
        )
    return merged.sort_values("rule").reset_index(drop=True)


def _format(value: object, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "N/A"
    return f"{float(value):.{digits}f}"


def run(output_dir: Path) -> Path:
    output_dir = output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    controlled = _controlled_stability()
    provenance = _provenance_stability()
    controlled.to_csv(output_dir / "controlled_smoke_main_stability.csv", index=False)
    provenance.to_csv(output_dir / "provenance_smoke_main_stability.csv", index=False)

    stability = {
        "controlled_far_spearman": finite_spearman(
            controlled["far_smoke"], controlled["far_main"]
        ),
        "controlled_certification_spearman": finite_spearman(
            controlled["worst_profile_certification_volume_smoke"],
            controlled["worst_profile_certification_volume_main"],
        ),
        "provenance_far_spearman": finite_spearman(
            provenance["far_smoke"], provenance["far_main"]
        ),
        "provenance_alr_spearman": finite_spearman(
            provenance["alr_smoke"], provenance["alr_main"]
        ),
    }
    (output_dir / "scale_stability.json").write_text(
        json.dumps(stability, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    controlled_main = pd.read_csv(
        ROOT / "results" / "main" / "raw_vs_certified_ranking.csv"
    ).set_index("rule")
    provenance_main = pd.read_csv(
        ROOT / "results" / "provenance_main" / "summary.csv"
    ).set_index("rule")
    traceability = pd.read_csv(
        ROOT / "results" / "provenance_main" / "by_traceability.csv"
    )
    hard_untraceable = traceability[
        (traceability["rule"] == "Provenance Hard Gate")
        & (traceability["traceability"] == "untraceable")
    ].iloc[0]

    lifecycle = controlled_main.loc["Lifecycle Checklist"]
    cost = controlled_main.loc["Cost-Aware Gate"]
    hard_role = provenance_main.loc["Hard Role Gate"]
    provenance_hard = provenance_main.loc["Provenance Hard Gate"]
    provenance_learned = provenance_main.loc["Provenance Learned Gate"]
    epv = provenance_main.loc["EPV Adapter"]
    report = [
        "# Main-Scale Validation Report",
        "",
        "This report covers 200,000 controlled rows and 200,000 controlled-provenance rows. All metrics use validation clusters only. Test outcomes remain sealed.",
        "",
        "## Scale",
        "",
        "- Controlled: 200,000 rows, 50,000 event clusters, 40,000 validation rows, 10,000 validation clusters.",
        "- Provenance: 200,000 rows, 50,000 event clusters, 40,000 validation rows, 10,000 validation clusters.",
        "- Bootstrap: 5,000 event-cluster replicates per rule/profile.",
        "",
        "## Main findings",
        "",
        (
            f"**Finding 1: Raw FAR and certification answer different questions.** "
            f"Cost-Aware Gate has FAR={_format(cost['far'])} at coverage="
            f"{_format(cost['coverage'])}, but its worst-profile certification volume is "
            f"{_format(cost['worst_profile_certification_volume'])} because authorization "
            f"leakage remains nonzero. Lifecycle Checklist has FAR={_format(lifecycle['far'])}, "
            f"coverage={_format(lifecycle['coverage'])}, ALR={_format(lifecycle['alr'])}, and "
            f"the only nonzero worst-profile certification volume="
            f"{_format(lifecycle['worst_profile_certification_volume'])}. This is a "
            "multi-objective finding, not a globally optimal rule claim."
        ),
        "",
        (
            f"**Finding 2: Clean current-role gating does not prevent indirect laundering.** "
            f"Hard Role Gate has direct leakage={_format(hard_role['direct_leakage_rate'])} "
            f"but indirect leakage and ALR={_format(hard_role['indirect_leakage_rate'])}."
        ),
        "",
        (
            f"**Finding 3: Provenance integrity and authorization quality trade off.** "
            f"Provenance Hard Gate has ALR={_format(provenance_hard['alr'])} but FAR="
            f"{_format(provenance_hard['far'])}; on untraceable rows its coverage is "
            f"{_format(hard_untraceable['coverage'])} and FAR is N/A. Provenance Learned "
            f"Gate retains safe-delegation coverage="
            f"{_format(provenance_learned['safe_delegation_coverage'])} with residual ALR="
            f"{_format(provenance_learned['alr'])} and FAR="
            f"{_format(provenance_learned['far'])}."
        ),
        "",
        (
            f"**Finding 4: Low FAR alone does not establish provenance safety.** EPV Adapter "
            f"has FAR={_format(epv['far'])} at coverage={_format(epv['coverage'])}, but ALR="
            f"{_format(epv['alr'])}. It remains an ordinary baseline rather than the "
            "benchmark identity or a method winner."
        ),
        "",
        "## Smoke-to-main stability",
        "",
        f"- Controlled FAR Spearman: {_format(stability['controlled_far_spearman'])}.",
        f"- Controlled certification-volume Spearman: {_format(stability['controlled_certification_spearman'])}.",
        f"- Provenance FAR Spearman: {_format(stability['provenance_far_spearman'])}.",
        f"- Provenance ALR Spearman: {_format(stability['provenance_alr_spearman'])}.",
        "",
        "These correlations are descriptive scale-stability diagnostics, not independent hypothesis tests, because smoke and main use the same frozen generator family with different seeds and sizes.",
        "",
        "## Boundary",
        "",
        "The controlled layers model authorization failures, not price paths, market profitability, deployed agents, institutional decisions, or investment advice. No test result is reported.",
        "",
    ]
    report_path = output_dir / "report.md"
    report_path.write_text("\n".join(report), encoding="utf-8")
    outputs = (
        "controlled_smoke_main_stability.csv",
        "provenance_smoke_main_stability.csv",
        "scale_stability.json",
        "report.md",
    )
    manifest = {
        "project": "FinAuth-Audit",
        "version": "0.2.0",
        "mode": "main-validation-summary",
        "confirmatory": False,
        "test_outcomes_evaluated": False,
        "outputs": {name: sha256(output_dir / name) for name in outputs},
        "inputs": {
            "results/smoke/manifest.json": sha256(ROOT / "results" / "smoke" / "manifest.json"),
            "results/main/manifest.json": sha256(ROOT / "results" / "main" / "manifest.json"),
            "results/provenance_smoke/manifest.json": sha256(
                ROOT / "results" / "provenance_smoke" / "manifest.json"
            ),
            "results/provenance_main/manifest.json": sha256(
                ROOT / "results" / "provenance_main" / "manifest.json"
            ),
        },
        "claim_boundary": "Validation-only smoke-to-main stability and main-scale findings; no test, deployment, or method-victory claim.",
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(output_dir)
    return output_dir


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize main-scale validation evidence.")
    parser.add_argument(
        "--output-dir", default=str(ROOT / "results" / "main_validation")
    )
    args = parser.parse_args()
    run(Path(args.output_dir))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
